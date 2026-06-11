"""
Bedrock対応の BasicExtractor。
CurateGPT の BasicExtractor を継承し、model プロパティを
Amazon Bedrock (litellm経由) に差し替える。
"""

import contextlib
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from threading import Lock

import litellm
from pydantic import ConfigDict

from metadata_support.extract.basic_extractor import BasicExtractor

logger = logging.getLogger(__name__)

# AWS Bedrock 1Mトークンあたりの料金 (USD)
MODEL_PRICING = {
    "bedrock/jp.anthropic.claude-haiku-4-5-20251001-v1:0": {"input": 1.00, "output": 5.00},
    "bedrock/jp.anthropic.claude-sonnet-4-5-20250929-v1:0": {"input": 3.00, "output": 15.00},
    "bedrock/jp.anthropic.claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
}

# 用途ラベル(purpose)の既知一覧（順序保持）。未知のpurposeも記録される。
KNOWN_PURPOSES = (
    "survey_name",
    "annotate",
    "rerank_external",
    "vocab_convert",
    "field_extract",
    "unknown",
)

# (model_id, purpose) ごとのトークン使用量（モジュールレベルで共有・スレッドセーフ）
_usage_lock = Lock()
_token_usage: dict[tuple[str, str], dict[str, int]] = defaultdict(
    lambda: {"input": 0, "output": 0, "calls": 0}
)

# スレッドローカル: BasicExtractor 経由など、prompt() に直接 purpose を渡せない経路用に
# 「現在のpurpose」を保持する。purpose_scope() で設定する。
# また直近呼び出しの実トークン使用量を保持し、呼び出し側で読み取れるようにする。
_thread_state = threading.local()


def _compute_cost(model_id: str, input_tokens: int, output_tokens: int) -> tuple[float, float]:
    """指定モデルのinput/outputトークンに対する料金(USD) (input_cost, output_cost) を返す。"""
    pricing = MODEL_PRICING.get(model_id, {"input": 0.0, "output": 0.0})
    return (
        input_tokens / 1_000_000 * pricing["input"],
        output_tokens / 1_000_000 * pricing["output"],
    )


def reset_token_usage():
    """全モデル・全用途の累計トークン使用量をリセットする。"""
    with _usage_lock:
        _token_usage.clear()


def get_token_usage():
    """(model_id, purpose) 単位の累計トークン使用量のスナップショットを返す。"""
    with _usage_lock:
        return {key: dict(v) for key, v in _token_usage.items()}


def calculate_cost(usage=None):
    """料金を計算する。戻り値は (合計USD, モデル別内訳dict)。

    モデル別内訳には用途別の breakdown ("by_purpose") も含む。
    """
    if usage is None:
        usage = get_token_usage()
    total_cost = 0.0
    breakdown: dict[str, dict] = {}
    for (model_id, purpose), tokens in usage.items():
        input_cost, output_cost = _compute_cost(model_id, tokens["input"], tokens["output"])
        model_entry = breakdown.setdefault(
            model_id,
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "calls": 0,
                "input_cost_usd": 0.0,
                "output_cost_usd": 0.0,
                "total_cost_usd": 0.0,
                "by_purpose": {},
            },
        )
        model_entry["input_tokens"] += tokens["input"]
        model_entry["output_tokens"] += tokens["output"]
        model_entry["calls"] += tokens["calls"]
        model_entry["input_cost_usd"] += input_cost
        model_entry["output_cost_usd"] += output_cost
        model_entry["total_cost_usd"] += input_cost + output_cost
        model_entry["by_purpose"][purpose] = {
            "input_tokens": tokens["input"],
            "output_tokens": tokens["output"],
            "calls": tokens["calls"],
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "total_cost_usd": input_cost + output_cost,
        }
        total_cost += input_cost + output_cost
    return total_cost, breakdown


def calculate_cost_by_purpose(usage=None):
    """用途別の集計を返す。戻り値は (合計USD, 用途別内訳dict)。

    用途別内訳には呼び出されたモデル別の breakdown ("by_model") も含む。
    """
    if usage is None:
        usage = get_token_usage()
    total_cost = 0.0
    breakdown: dict[str, dict] = {}
    for (model_id, purpose), tokens in usage.items():
        input_cost, output_cost = _compute_cost(model_id, tokens["input"], tokens["output"])
        purpose_entry = breakdown.setdefault(
            purpose,
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "calls": 0,
                "input_cost_usd": 0.0,
                "output_cost_usd": 0.0,
                "total_cost_usd": 0.0,
                "by_model": {},
            },
        )
        purpose_entry["input_tokens"] += tokens["input"]
        purpose_entry["output_tokens"] += tokens["output"]
        purpose_entry["calls"] += tokens["calls"]
        purpose_entry["input_cost_usd"] += input_cost
        purpose_entry["output_cost_usd"] += output_cost
        purpose_entry["total_cost_usd"] += input_cost + output_cost
        purpose_entry["by_model"][model_id] = {
            "input_tokens": tokens["input"],
            "output_tokens": tokens["output"],
            "calls": tokens["calls"],
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "total_cost_usd": input_cost + output_cost,
        }
        total_cost += input_cost + output_cost
    return total_cost, breakdown


@contextlib.contextmanager
def purpose_scope(purpose: str):
    """ブロック内で発生する LLM 呼び出しの purpose を一時的に設定するコンテキストマネージャ。

    BasicExtractor 経由のように prompt() に直接 purpose を渡せない経路で使用する。
    入れ子になった場合は内側の purpose が優先される。
    """
    prev = getattr(_thread_state, "purpose", None)
    _thread_state.purpose = purpose
    try:
        yield
    finally:
        _thread_state.purpose = prev


def get_last_call_usage():
    """直近の LLM 呼び出しの実トークン/料金 (このスレッド内) を返す。

    戻り値: dict {input_tokens, output_tokens, input_cost_usd, output_cost_usd,
                  total_cost_usd, model_id, purpose} または未呼び出しなら None。
    """
    return getattr(_thread_state, "last_usage", None)


class _BedrockResponse:
    """llm ライブラリの Response と同じインターフェースを提供するラッパー。

    実トークン数と料金を `usage` 属性で公開する。
    """

    def __init__(self, content: str, usage: dict | None = None):
        self._content = content
        self.usage = usage or {
            "input_tokens": 0,
            "output_tokens": 0,
            "input_cost_usd": 0.0,
            "output_cost_usd": 0.0,
            "total_cost_usd": 0.0,
            "model_id": "",
            "purpose": "unknown",
        }

    def text(self) -> str:
        return self._content


class _BedrockModel:
    """llm ライブラリの Model と同じインターフェースを提供するラッパー。
    内部で litellm.completion() を使って Bedrock を呼び出す。
    """

    needs_key = False
    key_env_var = None

    def __init__(self, model_name: str, aws_region: str = "ap-northeast-1"):
        self.model_id = model_name
        self._litellm_model = model_name
        self._aws_region = aws_region
        self._max_retries = 3
        self._retry_base_delay = 10

    def prompt(self, text: str, system: str = None, purpose: str = None) -> _BedrockResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": text})

        # purpose の決定: 明示指定 > スレッドローカル(purpose_scope) > "unknown"
        effective_purpose = (
            purpose
            or getattr(_thread_state, "purpose", None)
            or "unknown"
        )

        for attempt in range(self._max_retries):
            try:
                response = litellm.completion(
                    model=self._litellm_model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=16384,
                    aws_region_name=self._aws_region,
                )
                usage_obj = getattr(response, "usage", None)
                input_tokens = (getattr(usage_obj, "prompt_tokens", 0) or 0) if usage_obj else 0
                output_tokens = (getattr(usage_obj, "completion_tokens", 0) or 0) if usage_obj else 0
                input_cost, output_cost = _compute_cost(self.model_id, input_tokens, output_tokens)
                total_cost = input_cost + output_cost
                if usage_obj is not None:
                    with _usage_lock:
                        bucket = _token_usage[(self.model_id, effective_purpose)]
                        bucket["input"] += input_tokens
                        bucket["output"] += output_tokens
                        bucket["calls"] += 1
                logger.info(
                    "LLM_CALL | purpose=%s | model=%s | input=%d | output=%d | "
                    "input_cost=$%.4f | output_cost=$%.4f | total=$%.4f",
                    effective_purpose, self.model_id, input_tokens, output_tokens,
                    input_cost, output_cost, total_cost,
                )
                call_usage = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "input_cost_usd": input_cost,
                    "output_cost_usd": output_cost,
                    "total_cost_usd": total_cost,
                    "model_id": self.model_id,
                    "purpose": effective_purpose,
                }
                _thread_state.last_usage = call_usage
                return _BedrockResponse(response.choices[0].message.content, usage=call_usage)
            except litellm.RateLimitError:
                if attempt < self._max_retries - 1:
                    delay = self._retry_base_delay * (2 ** attempt)
                    logger.warning(
                        "Rate limit hit, retrying in %ds (attempt %d/%d)",
                        delay, attempt + 1, self._max_retries,
                    )
                    time.sleep(delay)
                else:
                    raise


@dataclass
class BedrockExtractor(BasicExtractor):
    """
    BasicExtractor を継承し、LLM 呼び出しを Amazon Bedrock に差し替えた Extractor。

    使い方:
        extractor = BedrockExtractor(
            model_name="bedrock/jp.anthropic.claude-sonnet-4-5-20250929-v1:0",
            aws_region="ap-northeast-1",
        )
    """

    model_config = ConfigDict(protected_namespaces=())

    model_name: str = "bedrock/jp.anthropic.claude-sonnet-4-5-20250929-v1:0"
    aws_region: str = "ap-northeast-1"

    @property
    def model(self):
        if self._model is None:
            self._model = _BedrockModel(
                model_name=self.model_name,
                aws_region=self.aws_region,
            )
        return self._model
