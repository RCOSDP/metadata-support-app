"""Bedrock Claude を用いて英語 description を日本語の短い説明文に整形する。"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import litellm

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "bedrock/jp.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_REGION = "ap-northeast-1"
DEFAULT_LOG_PATH = (
    Path(__file__).resolve().parents[4]
    / "input_documents" / "def_item" / "ddi_description_llm_log.txt"
)

_USER_PROMPT_TEMPLATE = """## タスク
社会科学データのメタデータ項目について、調査報告書・調査票からの情報抽出に使用する「定義文」を日本語で生成する。

## 定義文の要件
- 根拠は各項目の en_description のみ。外部知識・慣例による補足は禁止。
- 「その項目が何であるか」を1文で述べる叙述形式（〜である／〜を指す）。
  操作指示（〜を記述する／〜を入力する）は不可。
- 調査報告書・調査票を読んだ人が「どこを見ればよいか」が判断できる粒度にする。
- 冗長な例示・DDI仕様内部のメタ情報（パスや属性名）は省略する。
- 自然な日本語の1文。20〜50字を目安とするが、意味の正確さを優先する。
- マークアップ・括弧の多用・URLは含めない。
- 必ず「各項目のルール」を定義に含めること

## 各項目のルール
- タイトル: 調査の名称
- その他のタイトル: タイトル以外の名称
- 調査機器: 調査に使用された機器やツール
- 対象時期、調査日: 調査が行われた期間
- データタイプ: 調査で収集されたデータの種類・形式

## 出力形式
以下のJSONのみ返すこと。前置き・後置き・コードフェンス外のテキスト不要。

{{
  "タイトル": "...",
  "その他のタイトル": "...",
  ...
}}

## 入力データ
{payload}
"""


def generate_japanese_descriptions(
    en_data: dict[str, dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    aws_region: str = DEFAULT_REGION,
    max_tokens: int = 8000,
    log_path: Path | None = DEFAULT_LOG_PATH,
) -> dict[str, dict[str, Any]]:
    """en_data に ja_description を付与した辞書を返す (破壊しない)。

    log_path が指定されている場合、LLMのプロンプト・レスポンスをテキストファイルに保存する。
    """
    payload_items = [
        {
            "name": name,
            "leaf": v["leaf"],
            "en_title": v.get("en_title", ""),
            "en_description": v.get("en_description", ""),
        }
        for name, v in en_data.items()
    ]
    user_prompt = _USER_PROMPT_TEMPLATE.format(
        payload=json.dumps(payload_items, ensure_ascii=False, indent=2)
    )

    logger.info("Calling %s for %d items", model, len(payload_items))
    resp = litellm.completion(
        model=model,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
        aws_region_name=aws_region,
    )
    content = resp.choices[0].message.content

    if log_path is not None:
        try:
            _write_log_file(log_path, model, resp, user_prompt, content, len(payload_items))
        except Exception as e:
            logger.warning("description_generator log 出力失敗: %s", e)

    ja_map = _parse_json_block(content)

    missing = [name for name in en_data if name not in ja_map]
    if missing:
        logger.warning("LLM response missing %d items: %s", len(missing), missing)

    merged: dict[str, dict[str, Any]] = {}
    for name, v in en_data.items():
        merged[name] = {**v, "ja_description": ja_map.get(name, "")}
    return merged


def _write_log_file(
    log_path: Path,
    model: str,
    resp: Any,
    user_prompt: str,
    response_content: str,
    item_count: int,
) -> None:
    usage = getattr(resp, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    sep = "-" * 40
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"【description_generator LLM ログ】 項目数: {item_count} トークン数: 入力={input_tokens} / 出力={output_tokens}",
        f"日時: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"モデル: {model}",
        sep,
        "【ユーザープロンプト】",
        user_prompt,
        sep,
        "【LLM生レスポンス】",
        response_content,
        sep,
    ]
    log_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("description_generator LLM ログを出力: %s", log_path)


def _parse_json_block(text: str) -> dict[str, str]:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    raw = m.group(1) if m else text
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"JSON object not found in LLM response: {text[:300]}")
    return json.loads(raw[start : end + 1])
