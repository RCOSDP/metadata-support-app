"""Bedrock Claude を用いて METADATA_FIELDS を意味的にカテゴライズする。

LLM は項目名と説明文を受け取り、調査報告書・調査票の理解に有用な単位で
グルーピングを行う。結果は JSON にキャッシュし、入力（項目集合）が変化しない
限り再呼び出しは発生しない。
"""
from __future__ import annotations

import hashlib
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
DEFAULT_CACHE_PATH = (
    Path(__file__).resolve().parents[4]
    / "input_documents" / "def_item" / "ddi_field_groups.json"
)
DEFAULT_LOG_PATH = (
    Path(__file__).resolve().parents[4]
    / "input_documents" / "def_item" / "ddi_field_groups_llm_log.txt"
)

_USER_PROMPT_TEMPLATE = """## タスク
社会科学データのメタデータ項目（合計{n}項目）を、意味的なまとまりごとに4〜6個のカテゴリに分類する。

## 分類のルール
- 各カテゴリは「調査報告書・調査票を読み解く際の関心領域」が同じ項目をまとめる。
  例: 書誌情報、調査の設計や方法、トピックやデータタイプ、データ提供条件、外部参照情報 など。
- カテゴリ名は日本語で簡潔（10字以内）。
- 1項目は必ず1カテゴリに属する。重複は禁止。
- 全{n}項目を漏れなく分類する。
- 入力された項目名（"name" の値）はそのままの文字列で返すこと。改変・要約禁止。
- カテゴリ間の項目数が極端に偏らないようにする（1カテゴリ最大15項目目安）。

## 出力形式
以下のJSONのみ返すこと。前置き・後置き・コードフェンス外のテキスト不要。

{{
  "groups": [
    {{
      "name": "<カテゴリ名>",
      "fields": ["<項目名>", "<項目名>", ...]
    }},
    ...
  ]
}}

## 入力データ
{payload}
"""


def generate_field_groups(
    metadata_fields: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    aws_region: str = DEFAULT_REGION,
    max_tokens: int = 4000,
    log_path: Path | None = DEFAULT_LOG_PATH,
) -> list[dict]:
    """metadata_fields をLLMでカテゴライズし、グループのリストを返す。

    戻り値: [{"name": <カテゴリ名>, "fields": [<項目名>, ...]}]
    """
    payload_items = [
        {"name": f.get("name", ""), "desc": f.get("desc", "")}
        for f in metadata_fields
    ]
    user_prompt = _USER_PROMPT_TEMPLATE.format(
        n=len(payload_items),
        payload=json.dumps(payload_items, ensure_ascii=False, indent=2),
    )

    logger.info("Calling %s for field grouping (%d items)", model, len(payload_items))
    resp = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
        aws_region_name=aws_region,
    )
    content = resp.choices[0].message.content

    if log_path is not None:
        try:
            _write_log_file(log_path, model, resp, user_prompt, content, len(payload_items))
        except Exception as e:
            logger.warning("field_groups_generator log 出力失敗: %s", e)

    parsed = _parse_json_block(content)
    groups = parsed.get("groups", [])
    return _validate_and_complete(groups, metadata_fields)


def fields_signature(metadata_fields: list[dict]) -> str:
    """項目集合の署名（順序込みのハッシュ）。キャッシュ整合性チェックに用いる。"""
    payload = [
        {"name": f.get("name", ""), "desc": f.get("desc", "")}
        for f in metadata_fields
    ]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_cached_groups(cache_path: Path, signature: str) -> list[dict] | None:
    """キャッシュが署名と一致すれば groups を返す。一致しない/ない場合 None。"""
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("field_groups cache 読み込み失敗: %s", e)
        return None
    if data.get("signature") != signature:
        return None
    groups = data.get("groups")
    if not isinstance(groups, list):
        return None
    return groups


def save_cached_groups(cache_path: Path, signature: str, groups: list[dict]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {"signature": signature, "groups": groups},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )


def _validate_and_complete(
    groups: list[dict], metadata_fields: list[dict],
) -> list[dict]:
    """LLMが返したグループを検証し、欠落項目を「その他」に補完する。"""
    valid_names = {f["name"] for f in metadata_fields}
    seen: set[str] = set()
    cleaned: list[dict] = []
    for g in groups:
        gname = (g.get("name") or "").strip() or "未分類"
        gfields_raw = g.get("fields") or []
        gfields = []
        for n in gfields_raw:
            if not isinstance(n, str):
                continue
            n = n.strip()
            if n in valid_names and n not in seen:
                seen.add(n)
                gfields.append(n)
        if gfields:
            cleaned.append({"name": gname, "fields": gfields})
    missing = [f["name"] for f in metadata_fields if f["name"] not in seen]
    if missing:
        logger.warning(
            "LLM grouping missing %d items, putting into 'その他': %s",
            len(missing), missing,
        )
        cleaned.append({"name": "その他", "fields": missing})
    return cleaned


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
        f"【field_groups_generator LLM ログ】 項目数: {item_count} "
        f"トークン数: 入力={input_tokens} / 出力={output_tokens}",
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
    logger.info("field_groups_generator LLM ログを出力: %s", log_path)


def _parse_json_block(text: str) -> dict[str, Any]:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    raw = m.group(1) if m else text
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"JSON object not found in LLM response: {text[:300]}")
    return json.loads(raw[start : end + 1])
