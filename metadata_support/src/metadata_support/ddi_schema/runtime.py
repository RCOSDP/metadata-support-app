"""app 起動時に METADATA_FIELDS の desc を xsd 由来の説明で上書きするためのローダー。

キャッシュポリシー:
- ddi_descriptions_ja.json が存在 かつ codebook.xsd より新しい   -> JSONをそのままロード
- ddi_descriptions_ja.json が無い または xsd より古い (=xsd更新) -> 自動で再生成
  * 再生成時は xsd → 英語抽出 → Bedrock Claude で日本語整形 → JSON保存

一度生成すれば次回以降の起動は JSON ロードのみで完了するため、LLM呼び出しは発生しない。
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from metadata_support.ddi_schema.description_generator import generate_japanese_descriptions
from metadata_support.ddi_schema.field_groups_generator import (
    DEFAULT_CACHE_PATH as DEFAULT_GROUPS_CACHE_PATH,
    fields_signature,
    generate_field_groups,
    load_cached_groups,
    save_cached_groups,
)
from metadata_support.ddi_schema.xsd_parser import FIELDS, extract_english_descriptions

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_DEF_ITEM_DIR = _PROJECT_ROOT / "input_documents" / "def_item"
DEFAULT_XSD_PATH = _DEF_ITEM_DIR / "codebook.xsd"
DEFAULT_EN_JSON_PATH = _DEF_ITEM_DIR / "ddi_descriptions_en.json"
DEFAULT_JA_JSON_PATH = _DEF_ITEM_DIR / "ddi_descriptions_ja.json"

_SKIP_ENV = "DDI_SKIP_AUTO_BUILD"


def load_or_build_descriptions(
    *,
    xsd_path: Path = DEFAULT_XSD_PATH,
    en_json_path: Path = DEFAULT_EN_JSON_PATH,
    ja_json_path: Path = DEFAULT_JA_JSON_PATH,
    force_rebuild: bool = False,
) -> dict[str, str]:
    """項目名 -> 日本語説明 の辞書を返す。キャッシュ無ければ自動生成。

    環境変数 DDI_SKIP_AUTO_BUILD=1 を指定すると JSON 未生成時でも生成を試みず空dictを返す
    (オフライン環境や AWS 認証無しで app を起動したい場合のエスケープハッチ)。
    """
    if not force_rebuild and _is_cache_fresh(xsd_path, ja_json_path):
        return _load_ja_map(ja_json_path)

    if os.getenv(_SKIP_ENV):
        logger.warning(
            "%s set; skipping xsd-based description build. "
            "METADATA_FIELDS はハードコードされた desc のみで起動します。",
            _SKIP_ENV,
        )
        if ja_json_path.exists():
            return _load_ja_map(ja_json_path)
        return {}

    logger.info(
        "description cache miss (force_rebuild=%s, ja_exists=%s). "
        "xsd から説明を自動生成します...",
        force_rebuild, ja_json_path.exists(),
    )
    try:
        xsd_text = xsd_path.read_text(encoding="utf-8")
        en_data = extract_english_descriptions(xsd_text)
        en_json_path.parent.mkdir(parents=True, exist_ok=True)
        en_json_path.write_text(
            json.dumps(en_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        ja_data = generate_japanese_descriptions(en_data)
        ja_json_path.write_text(
            json.dumps(ja_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("生成完了: %s (%d件)", ja_json_path, len(ja_data))
        return {
            name: rec.get("ja_description", "")
            for name, rec in ja_data.items()
        }
    except Exception as e:  # LLM失敗・認証切れ等
        logger.error("xsd-based description build failed: %s", e)
        if ja_json_path.exists():
            logger.warning("既存の ja JSON にフォールバックします: %s", ja_json_path)
            return _load_ja_map(ja_json_path)
        return {}


def build_metadata_fields(
    *,
    xsd_path: Path = DEFAULT_XSD_PATH,
    en_json_path: Path = DEFAULT_EN_JSON_PATH,
    ja_json_path: Path = DEFAULT_JA_JSON_PATH,
    force_rebuild: bool = False,
    use_llm_summary: bool = True,
) -> list[dict]:
    """xsd_parser.FIELDS の構造情報 (no/name/ddi_path) と xsd 由来の説明を合成し、
    旧 METADATA_FIELDS と同形式の list[dict] を返す。

    説明の優先順位: ja_description (LLM整形済) > en_description (xsd原文)> 空文字列。
    LLM失敗や AWS 認証切れで ja が取れない場合でも、en (xsd原文) がフォールバックになるため
    app は必ず起動できる。

    use_llm_summary=False の場合は ja_description の読み込み・生成をスキップし、
    xsd 原文 (en_description) のみを使用する。
    """
    if use_llm_summary:
        ja_descs = load_or_build_descriptions(
            xsd_path=xsd_path,
            en_json_path=en_json_path,
            ja_json_path=ja_json_path,
            force_rebuild=force_rebuild,
        )
    else:
        ja_descs = {}
        logger.info("use_llm_summary=False: xsd原文(en_description)のみ使用")
    en_descs = _load_en_description_fallback(en_json_path)
    result: list[dict] = []
    for no, name, ddi_path, _leaf, _current_desc in FIELDS:
        desc = (ja_descs.get(name, "") or "").strip()
        if not desc:
            desc = (en_descs.get(name, "") or "").strip()
        result.append({"no": no, "name": name, "ddi": ddi_path, "desc": desc})
    logger.info("build_metadata_fields: %d項目を構築", len(result))
    return result


def _load_en_description_fallback(en_json_path: Path) -> dict[str, str]:
    if not en_json_path.exists():
        return {}
    try:
        data = json.loads(en_json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {
        name: rec.get("en_description", "") if isinstance(rec, dict) else ""
        for name, rec in data.items()
    }


_GROUP_LABELS: dict[str, str] = {
    "citation": "引用情報",
    "stdyInfo": "調査情報",
    "method": "調査方法",
    "dataAccs": "データアクセス",
    "othrStdyMat": "関連資料",
    "holdings": "保管情報",
}


def build_field_groups(
    metadata_fields: list[dict],
    *,
    cache_path: Path = DEFAULT_GROUPS_CACHE_PATH,
    force_rebuild: bool = False,
) -> list[dict]:
    """LLM で metadata_fields を意味的にカテゴライズしたフィールドグループを返す。

    キャッシュ (``cache_path``) が現在の項目集合と一致すればキャッシュを再利用し、
    LLM呼び出しは発生しない。一致しない場合はLLMでグルーピングを再生成する。
    LLM呼び出しが失敗した場合は xsd 階層由来の機械的グルーピングにフォールバックする。

    返り値の各要素は config.FIELD_GROUPS 互換:
      {"name": <日本語ラベル>, "fields": [<項目名>, ...], "fields_str": <bullet形式>}
    """
    desc_map = {f["name"]: f["desc"] for f in metadata_fields}
    signature = fields_signature(metadata_fields)

    groups_data: list[dict] | None = None
    if not force_rebuild:
        groups_data = load_cached_groups(cache_path, signature)
        if groups_data is not None:
            logger.info("field_groups: キャッシュから読み込み (%d groups)", len(groups_data))

    if groups_data is None:
        if os.getenv(_SKIP_ENV):
            logger.warning(
                "%s set; LLM grouping をスキップし、xsd 階層由来の機械的グルーピングを使用",
                _SKIP_ENV,
            )
            groups_data = _build_field_groups_mechanical(metadata_fields)
        else:
            try:
                groups_data = generate_field_groups(metadata_fields)
                save_cached_groups(cache_path, signature, groups_data)
                logger.info(
                    "field_groups: LLM で再生成 (%d groups), キャッシュ保存: %s",
                    len(groups_data), cache_path,
                )
            except Exception as e:
                logger.error(
                    "field_groups: LLM 呼び出し失敗 (%s)、機械的グルーピングへフォールバック",
                    e,
                )
                groups_data = _build_field_groups_mechanical(metadata_fields)

    groups: list[dict] = []
    for g in groups_data:
        names = g.get("fields", [])
        label = g.get("name", "")
        fields_str = "\n".join(
            f"- **{name}**: {desc_map.get(name, '')}" for name in names
        )
        groups.append({"name": label, "fields": names, "fields_str": fields_str})
    return groups


def _build_field_groups_mechanical(metadata_fields: list[dict]) -> list[dict]:
    """codebook.xsd の階層 (stdyDscr 直下) でグルーピングしたフォールバック実装。"""
    grouped: dict[str, list[str]] = {}
    order: list[str] = []
    for f in metadata_fields:
        key = _classify_group_key(f.get("ddi", ""))
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(f["name"])

    return [
        {"name": _GROUP_LABELS.get(key, key), "fields": grouped[key]}
        for key in order
    ]


def _classify_group_key(ddi_path: str) -> str:
    parts = [p for p in ddi_path.split("->") if p]
    if not parts:
        return "その他"
    if parts[0] == "stdyDscr":
        return parts[1] if len(parts) >= 2 else "stdyDscr"
    return parts[0]


def apply_to_metadata_fields(
    metadata_fields: list[dict],
    *,
    xsd_path: Path = DEFAULT_XSD_PATH,
    en_json_path: Path = DEFAULT_EN_JSON_PATH,
    ja_json_path: Path = DEFAULT_JA_JSON_PATH,
    force_rebuild: bool = False,
) -> list[dict]:
    """METADATA_FIELDS の各 entry の `desc` を ja_description で上書きする (in-place)。

    ja_description が空 or キー無しの項目は既存の desc をそのまま保持する (フォールバック)。
    """
    descs = load_or_build_descriptions(
        xsd_path=xsd_path,
        en_json_path=en_json_path,
        ja_json_path=ja_json_path,
        force_rebuild=force_rebuild,
    )
    if not descs:
        return metadata_fields
    applied = 0
    for entry in metadata_fields:
        new_desc = descs.get(entry.get("name", ""), "").strip()
        if new_desc:
            entry["desc"] = new_desc
            applied += 1
    logger.info("METADATA_FIELDS: %d/%d 項目に xsd 由来の説明を適用", applied, len(metadata_fields))
    return metadata_fields


def _is_cache_fresh(xsd_path: Path, ja_json_path: Path) -> bool:
    if not ja_json_path.exists():
        return False
    if not xsd_path.exists():
        return True
    return ja_json_path.stat().st_mtime >= xsd_path.stat().st_mtime


def _load_ja_map(ja_json_path: Path) -> dict[str, str]:
    data = json.loads(ja_json_path.read_text(encoding="utf-8"))
    return {
        name: rec.get("ja_description", "") if isinstance(rec, dict) else ""
        for name, rec in data.items()
    }
