from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from metadata_support.ddi_schema.applier import apply_descriptions, format_report
from metadata_support.ddi_schema.description_generator import (
    DEFAULT_MODEL,
    DEFAULT_REGION,
    generate_japanese_descriptions,
)
from metadata_support.ddi_schema.xsd_parser import extract_english_descriptions

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEF_ITEM_DIR = PROJECT_ROOT / "input_documents" / "def_item"
XSD_PATH = DEF_ITEM_DIR / "codebook.xsd"
EN_JSON_PATH = DEF_ITEM_DIR / "ddi_descriptions_en.json"
JA_JSON_PATH = DEF_ITEM_DIR / "ddi_descriptions_ja.json"
DEFAULT_APPLY_TARGET = (
    PROJECT_ROOT
    / "metadata_support"
    / "src"
    / "metadata_support"
    / "app"
    / "app_custom_no_vector.py"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build DDI-Codebook field descriptions for METADATA_FIELDS")
    p.add_argument("--skip-llm", action="store_true", help="LLM整形をスキップ (英語抽出のみ)")
    p.add_argument("--skip-extract", action="store_true", help="英語抽出をスキップ (既存ファイルを利用)")
    p.add_argument("--apply", action="store_true", help="METADATA_FIELDS に自動反映する")
    p.add_argument("--dry-run", action="store_true", help="--apply と併用。ファイル書換えせず差分表示のみ")
    p.add_argument(
        "--apply-target",
        type=Path,
        default=DEFAULT_APPLY_TARGET,
        help=f"反映先の .py パス (default: {DEFAULT_APPLY_TARGET})",
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Bedrock モデルID (default: {DEFAULT_MODEL})")
    p.add_argument("--region", default=DEFAULT_REGION, help=f"AWS リージョン (default: {DEFAULT_REGION})")
    p.add_argument("--xsd", type=Path, default=XSD_PATH, help=f"xsd パス (default: {XSD_PATH})")
    p.add_argument("--en-out", type=Path, default=EN_JSON_PATH, help=f"英語抽出結果の出力先 (default: {EN_JSON_PATH})")
    p.add_argument("--ja-out", type=Path, default=JA_JSON_PATH, help=f"日本語整形結果の出力先 (default: {JA_JSON_PATH})")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    if args.skip_extract:
        if not args.en_out.exists():
            print(f"ERROR: {args.en_out} が存在しません。--skip-extract を外して再実行してください。", file=sys.stderr)
            return 1
        en_data = json.loads(args.en_out.read_text(encoding="utf-8"))
        print(f"[1/2] 既存の英語抽出結果を読込: {args.en_out} ({len(en_data)}件)")
    else:
        print(f"[1/2] xsd から英語説明を抽出: {args.xsd}")
        xsd_text = args.xsd.read_text(encoding="utf-8")
        en_data = extract_english_descriptions(xsd_text)
        args.en_out.parent.mkdir(parents=True, exist_ok=True)
        args.en_out.write_text(
            json.dumps(en_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"      -> {args.en_out} に {len(en_data)}件保存")
        missing = [k for k, v in en_data.items() if not v.get("en_description")]
        if missing:
            print(f"      WARN: en_description 未抽出: {missing}", file=sys.stderr)

    if args.skip_llm:
        print("[2/3] スキップ (--skip-llm)")
        if args.ja_out.exists():
            ja_data = json.loads(args.ja_out.read_text(encoding="utf-8"))
        else:
            ja_data = None
    else:
        print(f"[2/3] Bedrock ({args.model}) で日本語整形を実行 ...")
        ja_data = generate_japanese_descriptions(en_data, model=args.model, aws_region=args.region)
        args.ja_out.parent.mkdir(parents=True, exist_ok=True)
        args.ja_out.write_text(
            json.dumps(ja_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"      -> {args.ja_out}")

        print("\n=== 差分プレビュー (* = 変更あり) ===")
        changed = 0
        for name, v in ja_data.items():
            cur = v.get("current_desc", "")
            new = v.get("ja_description", "")
            flag = "*" if cur != new else " "
            if cur != new:
                changed += 1
            print(f"{flag} {v['no']:>2} {name}")
            print(f"     現行: {cur}")
            print(f"     新案: {new}")
        print(f"\n合計 {changed}/{len(ja_data)} 項目で差分あり")

    if not args.apply:
        print("[3/3] スキップ (--apply 未指定)")
        return 0

    if ja_data is None:
        print(f"ERROR: {args.ja_out} が存在しないため --apply できません。先にLLM整形を実行してください。", file=sys.stderr)
        return 1
    print(f"\n[3/3] METADATA_FIELDS に反映: {args.apply_target} (dry_run={args.dry_run})")
    result = apply_descriptions(args.apply_target, ja_data, dry_run=args.dry_run)
    print(format_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
