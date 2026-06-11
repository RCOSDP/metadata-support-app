"""ddi_descriptions_ja.json の ja_description を app_custom_no_vector.py の
METADATA_FIELDS に in-place 反映する。

対象行は `{"no": N, "name": "X", ...}` 形式のみ。他の行は触らない。
既存のカラム整列・引用符・区切りはそのまま維持し、desc の値のみを置換する。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ApplyResult:
    updated: list[tuple[int, str, str, str]]  # (line_no, name, old_desc, new_desc)
    unchanged: list[tuple[int, str, str]]     # (line_no, name, desc)
    missing_in_ja: list[str]                  # ja_description が空 or キー無し
    not_found_in_source: list[str]            # target_file 内に name 行が無い


def apply_descriptions(
    target_file: Path,
    ja_data: dict[str, dict[str, Any]],
    *,
    dry_run: bool = False,
) -> ApplyResult:
    """target_file の METADATA_FIELDS の desc を ja_description で上書きする。"""
    source = target_file.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)

    updated: list[tuple[int, str, str, str]] = []
    unchanged: list[tuple[int, str, str]] = []
    missing_in_ja: list[str] = []
    names_written: set[str] = set()

    for name, rec in ja_data.items():
        ja_desc = rec.get("ja_description", "").strip()
        if not ja_desc:
            missing_in_ja.append(name)
            continue
        new_lines, hit = _replace_desc_for_name(lines, name, ja_desc)
        if hit is None:
            continue
        line_no, old_desc = hit
        if old_desc == ja_desc:
            unchanged.append((line_no, name, old_desc))
        else:
            updated.append((line_no, name, old_desc, ja_desc))
            names_written.add(name)
        lines = new_lines

    not_found_in_source = [
        name for name, rec in ja_data.items()
        if rec.get("ja_description", "").strip()
        and name not in names_written
        and not any(u[1] == name for u in updated)
        and not any(u[1] == name for u in unchanged)
    ]

    if not dry_run and updated:
        target_file.write_text("".join(lines), encoding="utf-8")
        logger.info("wrote %d description updates to %s", len(updated), target_file)

    return ApplyResult(
        updated=updated,
        unchanged=unchanged,
        missing_in_ja=missing_in_ja,
        not_found_in_source=not_found_in_source,
    )


_DESC_RE = re.compile(r'("desc"\s*:\s*")([^"]*)(")')


def _build_name_line_re(name: str) -> re.Pattern[str]:
    """`{"no": N, "name": "<name>", ... "desc": "..."}` の行を特定する正規表現を作る。"""
    return re.compile(
        r'^\s*\{"no":\s*\d+,\s*"name":\s*"'
        + re.escape(name)
        + r'"\s*,.*?"desc"\s*:\s*"[^"]*".*$'
    )


def _replace_desc_for_name(
    lines: list[str],
    name: str,
    new_desc: str,
) -> tuple[list[str], tuple[int, str] | None]:
    """METADATA_FIELDS の name 行を見つけ desc を置換。(新lines, (line_no, old_desc)) を返す。"""
    pattern = _build_name_line_re(name)
    escaped_new = new_desc.replace("\\", "\\\\").replace('"', '\\"')
    for i, line in enumerate(lines):
        if not pattern.match(line.rstrip("\n")):
            continue
        desc_match = _DESC_RE.search(line)
        if desc_match is None:
            continue
        old_desc = desc_match.group(2)
        new_line = _DESC_RE.sub(
            lambda mm: f'{mm.group(1)}{escaped_new}{mm.group(3)}',
            line,
            count=1,
        )
        new_lines = lines.copy()
        new_lines[i] = new_line
        return new_lines, (i + 1, old_desc)
    return lines, None


def format_report(result: ApplyResult) -> str:
    out: list[str] = []
    out.append(f"=== 反映レポート ===")
    out.append(f"更新: {len(result.updated)} 件")
    for ln, name, old, new in result.updated:
        out.append(f"  L{ln} {name}")
        out.append(f"    - {old}")
        out.append(f"    + {new}")
    if result.unchanged:
        out.append(f"変更なし: {len(result.unchanged)} 件")
        for ln, name, desc in result.unchanged:
            out.append(f"  L{ln} {name}: {desc}")
    if result.missing_in_ja:
        out.append(f"ja_description 欠落: {result.missing_in_ja}")
    if result.not_found_in_source:
        out.append(f"ターゲットに該当行なし: {result.not_found_in_source}")
    return "\n".join(out)
