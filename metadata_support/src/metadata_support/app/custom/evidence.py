"""各メタデータ項目の根拠情報（どの入力ファイルのどこを参照して生成したか）の構築・保存・表示。

- ChunkRef: 項目生成に使ったチャンク 1件（テキスト・通番・ファイル・確信度・判定項目）
- EvidenceBuilder: 項目毎の根拠レコードと実行情報をまとめた dict を構築する
- EvidenceStore / LocalEvidenceStore: 管理者向け JSON の保存先（S3 等へ差し替え可能）
- evidence_to_html: UI 表示用 HTML（LLM/PDF 由来のテキストはすべてエスケープ）
"""
import hashlib
import html
import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import NamedTuple

logger = logging.getLogger(__name__)

NO_VALUE = "該当なし"

# 根拠の種別
DERIVATION_DOCUMENT = "document"            # 入力文書のチャンクから生成
DERIVATION_BORROWED = "borrowed"            # 他項目として判定されたチャンクを流用して生成
DERIVATION_VOCAB = "vocab_converted"        # 入力文書のチャンクを統制語彙に変換して生成
DERIVATION_EXTERNAL = "external"            # 外部リポジトリのレコードから生成
DERIVATION_MERGED = "merged"                # LLM 生成値と外部レコードの値を統合
DERIVATION_SKIPPED = "skipped"              # 生成処理をスキップ
DERIVATION_NO_REFERENCE = "no_reference"    # 値はあるが文書中の参照箇所なし
DERIVATION_NONE = "none"                    # 値なし（該当なし）

DERIVATION_LABELS = {
    DERIVATION_DOCUMENT: "入力文書の記述から生成",
    DERIVATION_BORROWED: "入力文書の記述（他項目として判定された記述を含む）から生成",
    DERIVATION_VOCAB: "入力文書の記述を統制語彙に変換して生成",
    DERIVATION_EXTERNAL: "外部リポジトリのレコードから生成",
    DERIVATION_MERGED: "入力文書からの生成値と外部リポジトリの値を統合",
    DERIVATION_SKIPPED: "生成処理をスキップ",
    DERIVATION_NO_REFERENCE: "入力文書中の参照箇所なし",
    DERIVATION_NONE: "該当する値なし",
}

ROLE_MAIN = "主"
ROLE_CONTEXT = "前後文脈"

# 他項目のチャンクを流用・統合する項目と、その注記
BORROW_NOTES = {
    "調査機器": "調査方法として判定された記述を参照して生成しています。",
    "調査日": "調査日・対象時期として判定された記述を統合して参照しています。",
    "対象時期": "調査日・対象時期として判定された記述を統合して参照しています。",
    "その他のタイトル": "タイトルとして判定された記述も参照しています。",
}


class ChunkRef(NamedTuple):
    """項目生成に使ったチャンク 1件。

    先頭3要素 (text, doc_chunk_no, source_file) は従来の field_chunks_map のタプルと互換。
    doc_chunk_no はファイル内の 1始まりの通番（統制語彙変換後の値など文書外のものは 0）。
    """

    text: str
    doc_chunk_no: int
    source_file: str
    confidence: float | None = None
    field: str = ""

    @property
    def is_document(self) -> bool:
        """入力文書内の位置を持つチャンクか。"""
        return bool(self.source_file) and self.doc_chunk_no > 0


def chunk_refs_from_df(df) -> list[ChunkRef]:
    """chunks_df（ドキュメント内通番/ソースファイル/該当項目/テキスト/確信度 列）から ChunkRef のリストを作る。"""
    refs = []
    for row in df.to_dict("records"):
        conf = row.get("確信度")
        refs.append(ChunkRef(
            text=str(row.get("テキスト", "")),
            doc_chunk_no=int(row.get("ドキュメント内通番", 0) or 0),
            source_file=str(row.get("ソースファイル", "") or ""),
            confidence=float(conf) if conf is not None else None,
            field=str(row.get("該当項目", "") or ""),
        ))
    return refs


def _is_empty_value(value) -> bool:
    v = "" if value is None else str(value).strip()
    return v == "" or v == NO_VALUE


def _split_values(value) -> list[str]:
    return [p.strip() for p in str(value or "").split(";") if p.strip()]


def _to_jsonable(obj):
    """usage オブジェクト等を JSON 化できる形に変換する。"""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "model_dump"):
        try:
            return _to_jsonable(obj.model_dump())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return {k: _to_jsonable(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


class EvidenceBuilder:
    """メタデータ生成の各段階の中間結果から、項目毎の根拠レコードを構築する。

    主な入力:
      - metadata_fields: METADATA_FIELDS（no/name/ddi を持つ dict のリスト）
      - values: {項目名: 最終値}
      - field_refs: {項目名: [ChunkRef]} 項目抽出（LLM）に渡したチャンク（流用・統合後）
      - skipped_fields: {項目名: 理由} 生成をスキップした項目
      - vocab_sources: {項目名: [ChunkRef]} 統制語彙変換の元にした原文チャンク
      - vocab_converted: {項目名: 変換結果}（変換できなかった項目は含まない）
      - external_records: [{source, record_id, title, fields}] 参照した外部レコード
      - external_derived_fields: 外部レコードのみから生成した項目（関連情報）
      - merged_external: {項目名: [外部レコード由来の値]} LLM 値と統合した項目（URI/トピック）
      - pre_merge_values: {項目名: 統合前の LLM 生成値}
      - file_ordered_chunks: {ファイル名: ["", chunk1, ...]}
      - file_chunk_pages: {ファイル名: [None, (開始ページ, 終了ページ) | None, ...]}
      - llm_logs: {項目名: {"prompt", "response", "usage"}}（管理者向け JSON のみに出力）
    """

    def __init__(
        self,
        *,
        metadata_fields: list[dict],
        values: dict,
        field_refs: dict[str, list[ChunkRef]] | None = None,
        skipped_fields: dict[str, str] | None = None,
        vocab_sources: dict[str, list[ChunkRef]] | None = None,
        vocab_converted: dict[str, str] | None = None,
        external_records: list[dict] | None = None,
        external_derived_fields=None,
        merged_external: dict[str, list[str]] | None = None,
        pre_merge_values: dict | None = None,
        file_ordered_chunks: dict[str, list[str]] | None = None,
        file_chunk_pages: dict[str, list] | None = None,
        llm_logs: dict[str, dict] | None = None,
        context_window: int = 2,
    ):
        self.metadata_fields = metadata_fields
        self.values = values or {}
        self.field_refs = field_refs or {}
        self.skipped_fields = skipped_fields or {}
        self.vocab_sources = vocab_sources or {}
        self.vocab_converted = vocab_converted or {}
        self.external_records = external_records or []
        self.external_derived_fields = set(external_derived_fields or ())
        self.merged_external = merged_external or {}
        self.pre_merge_values = pre_merge_values or {}
        self.file_ordered_chunks = file_ordered_chunks or {}
        self.file_chunk_pages = file_chunk_pages or {}
        self.llm_logs = llm_logs or {}
        self.context_window = context_window

    # ── 位置情報 ──
    def _pages(self, source_file: str, doc_chunk_no: int):
        pages = self.file_chunk_pages.get(source_file) or []
        if 0 < doc_chunk_no < len(pages) and pages[doc_chunk_no]:
            ps, pe = pages[doc_chunk_no]
            return ps, pe
        return None, None

    def _ref_dict(self, ref: ChunkRef, role: str | None = None) -> dict:
        ps, pe = self._pages(ref.source_file, ref.doc_chunk_no)
        d = {
            "source_file": ref.source_file,
            "page_start": ps,
            "page_end": pe,
            "doc_chunk_no": ref.doc_chunk_no,
            "confidence": ref.confidence,
            "text": ref.text,
        }
        if ref.field:
            d["annotated_field"] = ref.field
        if role is not None:
            d = {**d, "role": role}
        return d

    def _references(self, refs: list[ChunkRef]) -> list[dict]:
        """主チャンクと、その前後 context_window チャンク（前後文脈）を返す。"""
        main_refs = [r for r in refs if r.is_document]
        main_keys = {(r.source_file, r.doc_chunk_no) for r in main_refs}
        out: list[dict] = []
        seen_ctx: set = set()
        for ref in main_refs:
            out.append(self._ref_dict(ref, ROLE_MAIN))
            ordered = self.file_ordered_chunks.get(ref.source_file) or [""]
            for offset in list(range(-self.context_window, 0)) + list(range(1, self.context_window + 1)):
                no = ref.doc_chunk_no + offset
                key = (ref.source_file, no)
                if not (1 <= no < len(ordered)) or key in main_keys or key in seen_ctx:
                    continue
                seen_ctx.add(key)
                ctx = self._ref_dict(ChunkRef(ordered[no], no, ref.source_file), ROLE_CONTEXT)
                ctx["context_for"] = ref.doc_chunk_no
                out.append(ctx)
        return out

    # ── 外部レコード ──
    def _external_entries(self, field_name: str, only_values: list[str] | None = None) -> list[dict]:
        """外部レコードのうち、当該項目に値を持つもの（only_values 指定時はその値を含むもの）。"""
        entries = []
        for rec in self.external_records:
            fields = rec.get("fields") or {}
            value = str(fields.get(field_name) or "").strip()
            if not value:
                continue
            if only_values is not None and not (set(_split_values(value)) & set(only_values)):
                continue
            entries.append({
                "source": rec.get("source", ""),
                "record_id": rec.get("record_id", ""),
                "title": rec.get("title", "") or rec.get("label", ""),
                "value": value,
            })
        return entries

    def _external_exact_match(self, field_name: str, value) -> list[dict]:
        """値が外部レコードの同項目の値と一致する（または全要素が含まれる）レコードを返す。"""
        target = str(value or "").strip()
        target_parts = set(_split_values(target))
        matches = []
        for entry in self._external_entries(field_name):
            if entry["value"] == target or (
                target_parts and target_parts <= set(_split_values(entry["value"]))
            ):
                matches.append(entry)
        return matches

    # ── 項目毎のレコード ──
    def build_field(self, field: dict) -> dict:
        name = field["name"]
        value = self.values.get(name, "")
        value = "" if value is None else str(value)
        refs = self.field_refs.get(name, [])
        doc_refs = [r for r in refs if r.is_document]
        notes: list[str] = []
        external: list[dict] = []
        merge_info = None

        skipped = name in self.skipped_fields
        references = [] if skipped else self._references(doc_refs)
        vocab_original = [
            {k: v for k, v in self._ref_dict(r).items() if k != "annotated_field"}
            for r in self.vocab_sources.get(name, []) if r.is_document
        ]

        if name in self.external_derived_fields:
            external = self._external_entries("URI") if not _is_empty_value(value) else []
            derivation = DERIVATION_EXTERNAL
            notes.append("LLM を使わず、外部リポジトリのレコードのタイトルと URI から生成しています。")
        elif name in self.merged_external and self.merged_external[name]:
            ext_values = list(self.merged_external[name])
            llm_value = str(self.pre_merge_values.get(name, "") or "")
            external = self._external_entries(name, only_values=ext_values)
            merge_info = {"llm_value": llm_value, "external_values": ext_values}
            if _is_empty_value(llm_value):
                derivation = DERIVATION_EXTERNAL
                notes.append("入力文書からは値が得られず、外部リポジトリのレコードの値を使用しています。")
            else:
                derivation = DERIVATION_MERGED
                notes.append(
                    f"入力文書からの生成値「{llm_value}」に外部リポジトリの値"
                    f"「{'; '.join(ext_values)}」を統合しています。"
                )
            if name in self.vocab_converted and vocab_original:
                notes.append(f"入力文書の記述は統制語彙「{self.vocab_converted[name]}」に変換されています。")
        elif name in self.vocab_sources:
            if name in self.vocab_converted:
                derivation = DERIVATION_VOCAB
                notes.append(f"入力文書の記述を統制語彙「{self.vocab_converted[name]}」に変換しています。")
            else:
                derivation = DERIVATION_NONE
                notes.append("入力文書の記述を統制語彙に変換できませんでした。")
        elif skipped:
            derivation = DERIVATION_SKIPPED
            notes.append(self.skipped_fields[name])
        elif doc_refs:
            borrowed = any(r.field and r.field != name for r in doc_refs)
            derivation = DERIVATION_BORROWED if borrowed else DERIVATION_DOCUMENT
        else:
            derivation = DERIVATION_NO_REFERENCE
            notes.append("この項目に該当すると判定された記述が入力文書中にありませんでした。")

        if name in BORROW_NOTES and doc_refs and not skipped:
            notes.append(BORROW_NOTES[name])

        # 文書中の参照箇所がない値は、外部レコードの値と一致すれば外部由来とみなす
        if derivation in (DERIVATION_NO_REFERENCE, DERIVATION_SKIPPED) and not _is_empty_value(value):
            matches = self._external_exact_match(name, value)
            if matches:
                derivation = DERIVATION_EXTERNAL
                external = matches
                notes.append("値が外部リポジトリのレコードの値と一致するため、外部レコード由来とみなしています。")
            else:
                notes.append("値は LLM が外部レコードの例示等から推定した可能性があります。")

        if _is_empty_value(value) and derivation not in (DERIVATION_SKIPPED, DERIVATION_NONE):
            if derivation in (DERIVATION_DOCUMENT, DERIVATION_BORROWED, DERIVATION_VOCAB):
                notes.append("参照した記述から値を特定できませんでした（該当なし）。")
            derivation = DERIVATION_NONE

        record = {
            "no": field.get("no"),
            "name": name,
            "ddi": field.get("ddi", ""),
            "value": value,
            "derivation": derivation,
            "derivation_label": DERIVATION_LABELS[derivation],
            "note": " ".join(notes),
            "references": references,
            "external": external,
            "vocab_original": vocab_original,
        }
        if merge_info is not None:
            record["merge"] = merge_info
        if name in self.llm_logs:
            log = self.llm_logs[name] or {}
            record["llm"] = {
                "prompt": log.get("prompt", ""),
                "response": log.get("response", ""),
                "usage": _to_jsonable(log.get("usage")),
            }
        return record

    def build_fields(self) -> list[dict]:
        return [self.build_field(f) for f in self.metadata_fields]

    def build(self, run_info: dict | None = None) -> dict:
        """実行情報と項目毎の根拠レコードをまとめた dict を返す（管理者向け JSON の内容）。"""
        return {**(run_info or {}), "fields": self.build_fields()}

    @staticmethod
    def build_run_info(
        *,
        model: dict,
        pdf_bytes: dict[str, bytes],
        page_counts: dict[str, int] | None = None,
        external_search: dict | None = None,
        settings: dict | None = None,
        generated_at: str | None = None,
    ) -> dict:
        """実行情報（生成日時・モデル・入力ファイル・外部検索設定など）を返す。"""
        page_counts = page_counts or {}
        files = [
            {
                "name": name,
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "pages": page_counts.get(name),
            }
            for name, data in pdf_bytes.items()
        ]
        return {
            "generated_at": generated_at or datetime.now(timezone.utc).astimezone().isoformat(),
            "model": model,
            "input_files": files,
            "external_search": external_search or {"enabled": False},
            "settings": settings or {},
        }

    @staticmethod
    def for_display(evidence: dict) -> dict:
        """UI 表示用に LLM のプロンプト等を除いたコピーを返す。"""
        fields = [{k: v for k, v in f.items() if k != "llm"} for f in evidence.get("fields", [])]
        return {**{k: v for k, v in evidence.items() if k != "fields"}, "fields": fields}


# ══════════════════════════════════════════════
# 保存先
# ══════════════════════════════════════════════
class EvidenceStore(ABC):
    """根拠情報 JSON の保存先インターフェース（S3 等の実装に差し替え可能）。"""

    @abstractmethod
    def save(self, evidence: dict, run_id: str) -> str:
        """根拠情報を保存し、保存先（パスや URI）を返す。"""


class LocalEvidenceStore(EvidenceStore):
    """ローカルファイルシステムに `{base_dir}/{run_id}/evidence.json` として保存する。"""

    FILENAME = "evidence.json"

    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def save(self, evidence: dict, run_id: str) -> str:
        out_dir = os.path.join(self.base_dir, run_id)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, self.FILENAME)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(evidence), f, ensure_ascii=False, indent=2)
        logger.info("EVIDENCE_SAVED | path=%s | fields=%d", path, len(evidence.get("fields", [])))
        return path


# ══════════════════════════════════════════════
# UI 表示用 HTML
# ══════════════════════════════════════════════
def find_field_evidence(evidence: dict | None, field_name: str) -> dict | None:
    """根拠情報 dict から指定項目のレコードを返す（なければ None）。"""
    for record in (evidence or {}).get("fields", []):
        if record.get("name") == field_name:
            return record
    return None


def format_location(ref: dict) -> str:
    """`ファイル名 / p.X(–Y)` 形式の位置表記（エスケープ前）。

    チャンク通番・確信度は内部処理の値のため UI には表示しない（根拠情報 JSON には残す）。
    """
    parts = [ref.get("source_file") or "(ファイル不明)"]
    ps, pe = ref.get("page_start"), ref.get("page_end")
    if ps is None:
        parts.append("p.不明")
    elif pe is None or pe == ps:
        parts.append(f"p.{ps}")
    else:
        parts.append(f"p.{ps}–{pe}")
    return " / ".join(parts)


_E = html.escape

_STYLE_LOC = "font-size:13px;font-weight:bold;color:#333333;margin-top:6px;"
_STYLE_TEXT = (
    "font-size:13px;color:#333333;background:#f7f7f7;border-left:3px solid #1565c0;"
    "padding:4px 8px;margin:2px 0 6px 0;white-space:pre-wrap;"
)
_STYLE_CTX_TEXT = (
    "font-size:12px;color:#777777;background:#fafafa;border-left:2px solid #cccccc;"
    "padding:3px 8px;margin:2px 0 4px 0;white-space:pre-wrap;"
)
_STYLE_NOTE = "font-size:13px;color:#555555;margin-bottom:4px;"
_STYLE_HEAD = "font-size:13px;font-weight:bold;color:#555555;margin-top:8px;"


def _ref_html(ref: dict) -> str:
    return (
        f'<div style="{_STYLE_LOC}">{_E(format_location(ref))}</div>'
        f'<div style="{_STYLE_TEXT}">{_E(ref.get("text") or "")}</div>'
    )


def _context_html(ctx_refs: list[dict]) -> str:
    if not ctx_refs:
        return ""
    items = "".join(
        f'<div style="font-size:12px;color:#777777;">{_E(format_location(c))}</div>'
        f'<div style="{_STYLE_CTX_TEXT}">{_E(c.get("text") or "")}</div>'
        for c in sorted(ctx_refs, key=lambda c: c.get("doc_chunk_no") or 0)
    )
    return (
        '<details style="margin:0 0 8px 12px;">'
        f'<summary style="font-size:12px;color:#777777;cursor:pointer;">前後の文脈（{len(ctx_refs)}件）</summary>'
        f"{items}</details>"
    )


def evidence_to_html(record: dict) -> str:
    """1項目分の根拠レコードを HTML にする。テキストはすべて html.escape 済み。"""
    parts = [
        f'<div style="{_STYLE_NOTE}"><b>根拠の種別:</b> {_E(record.get("derivation_label") or "")}</div>'
    ]
    if record.get("note"):
        parts.append(f'<div style="{_STYLE_NOTE}">{_E(record["note"])}</div>')

    merge = record.get("merge")
    if merge:
        parts.append(
            f'<div style="{_STYLE_NOTE}">入力文書からの生成値: {_E(merge.get("llm_value") or NO_VALUE)}<br>'
            f'外部リポジトリの値: {_E("; ".join(merge.get("external_values") or []))}</div>'
        )

    refs = record.get("references") or []
    mains = [r for r in refs if r.get("role") == ROLE_MAIN]
    if mains:
        parts.append(f'<div style="{_STYLE_HEAD}">参照した記述（{len(mains)}件）</div>')
        for m in mains:
            parts.append(_ref_html(m))
            ctx = [
                r for r in refs
                if r.get("role") == ROLE_CONTEXT
                and r.get("source_file") == m.get("source_file")
                and r.get("context_for") == m.get("doc_chunk_no")
            ]
            parts.append(_context_html(ctx))

    vocab = record.get("vocab_original") or []
    if vocab:
        parts.append(f'<div style="{_STYLE_HEAD}">統制語彙変換の元にした記述（{len(vocab)}件）</div>')
        parts.extend(_ref_html(v) for v in vocab)

    external = record.get("external") or []
    if external:
        parts.append(f'<div style="{_STYLE_HEAD}">参照した外部リポジトリのレコード（{len(external)}件）</div>')
        for e in external:
            loc = " / ".join(x for x in [e.get("source"), e.get("record_id"), e.get("title")] if x)
            parts.append(
                f'<div style="{_STYLE_LOC}">{_E(loc)}</div>'
                f'<div style="{_STYLE_TEXT}">{_E(e.get("value") or "")}</div>'
            )

    if not mains and not vocab and not external:
        parts.append(f'<div style="{_STYLE_NOTE}">参照箇所はありません。</div>')
    return "".join(parts)
