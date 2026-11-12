"""アップロードされた PDF ファイルの入力バリデーション。"""
import logging

import fitz
from streamlit import config as st_config

logger = logging.getLogger(__name__)

# 合計サイズ上限。値は .streamlit/config.toml の server.maxUploadSize（MB）を使い、
# Streamlit の 1ファイル単位の判定と同じく 1MB = 1024^2 バイトで換算する。
MAX_TOTAL_UPLOAD_MB = st_config.get_option("server.maxUploadSize")
_BYTES_PER_MB = 1024 * 1024
MAX_TOTAL_UPLOAD_BYTES = MAX_TOTAL_UPLOAD_MB * _BYTES_PER_MB

_PDF_MAGIC = b"%PDF-"
# PDF 仕様上、ヘッダは先頭 1024 バイト以内にあればよい
_PDF_HEADER_SEARCH_BYTES = 1024


class UploadValidator:
    """アップロードファイル群を検証し、エラーメッセージのリストを返す。

    1件でもエラーがあれば処理全体を止める前提のため、全ファイルを検査して
    エラーをまとめて返す。
    """

    def __init__(self, uploaded_files, max_total_bytes: int = MAX_TOTAL_UPLOAD_BYTES):
        """
        uploaded_files: Streamlit の UploadedFile のリスト（.name / .size / .getvalue() を持つ）
        """
        self.uploaded_files = uploaded_files
        self.max_total_bytes = max_total_bytes

    def validate(self) -> list[str]:
        errors: list[str] = []

        total_bytes = sum(f.size for f in self.uploaded_files)
        if total_bytes > self.max_total_bytes:
            errors.append(
                f"アップロードファイルの合計サイズが上限を超えています"
                f"（合計 {_format_mb(total_bytes)} / 上限 {_format_mb(self.max_total_bytes)}）。"
            )

        for f in self.uploaded_files:
            error = self._validate_pdf(f.name, f.getvalue())
            if error:
                errors.append(error)

        if errors:
            logger.warning(
                "UPLOAD_VALIDATION_FAILED | files=%d | total_bytes=%d | errors=%s",
                len(self.uploaded_files), total_bytes, errors,
            )
        return errors

    @staticmethod
    def _validate_pdf(name: str, data: bytes) -> str | None:
        """PDF として読み込めるかを検証し、問題があればエラーメッセージを返す。"""
        if _PDF_MAGIC not in data[:_PDF_HEADER_SEARCH_BYTES]:
            return f"「{name}」はPDFファイルではありません。"
        try:
            doc = fitz.open(stream=data, filetype="pdf")
        except Exception as e:
            logger.warning("PDF_OPEN_ERROR | file=%s | error=%s", name, e)
            return f"「{name}」はPDFファイルとして読み込めません（破損している可能性があります）。"
        try:
            if doc.needs_pass:
                return f"「{name}」はパスワードで保護されているため読み込めません。"
            if doc.page_count == 0:
                return f"「{name}」にページがありません。"
        finally:
            doc.close()
        return None


def _format_mb(num_bytes: int) -> str:
    return f"{num_bytes / _BYTES_PER_MB:.1f}MB"
