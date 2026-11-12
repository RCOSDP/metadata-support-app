"""PDF のテキスト抽出と正規化を担うクラス。"""
import io
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fitz
import pytesseract
from PIL import Image

from metadata_support.app.custom.text_utils import TextPreprocessor

logger = logging.getLogger(__name__)

# 文字化け検出: ToUnicode CMap の欠落で NKo 文字 (U+07C0-U+07FF) などに化けるケースを検出
_GARBLED_CHAR_RE = re.compile(r'[߀-߿]')
_GARBLED_RATIO_THRESHOLD = 0.01  # この比率を超える NKo 文字が含まれていれば文字化けと判定

# OCR 設定
_OCR_DPI = 300
_OCR_LANG = "jpn+jpn_vert+eng"
_OCR_PSM = 6  # Page Segmentation Mode 6: Assume a single uniform block of text
_OCR_MAX_WORKERS = 4


class _PathFile:
    """ローカルパスを Streamlit UploadedFile 互換にする軽量アダプタ。"""

    def __init__(self, path):
        self.path = Path(path)
        self.name = self.path.name

    def read(self) -> bytes:
        return self.path.read_bytes()


class PDFTextExtractor:
    """アップロードされた複数のPDFから本文を抽出し、ファイル毎・結合テキストを返す。"""

    def __init__(self, uploaded_files):
        """
        uploaded_files: Streamlit の UploadedFile のリスト（.name / .read() を持つ）
        """
        self.uploaded_files = uploaded_files
        self.all_pdf_bytes: dict[str, bytes] = {}
        self.all_full_texts: dict[str, str] = {}
        # {ファイル名: [1ページ目の生テキスト, 2ページ目, ...]}（根拠情報のページ番号算出用）
        self.all_page_texts: dict[str, list[str]] = {}
        self.combined_text: str = ""

    @classmethod
    def from_paths(cls, paths):
        """PDFファイルパスのリスト（str または Path）から PDFTextExtractor を生成する。

        例:
            all_pdf_bytes, all_full_texts, combined_text = (
                PDFTextExtractor.from_paths(["/path/to/a.pdf", "/path/to/b.pdf"]).extract()
            )
        """
        return cls([_PathFile(p) for p in paths])

    def extract(self):
        """全PDFをファイル名順に読み込み、テキスト抽出・正規化を行う。

        戻り値: (all_pdf_bytes, all_full_texts, combined_text)
          - all_pdf_bytes: {ファイル名: bytes}
          - all_full_texts: {ファイル名: 正規化済みテキスト}
          - combined_text: 全テキストを改行で連結し、連続改行を圧縮したもの

        ページ毎の生テキストは `self.all_page_texts` に保持する（戻り値は従来通り3要素）。
        """
        for uploaded_file in sorted(self.uploaded_files, key=lambda f: f.name):
            pdf_bytes = uploaded_file.read()
            self.all_pdf_bytes[uploaded_file.name] = pdf_bytes
            page_texts = self._extract_pdf_pages(pdf_bytes)
            full_text = "".join(page_texts)
            if self._is_garbled(full_text):
                logger.warning(
                    "OCR_FALLBACK | file=%s | reason=garbled_text | original_chars=%d",
                    uploaded_file.name, len(full_text),
                )
                page_texts = self._extract_pdf_pages_ocr(pdf_bytes, uploaded_file.name)
                full_text = "\n\n".join(page_texts)
            self.all_page_texts[uploaded_file.name] = page_texts
            full_text = TextPreprocessor.strip_survey_questions(full_text, uploaded_file.name)
            full_text = TextPreprocessor.normalize_pdf_text(full_text, uploaded_file.name)
            self.all_full_texts[uploaded_file.name] = full_text

        self.combined_text = "\n".join(self.all_full_texts.values())
        self.combined_text = re.sub(r"\n{2,}", "\n", self.combined_text)
        logger.info("TEXT_EXTRACT_DONE | total_chars=%d", len(self.combined_text))
        return self.all_pdf_bytes, self.all_full_texts, self.combined_text

    @staticmethod
    def _extract_pdf_pages(pdf_bytes: bytes) -> list[str]:
        """PDFバイト列からページ毎の生テキストを抽出する。"""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            return [page.get_text() for page in doc]
        finally:
            doc.close()

    @staticmethod
    def _is_garbled(text: str) -> bool:
        """ToUnicode CMap 欠落由来の文字化けを検出する。

        NKo (U+07C0-U+07FF) は日本語文書には通常出現しないため、これが一定比率以上
        含まれていれば PDF の埋め込みフォント由来の文字化けと判定する。
        """
        if not text:
            return False
        garbled = len(_GARBLED_CHAR_RE.findall(text))
        return garbled / len(text) > _GARBLED_RATIO_THRESHOLD

    @staticmethod
    def _extract_pdf_pages_ocr(pdf_bytes: bytes, fname: str) -> list[str]:
        """Tesseract で各ページを OCR してページ毎のテキストを返す。

        - 各ページを fitz で 300DPI レンダリング → PIL Image
        - グレースケール化のみ (2値化はテキスト品質を落とす場合があるため避ける)
        - 並列実行 (ThreadPoolExecutor)
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            page_count = doc.page_count
            page_images: list[bytes] = []
            for i in range(page_count):
                pix = doc[i].get_pixmap(dpi=_OCR_DPI)
                page_images.append(pix.tobytes("png"))
        finally:
            doc.close()

        def ocr_one(idx_png: tuple[int, bytes]) -> tuple[int, str]:
            idx, png = idx_png
            img = Image.open(io.BytesIO(png)).convert("L")
            text = pytesseract.image_to_string(
                img, lang=_OCR_LANG, config=f"--psm {_OCR_PSM}"
            )
            return idx, text

        results: list[tuple[int, str]] = []
        with ThreadPoolExecutor(max_workers=_OCR_MAX_WORKERS) as executor:
            for idx, text in executor.map(ocr_one, enumerate(page_images)):
                results.append((idx, text))
        results.sort(key=lambda x: x[0])
        page_texts = [text for _, text in results]
        logger.info(
            "OCR_DONE | file=%s | pages=%d | chars=%d",
            fname, page_count, sum(len(t) for t in page_texts),
        )
        return page_texts
