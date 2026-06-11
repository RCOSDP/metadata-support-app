import logging
import re

from metadata_support.app.custom.config import _QUESTION_MARKER_RE, _QUESTION_THRESHOLD

logger = logging.getLogger(__name__)

_SENTENCE_END_CHARS = set("。？！.!?")
_PARAGRAPH_PLACEHOLDER = "\x00P\x00"


class TextPreprocessor:
    """PDFから抽出した生テキストの前処理。"""

    @staticmethod
    def strip_survey_questions(text, filename=""):
        """調査票テキストから質問部分を除去し、表紙・メタデータ部分のみ返す。
        行頭の質問マーカー（問N, 目N 等）が一定数以上あれば調査票と判定し、
        最初の質問マーカー以前のテキストのみを返す。
        """
        markers = list(_QUESTION_MARKER_RE.finditer(text))
        if len(markers) < _QUESTION_THRESHOLD:
            return text
        first = markers[0]
        match_start = first.start()
        if match_start < len(text) and text[match_start] == '\n':
            match_start += 1
        cut_pos = text.rfind('\n', 0, match_start)
        if cut_pos <= 0:
            cut_pos = match_start
        kept = text[:cut_pos].strip()
        logger.info(
            "STRIP_SURVEY_QUESTIONS | file=%s | markers=%d | original=%d chars | kept=%d chars",
            filename, len(markers), len(text), len(kept),
        )
        return kept

    @staticmethod
    def normalize_pdf_text(text, filename=""):
        """PDFレイアウト由来のソフト改行・余分な空白を正規化する。

        - 段落境界（連続改行）は保持
        - 文末（。？！.!?）直後の改行は保持
        - 日本語同士のソフト改行は連結（空白なし）
        - 英数字を含む境界のソフト改行は半角スペースで連結
        """
        before_len = len(text)
        # 行末の半角スペース/タブを除去
        text = re.sub(r'[ \t]+\n', '\n', text)
        # 空白のみの行を含む連続改行を \n\n に圧縮
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # 段落境界をプレースホルダーに退避
        text = text.replace('\n\n', _PARAGRAPH_PLACEHOLDER)
        # 残った単一 \n を前後文脈で処理
        result = []
        for i, ch in enumerate(text):
            if ch == '\n':
                prev = result[-1] if result else ''
                nxt = text[i + 1] if i + 1 < len(text) else ''
                if prev in _SENTENCE_END_CHARS:
                    result.append('\n')
                elif (prev.isascii() and prev.isalnum()) or (nxt.isascii() and nxt.isalnum()):
                    result.append(' ')
                # 日本語同士等 → 何も挟まず結合
            else:
                result.append(ch)
        text = ''.join(result)
        # プレースホルダーを \n\n に戻す
        text = text.replace(_PARAGRAPH_PLACEHOLDER, '\n\n')
        # 連続スペースを1つに
        text = re.sub(r' {2,}', ' ', text)
        text = text.strip()
        logger.info(
            "NORMALIZE_PDF_TEXT | file=%s | before=%d chars | after=%d chars",
            filename, before_len, len(text),
        )
        return text
