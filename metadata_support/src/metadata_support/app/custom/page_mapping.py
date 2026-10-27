"""チャンクと PDF ページ番号の対応付け。

テキスト抽出後の前処理（`TextPreprocessor.strip_survey_questions` は末尾の切り詰め、
`TextPreprocessor.normalize_pdf_text` と `LLMProcessor.split_text` は空白・改行の
変更のみ）は空白以外の文字を変更しない。そのため「空白を除去した文字列」の上では

    空白除去(正規化済み全文) == 空白除去(全ページ生テキストの連結) の先頭部分

が成り立つ。この性質を利用し、ページ毎の非空白文字数の累積値（ページ境界インデックス）
と、空白除去済みチャンクの出現位置からページ番号を求める。
"""
import bisect
import logging

logger = logging.getLogger(__name__)


def strip_whitespace(text: str) -> str:
    """空白文字（str.isspace() が True の文字。全角空白・改行を含む）をすべて除去する。

    正規表現の `\\s` と同じ文字集合を対象とするため、正規化処理で変化しうる文字は
    すべてここで除去される。
    """
    return "".join(ch for ch in text if not ch.isspace())


class PageIndex:
    """ページ毎の非空白文字数の累積値から、空白除去済みテキスト上の位置→ページ番号を引く。"""

    def __init__(self, page_texts: list[str]):
        self.page_count = len(page_texts)
        # cumulative[k] = 1〜(k+1) ページ目までの非空白文字数の合計
        self.cumulative: list[int] = []
        total = 0
        for page_text in page_texts:
            total += len(strip_whitespace(page_text or ""))
            self.cumulative.append(total)

    @property
    def total_chars(self) -> int:
        return self.cumulative[-1] if self.cumulative else 0

    def page_of(self, char_idx: int) -> int | None:
        """空白除去済みテキスト上の文字位置（0始まり）が属するページ番号（1始まり）を返す。"""
        if char_idx < 0 or char_idx >= self.total_chars:
            return None
        return bisect.bisect_right(self.cumulative, char_idx) + 1


def map_chunks_to_pages(
    full_text: str,
    chunks: list[str],
    page_texts: list[str] | None,
    fname: str = "",
) -> list[tuple[int, int] | None]:
    """各チャンクの (開始ページ, 終了ページ) を返す（1始まり、チャンクと同じ順序）。

    - full_text: チャンク分割の入力となった正規化済み全文
    - chunks: full_text を分割したチャンク（出現順）
    - page_texts: ページ毎の生テキスト（PDFTextExtractor.all_page_texts の値）

    チャンクは出現順に並んでいるため、直前の一致位置以降を順に検索する。
    見つからない場合は先頭から再検索し、それでも見つからなければ None とする。
    """
    if not page_texts:
        return [None] * len(chunks)
    index = PageIndex(page_texts)
    stripped_full = strip_whitespace(full_text)
    results: list[tuple[int, int] | None] = []
    cursor = 0
    unmapped = 0
    for chunk in chunks:
        needle = strip_whitespace(chunk or "")
        if not needle:
            results.append(None)
            unmapped += 1
            continue
        pos = stripped_full.find(needle, cursor)
        if pos < 0:
            pos = stripped_full.find(needle)
        if pos < 0:
            results.append(None)
            unmapped += 1
            continue
        page_start = index.page_of(pos)
        page_end = index.page_of(pos + len(needle) - 1)
        if page_start is None or page_end is None:
            results.append(None)
            unmapped += 1
            continue
        results.append((page_start, page_end))
        cursor = pos + len(needle)
    if unmapped:
        logger.warning(
            "PAGE_MAPPING_UNMAPPED | file=%s | chunks=%d | unmapped=%d",
            fname, len(chunks), unmapped,
        )
    else:
        logger.info("PAGE_MAPPING_DONE | file=%s | chunks=%d", fname, len(chunks))
    return results
