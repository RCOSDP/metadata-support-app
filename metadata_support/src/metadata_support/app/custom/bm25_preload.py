"""BM25-based local store for OAI-PMH preloaded records (JGSSDDS/RUDA/CSRDA).

レコードはリポジトリごとに ``records_<REPO>.json`` として保存し、起動時に
存在するものだけロードする。あるリポジトリの取得が失敗した場合でも、
他リポジトリのキャッシュを汚さず、次回起動時に未取得分のみ再フェッチできる。
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(
    r"[一-龥]+"          # CJK Unified Ideographs
    r"|[ぁ-んゔー]+"       # Hiragana
    r"|[ァ-ヴー]+"         # Katakana
    r"|[a-z0-9]+"         # ASCII alnum
)


def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenize Japanese / English text for BM25.

    Japanese (CJK / kana) chunks are expanded into character bigrams; ASCII
    alphanumeric runs are kept as whole tokens. Punctuation is dropped.
    """
    if not text:
        return []
    tokens: List[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        chunk = match.group()
        if chunk[0].isascii():
            tokens.append(chunk)
        elif len(chunk) == 1:
            tokens.append(chunk)
        else:
            tokens.extend(chunk[i : i + 2] for i in range(len(chunk) - 1))
    return tokens


class BM25PreloadStore:
    """In-memory BM25 index over preloaded OAI-PMH records, persisted as JSON per repo."""

    def __init__(self, path: str):
        self.path = path
        self.records: List[Dict] = []
        self._bm25: Optional[BM25Okapi] = None

    def _repo_file(self, repo: str) -> str:
        return os.path.join(self.path, f"records_{repo}.json")

    def has_repo_data(self, repo: str) -> bool:
        return os.path.exists(self._repo_file(repo))

    def load_repo(self, repo: str) -> int:
        path = self._repo_file(repo)
        if not os.path.exists(path):
            return 0
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        self.records.extend(records)
        logger.info("BM25_PRELOAD_LOAD | repo=%s | records=%d", repo, len(records))
        return len(records)

    def save_repo(self, repo: str, records: List[Dict]) -> None:
        os.makedirs(self.path, exist_ok=True)
        with open(self._repo_file(repo), "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        logger.info("BM25_PRELOAD_SAVE | repo=%s | records=%d", repo, len(records))

    def add_records(self, records: List[Dict]) -> None:
        self.records.extend(records)

    def build_index(self) -> None:
        if not self.records:
            self._bm25 = None
            return
        tokenized_corpus = [
            tokenize_for_bm25(r.get("text") or r.get("title", "")) for r in self.records
        ]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def search(
        self,
        query: str,
        repos: Optional[List[str]] = None,
        limit: int = 5,
    ) -> List[Tuple[Dict, float]]:
        """Return up to ``limit`` (record, score) pairs ranked by BM25 score (desc)."""
        if self._bm25 is None or not self.records:
            return []
        q_tokens = tokenize_for_bm25(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        candidates = range(len(self.records))
        if repos:
            allowed = set(repos)
            candidates = [
                i for i in candidates
                if self.records[i].get("source_repository") in allowed
            ]
        ranked = sorted(candidates, key=lambda i: scores[i], reverse=True)
        return [(self.records[i], float(scores[i])) for i in ranked[:limit]]

    def __len__(self) -> int:
        return len(self.records)
