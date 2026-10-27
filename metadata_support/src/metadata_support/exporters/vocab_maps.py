"""登録形式への変換で使う統制語彙の対応表。"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEF_ITEM_DIR = Path(__file__).resolve().parents[4] / "input_documents" / "def_item"
CESSDA_TOPIC_MAP_PATH = DEF_ITEM_DIR / "cessda_topic_map.csv"

# CESSDA Topic Classification
CESSDA_TOPIC_VOCAB_NAME = "CESSDATopic"
CESSDA_TOPIC_VOCAB_URI_WEKO = "https://vocabularies.cessda.eu/urn/urn:ddi:int.cessda.cv:TopicClassification"
CESSDA_TOPIC_SCHEME_URI = "https://vocabularies.cessda.eu/vocabulary/TopicClassification"


@dataclass(frozen=True)
class CessdaTopic:
    ja: str
    en: str
    notation: str


@lru_cache(maxsize=1)
def load_cessda_topic_map(path: str | None = None) -> dict[str, CessdaTopic]:
    """CESSDA トピック対応表（日本語ラベル→英語ラベル・notation）を読み込む。

    英語ラベル・notation が空の行（CESSDA 語彙に該当なし）は対応表に含めない。
    """
    p = Path(path) if path else CESSDA_TOPIC_MAP_PATH
    mapping: dict[str, CessdaTopic] = {}
    with open(p, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ja = (row.get("ja") or "").strip()
            en = (row.get("en") or "").strip()
            notation = (row.get("notation") or "").strip()
            if ja and en and notation:
                mapping[ja] = CessdaTopic(ja=ja, en=en, notation=notation)
    return mapping


def lookup_cessda_topic(label: str) -> CessdaTopic | None:
    """日本語（または英語・notation）のトピックラベルから CESSDA の語を引く。"""
    label = (label or "").strip()
    if not label:
        return None
    mapping = load_cessda_topic_map()
    if label in mapping:
        return mapping[label]
    lowered = label.lower()
    for topic in mapping.values():
        if topic.en.lower() == lowered or topic.notation.lower() == lowered:
            return topic
    return None


# COAR アクセス権語彙（http://purl.org/coar/access_right/）
@dataclass(frozen=True)
class AccessRight:
    label_en: str
    uri: str


_COAR_ACCESS = {
    "open access": AccessRight("open access", "http://purl.org/coar/access_right/c_abf2"),
    "restricted access": AccessRight("restricted access", "http://purl.org/coar/access_right/c_16ec"),
    "metadata only access": AccessRight("metadata only access", "http://purl.org/coar/access_right/c_14cb"),
    "embargoed access": AccessRight("embargoed access", "http://purl.org/coar/access_right/c_f1cf"),
}

# アプリの日本語統制語彙（config.CONTROLLED_VOCABULARY["アクセス権"]）→ COAR
_ACCESS_JA_TO_EN = {
    "オープンアクセス": "open access",
    "制約付きアクセス": "restricted access",
    "メタデータのみ": "metadata only access",
    "エンバーゴ期間中": "embargoed access",
}


def lookup_access_right(label: str) -> AccessRight | None:
    """アクセス権の値（日本語統制語彙または COAR の英語ラベル）から COAR の語を引く。"""
    key = (label or "").strip()
    if key in _ACCESS_JA_TO_EN:
        key = _ACCESS_JA_TO_EN[key]
    return _COAR_ACCESS.get(key.lower())


# 資源タイプ（COAR resource type: dataset）
RESOURCE_TYPE_DATASET = "dataset"
RESOURCE_TYPE_DATASET_URI = "http://purl.org/coar/resource_type/c_ddb1"
