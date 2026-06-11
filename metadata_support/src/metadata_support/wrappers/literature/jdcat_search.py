"""JDCAT (https://jdcat.jsps.go.jp) の公開APIを用いたキーワード検索クライアント。

外部依存は httpx のみ（低レベル） + oai_pmh_wrapper（DDIレコード取得ヘルパー）。
"""
import logging
from typing import Any, Dict, List, Optional

import httpx

from metadata_support.wrappers.literature.oai_pmh_wrapper import (
    OAI_REPOSITORIES,
    fetch_jpcoar_record,
)

logger = logging.getLogger(__name__)

JDCAT_API_URL = "https://jdcat.jsps.go.jp/api/records/"


def search_jdcat(
    query: str,
    page: int = 1,
    size: int = 20,
    sort: str = "wtl",
    search_type: int = 0,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """JDCAT API にキーワード検索をかけて 1 ページ分の JSON を返す。"""
    params = {
        "page": page,
        "size": size,
        "sort": sort,
        "search_type": search_type,
        "q": query,
    }
    headers = {"Accept": "application/json"}
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(JDCAT_API_URL, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()


def search_jdcat_all_pages(
    query: str,
    size: int = 20,
    max_pages: int = 100,
    sort: str = "wtl",
    search_type: int = 0,
    timeout: float = 30.0,
) -> List[Dict[str, Any]]:
    """JDCAT API を全ページ走査してヒット一覧を返す。

    - total に到達したら終了
    - ヒット数が `size` 未満のページは末尾とみなして終了
    - max_pages で暴走防止（size=20 × 100 = 2000件）
    """
    logger.info("JDCAT_KEYWORD_SEARCH | query=%s | size=%d", query[:100], size)
    all_hits: List[Dict[str, Any]] = []
    page = 1
    try:
        while page <= max_pages:
            data = search_jdcat(
                query=query,
                page=page,
                size=size,
                sort=sort,
                search_type=search_type,
                timeout=timeout,
            )
            hits = extract_hits(data)
            if not hits:
                break
            all_hits.extend(hits)
            total = data.get("hits", {}).get("total")
            if isinstance(total, dict):
                total = total.get("value")
            if isinstance(total, int) and len(all_hits) >= total:
                break
            if len(hits) < size:
                break
            page += 1
    except Exception as e:
        logger.error(
            "JDCAT_KEYWORD_SEARCH_ERROR | query=%s | error=%s | collected=%d",
            query[:80], e, len(all_hits),
        )
        return all_hits
    logger.info(
        "JDCAT_KEYWORD_SEARCH_DONE | query=%s | pages=%d | hits=%d",
        query[:80], page, len(all_hits),
    )
    return all_hits


def extract_oai_id(hit: Dict[str, Any]) -> str:
    """検索結果の1ヒットから OAI-PMH 識別子（oai:jdcat.jsps.go.jp:XXXXXXXX）を取り出す。"""
    meta = hit.get("metadata", {})
    oai = meta.get("_oai", {})
    return oai.get("id", "") if isinstance(oai, dict) else ""


def extract_hits(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """JDCAT API レスポンスから hits の配列を取り出す。"""
    return data.get("hits", {}).get("hits", []) or []


def hits_to_ddi_records(
    hits: List[Dict[str, Any]],
    source_survey: str = "",
) -> List[Dict[str, Any]]:
    """各 hit の OAI 識別子から jpcoar 1.0 レコードを取得し、整形済みレコードのリストを返す。

    戻り値の各レコード:
      {
        "id": OAI識別子,
        "text": title,  # ベクトル検索等のキー用
        "fields": {項目名: 値, ...},  # jpcoar 1.0 正規化済み
        "source_repository": "JDCAT",
        "source_survey": 検索元クエリ（任意）,
        "title": title,
      }
    """
    jdcat_config = OAI_REPOSITORIES["JDCAT"]
    jdcat_base_url = jdcat_config["url"]
    metadata_prefix = jdcat_config.get("metadata_prefix", "jpcoar_1.0")
    records: List[Dict[str, Any]] = []
    seen_ids = set()
    for hit in hits:
        oai_id = extract_oai_id(hit)
        if not oai_id or oai_id in seen_ids:
            continue
        seen_ids.add(oai_id)
        rec = fetch_jpcoar_record(
            base_url=jdcat_base_url,
            identifier=oai_id,
            metadata_prefix=metadata_prefix,
        )
        if not rec:
            logger.warning("JDCAT_JPCOAR_FETCH_FAILED | identifier=%s", oai_id)
            continue
        fields = {k: v for k, v in (rec.get("fields") or {}).items() if v}
        title = fields.get("タイトル", "") or rec.get("title", "")
        if not title:
            continue
        records.append({
            "id": oai_id,
            "text": title,
            "fields": fields,
            "source_repository": "JDCAT",
            "source_survey": source_survey,
            "title": title,
        })
    logger.info(
        "JDCAT_JPCOAR_BUILD | survey=%s | records=%d | prefix=%s",
        (source_survey or "")[:80], len(records), metadata_prefix,
    )
    return records
