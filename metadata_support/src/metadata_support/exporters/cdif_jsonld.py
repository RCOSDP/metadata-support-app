"""生成結果を CDIF Core 1.1（https://w3id.org/cdif/core/1.1）準拠の JSON-LD に変換する。

- @id / schema:identifier は DOI → URI → 生成した urn:uuid の順で決定する
- schema:dateModified と schema:subjectOf の schema:sdDatePublished は変換日とする
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import date
from typing import Any, Callable

from metadata_support.exporters.value_parsers import (
    doi_to_url,
    extract_doi,
    is_iso_date,
    kaken_grant_url,
    looks_like_organization,
    parse_date_range,
    pair_funding,
    parse_related_info,
    results_to_map,
    sanitize_cell,
    split_multi,
)
from metadata_support.exporters.vocab_maps import (
    CESSDA_TOPIC_SCHEME_URI,
    lookup_access_right,
    lookup_cessda_topic,
)

CDIF_CORE_PROFILE = "https://w3id.org/cdif/core/1.1"

CONTEXT = {
    "schema": "http://schema.org/",
    "dcterms": "http://purl.org/dc/terms/",
    "dcat": "http://www.w3.org/ns/dcat#",
    "prov": "http://www.w3.org/ns/prov#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}

CATALOG_RECORD_DESCRIPTION = (
    "メタデータ生成支援アプリの生成結果（DDI-Codebook 2.5準拠の34項目）からCDIF Core 1.1へ変換。"
    "temporalCoverage/spatialCoverage/wasGeneratedBy/additionalPropertyはCDIF Discovery相当の追加項目。"
)

DATA_CATALOG = {
    "@id": "https://jdcat.jsps.go.jp/",
    "@type": ["schema:DataCatalog"],
    "schema:name": "JDCat (Japan Data Catalog for the Humanities and Social Sciences)",
    "schema:url": "https://jdcat.jsps.go.jp/",
}

# schema:additionalProperty に出力する項目（項目番号順）
ADDITIONAL_PROPERTY_FIELDS = (
    "観察単位", "母集団", "データタイプ", "時間的設計", "サンプリング方法", "調査方法",
    "調査機器", "ウェイト", "クリーニング処理", "回収率", "データファイルURI",
)
# 上記のうち「;」区切りの複数値として扱う項目（それ以外は1つの文として扱う）
_MULTI_VALUE_PROPERTY_FIELDS = {"観察単位", "データタイプ", "サンプリング方法", "調査方法", "データファイルURI"}


def _org(name: str) -> dict[str, Any]:
    return {"@type": ["schema:Organization"], "schema:name": name}


def _agent(name: str) -> dict[str, Any]:
    """作成者・寄託者用のノード。組織名と判定できない場合は @type を付けない。"""
    if looks_like_organization(name):
        return _org(name)
    return {"schema:name": name}


def _role(role_name: str, agent: dict[str, Any]) -> dict[str, Any]:
    return {
        "@type": ["schema:Role"],
        "schema:roleName": role_name,
        "schema:contributor": agent,
    }


def _one_or_list(items: list[Any]) -> Any:
    return items[0] if len(items) == 1 else items


def _typed_date(value: str) -> dict[str, str] | str:
    """日付を xsd 型付きリテラルにする（解釈できない値は文字列のまま）。"""
    if is_iso_date(value):
        dtype = {4: "xsd:gYear", 7: "xsd:gYearMonth", 10: "xsd:date"}[len(value)]
        return {"@value": value, "@type": dtype}
    return value


def _ddi_property_id(ddi_path: str) -> str:
    return "DDI-Codebook 2.5 " + "/".join(p.strip() for p in ddi_path.split("->"))


def _is_pdf(url: str) -> bool:
    return bool(re.search(r"\.pdf$", url.split("?")[0].split("#")[0], re.IGNORECASE))


def build_cdif_jsonld(
    results: list[dict],
    conversion_date: date | None = None,
    uuid_factory: Callable[[], uuid.UUID] = uuid.uuid4,
) -> dict[str, Any]:
    """生成結果を CDIF Core 1.1 の JSON-LD（dict）に変換する。"""
    conversion_date = conversion_date or date.today()
    today = conversion_date.isoformat()
    m = results_to_map(results)
    ddi_paths = {r["name"]: r.get("ddi", "") for r in results}

    doi = extract_doi(m.get("引用上の注意"))
    uris = split_multi(m.get("URI"))
    if doi:
        dataset_id = doi_to_url(doi)
    elif uris:
        dataset_id = uris[0]
    else:
        dataset_id = f"urn:uuid:{uuid_factory()}"

    doc: dict[str, Any] = {
        "@context": dict(CONTEXT),
        "@id": dataset_id,
        "@type": ["schema:Dataset"],
    }

    # 1 タイトル → name、2 その他のタイトル → alternateName
    if m.get("タイトル"):
        doc["schema:name"] = sanitize_cell(m["タイトル"])
    alt_titles = split_multi(m.get("その他のタイトル"))
    if alt_titles:
        doc["schema:alternateName"] = alt_titles

    # 15 概要 → description
    if m.get("概要"):
        doc["schema:description"] = sanitize_cell(m["概要"])

    # 識別子: DOI（引用上の注意から抽出）→ 整理番号 → @id
    if doi:
        doc["schema:identifier"] = {
            "@id": f"{dataset_id}#doi",
            "@type": ["schema:PropertyValue"],
            "schema:propertyID": {"@id": "https://registry.identifiers.org/registry/doi"},
            "schema:value": doi,
            "schema:url": dataset_id,
        }
    else:
        id_nos = split_multi(m.get("整理番号"))
        if id_nos:
            distributors = split_multi(m.get("配布者"))
            identifiers = []
            for no in id_nos:
                pv: dict[str, Any] = {"@type": ["schema:PropertyValue"]}
                # 整理番号の付与機関（配布者）を識別子のスキームとして記載する
                pv["schema:propertyID"] = distributors[0] if distributors else "整理番号"
                pv["schema:value"] = no
                identifiers.append(pv)
            doc["schema:identifier"] = _one_or_list(identifiers)
        else:
            doc["schema:identifier"] = dataset_id

    # 13 URI → sameAs / url
    if uris:
        doc["schema:sameAs"] = uris

    # 11 バージョン情報 → version
    if m.get("バージョン情報"):
        doc["schema:version"] = sanitize_cell(m["バージョン情報"])

    if uris:
        doc["schema:url"] = uris[0]
    elif doi:
        doc["schema:url"] = dataset_id

    # 32 利用条件（優先）/ 10 権利情報 → conditionsOfAccess（文字列）、30 アクセス権 → COAR URI
    conditions: list[Any] = []
    cond_text = sanitize_cell(m.get("利用条件") or m.get("権利情報") or "")
    if cond_text:
        conditions.append(cond_text)
    for label in split_multi(m.get("アクセス権")):
        access = lookup_access_right(label)
        conditions.append({"@id": access.uri} if access else label)
    if conditions:
        doc["schema:conditionsOfAccess"] = conditions

    # 14 トピック → keywords（CESSDA にある語は skos:Concept、それ以外は文字列）
    keywords: list[Any] = []
    for label in split_multi(m.get("トピック")):
        topic = lookup_cessda_topic(label)
        if topic is None:
            keywords.append(label)
            continue
        keywords.append({
            "@type": ["skos:Concept"],
            "skos:prefLabel": [
                {"@value": topic.ja, "@language": "ja"},
                {"@value": topic.en, "@language": "en"},
            ],
            "skos:notation": topic.notation,
            "skos:inScheme": {"@id": CESSDA_TOPIC_SCHEME_URI},
        })
    if keywords:
        doc["schema:keywords"] = keywords

    # 3 作成者 → creator（@list で順序保持）
    creators = split_multi(m.get("作成者"))
    if creators:
        doc["schema:creator"] = {"@list": [_agent(n) for n in creators]}

    # 7 編集者 / 23 データ収集者 / 9 寄託者 → contributor（Role）
    producers = split_multi(m.get("編集者"))
    collectors = split_multi(m.get("データ収集者"))
    depositors = split_multi(m.get("寄託者"))
    contributors = (
        [_role("Producer", _org(n)) for n in producers]
        + [_role("DataCollector", _org(n)) for n in collectors]
        + [_role("Depositor", _agent(n)) for n in depositors]
    )
    if contributors:
        doc["schema:contributor"] = contributors

    # 7 編集者 → publisher、8 配布者 → provider
    if producers:
        doc["schema:publisher"] = _one_or_list([_org(n) for n in producers])
    distributors = split_multi(m.get("配布者"))
    if distributors:
        doc["schema:provider"] = [_org(n) for n in distributors]

    # 4 研究助成機関 / 5 研究費番号 → funding（MonetaryGrant）
    grants = []
    for funder, grant_no in pair_funding(m.get("研究助成機関"), m.get("研究費番号")):
        grant: dict[str, Any] = {"@type": ["schema:MonetaryGrant"]}
        if funder:
            grant["schema:name"] = funder
        if grant_no:
            pv = {"@type": ["schema:PropertyValue"]}
            kaken_url = kaken_grant_url(grant_no)
            if kaken_url:
                pv["schema:propertyID"] = "KAKEN grant number"
            pv["schema:value"] = grant_no
            if kaken_url:
                pv["schema:url"] = kaken_url
            grant["schema:identifier"] = pv
        grants.append(grant)
    if grants:
        doc["schema:funding"] = grants

    # 12 引用上の注意 → creditText
    if m.get("引用上の注意"):
        doc["schema:creditText"] = sanitize_cell(m["引用上の注意"])

    # 33 関連情報 → relatedLink
    links = []
    for title, url in parse_related_info(m.get("関連情報")):
        target: dict[str, Any] = {"@type": ["schema:EntryPoint"]}
        if title:
            target["schema:name"] = title
        if url:
            target["schema:url"] = url
            if _is_pdf(url):
                target["schema:encodingFormat"] = "application/pdf"
        links.append({
            "@type": ["schema:LinkRole"],
            "schema:linkRelationship": "related",
            "schema:target": target,
        })
    if links:
        doc["schema:relatedLink"] = links

    # 34 関連文献 → dcterms:isReferencedBy
    publications = split_multi(m.get("関連文献"))
    if publications:
        doc["dcterms:isReferencedBy"] = publications

    # 16 対象時期 → temporalCoverage
    start, end = parse_date_range(m.get("対象時期"))
    if start or end:
        if start == end or not end:
            coverage = start
        elif not start:
            coverage = f"../{end}"
        else:
            coverage = f"{start}/{end}"
        doc["schema:temporalCoverage"] = [coverage]

    # 18 対象地域 → spatialCoverage
    places = split_multi(m.get("対象地域"))
    if places:
        doc["schema:spatialCoverage"] = [
            {"@type": ["schema:Place"], "schema:name": p} for p in places
        ]

    # 17 調査日 / 23 データ収集者 → prov:wasGeneratedBy
    start, end = parse_date_range(m.get("調査日"))
    if start or end or collectors:
        activity: dict[str, Any] = {
            "@type": ["prov:Activity"],
            "schema:name": "データ収集 (DDI-C collDate)",
        }
        if start:
            activity["prov:startedAtTime"] = _typed_date(start)
        if end:
            activity["prov:endedAtTime"] = _typed_date(end)
        if collectors:
            activity["prov:wasAssociatedWith"] = _one_or_list([_org(n) for n in collectors])
        doc["prov:wasGeneratedBy"] = [activity]

    # その他の調査設計項目 → additionalProperty
    props = []
    for name in ADDITIONAL_PROPERTY_FIELDS:
        raw = m.get(name)
        values = split_multi(raw) if name in _MULTI_VALUE_PROPERTY_FIELDS else (
            [sanitize_cell(raw)] if raw else []
        )
        for value in values:
            props.append({
                "@type": ["schema:PropertyValue"],
                "schema:propertyID": _ddi_property_id(ddi_paths.get(name, "")),
                "schema:name": name,
                "schema:value": value,
            })
    if props:
        doc["schema:additionalProperty"] = props

    doc["schema:dateModified"] = today

    # メタデータレコード（dcat:CatalogRecord）
    doc["schema:subjectOf"] = {
        "@id": f"{dataset_id}#cdif-metadata",
        "@type": ["schema:Dataset"],
        "schema:additionalType": [{"@id": "dcat:CatalogRecord"}],
        "schema:about": {"@id": dataset_id},
        "dcterms:conformsTo": [{"@id": CDIF_CORE_PROFILE}],
        "schema:sdDatePublished": today,
        "schema:description": CATALOG_RECORD_DESCRIPTION,
        "schema:includedInDataCatalog": dict(DATA_CATALOG),
    }
    return doc


def build_cdif_json_text(results: list[dict], conversion_date: date | None = None, **kwargs: Any) -> str:
    """CDIF Core 1.1 の JSON-LD を整形済み文字列で返す。"""
    doc = build_cdif_jsonld(results, conversion_date=conversion_date, **kwargs)
    return json.dumps(doc, ensure_ascii=False, indent=2) + "\n"
