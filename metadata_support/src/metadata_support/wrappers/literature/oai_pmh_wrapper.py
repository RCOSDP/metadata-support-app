"""OAI-PMH wrapper for social science data repositories."""

import logging
import time
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional
from xml.etree import ElementTree as ET

import requests

from metadata_support.wrappers.base_wrapper import BaseWrapper

logger = logging.getLogger(__name__)

# OAI-PMH namespaces
NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/",
    "dc": "http://purl.org/dc/elements/1.1/",
}

# JPCOAR 1.0 namespaces (used by JGSSDDS — oai_dc is empty there)
JPCOAR_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "jpcoar": "https://github.com/JPCOAR/schema/blob/master/1.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
    "datacite": "https://schema.datacite.org/meta/kernel-4/",
}

# JPCOAR 2.0 namespaces (used by RUDA — provides richer fields than 1.0)
JPCOAR2_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "jpcoar": "https://github.com/JPCOAR/schema/blob/master/2.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
    "datacite": "https://schema.datacite.org/meta/kernel-4/",
}

# DDI Codebook 2.5 namespace
DDI_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "ddi": "ddi:codebook:2_5",
}

# DDI field definitions: (field_name, section, xpath_within_section, multi-value)
# section: top-level child of stdyDscr that carries xml:lang (citation, stdyInfo, etc.)
# Special section "codebook" means search from codeBook element directly.
_DDI_FIELD_DEFS = [
    ("タイトル",          "citation",     "ddi:titlStmt/ddi:titl",            False),
    ("その他のタイトル",   "citation",     "ddi:titlStmt/ddi:altTitl",         False),
    ("作成者",            "citation",     "ddi:rspStmt/ddi:AuthEnty",         False),
    ("研究助成機関",       "citation",     "ddi:prodStmt/ddi:fundAg",          False),
    ("研究費番号",         "citation",     "ddi:prodStmt/ddi:grantNo",         False),
    ("整理番号",          "citation",     "ddi:titlStmt/ddi:IDNo",            True),
    ("編集者",            "citation",     "ddi:prodStmt/ddi:producer",        False),
    ("配布者",            "citation",     "ddi:distStmt/ddi:distrbtr",        False),
    ("寄託者",            "citation",     "ddi:distStmt/ddi:depositr",        False),
    ("権利情報",          "citation",     "ddi:prodStmt/ddi:copyright",       False),
    ("バージョン情報",     "citation",     "ddi:verStmt/ddi:version",          False),
    ("引用上の注意",       "citation",     "ddi:biblCit",                      False),
    ("URI",              "codebook",     "ddi:holdings",                      True),
    ("トピック",          "stdyInfo",     "ddi:subject/ddi:topcClas",         True),
    ("概要",             "stdyInfo",     "ddi:abstract",                      False),
    ("対象時期",          "stdyInfo",     "ddi:sumDscr/ddi:timePrd",          True),
    ("調査日",            "stdyInfo",     "ddi:sumDscr/ddi:collDate",         True),
    ("対象地域",          "stdyInfo",     "ddi:sumDscr/ddi:geogCover",        False),
    ("観察単位",          "stdyInfo",     "ddi:sumDscr/ddi:anlyUnit",         False),
    ("母集団",            "stdyInfo",     "ddi:sumDscr/ddi:universe",         False),
    ("データタイプ",       "stdyInfo",     "ddi:sumDscr/ddi:dataKind",         False),
    ("時間的設計",         "method",       "ddi:dataColl/ddi:timeMeth",        False),
    ("データ収集者",       "method",       "ddi:dataColl/ddi:dataCollector",   False),
    ("サンプリング方法",   "method",       "ddi:dataColl/ddi:sampProc",        True),
    ("調査方法",          "method",       "ddi:dataColl/ddi:collMode",         False),
    ("調査機器",          "method",       "ddi:dataColl/ddi:resInstru",        False),
    ("ウェイト",          "method",       "ddi:dataColl/ddi:weight",           False),
    ("クリーニング処理",   "method",       "ddi:dataColl/ddi:cleanOps",        False),
    ("回収率",            "method",       "ddi:anlyInfo/ddi:respRate",         False),
    ("アクセス権",         "dataAccs",     "ddi:setAvail/ddi:avlStatus",      False),
    ("データファイルURI",  "dataAccs",     "ddi:setAvail/ddi:accsPlac",       True),
    ("利用条件",          "dataAccs",     "ddi:useStmt/ddi:conditions",       False),
    ("関連情報",          "othrStdyMat",  "ddi:relMat",                       False),
    ("関連文献",          "othrStdyMat",  "ddi:relPubl",                      False),
]

# Supported repositories
OAI_REPOSITORIES = {
    "JDCAT": {
        "url": "https://jdcat.jsps.go.jp/oai",
        "label": "人文学・社会科学データカタログ",
        "metadata_prefix": "jpcoar_1.0",
        "fetch_mode": "get_record",
    },
    "JGSSDDS": {
        "url": "https://jgssdds.repo.nii.ac.jp//oai",
        "label": "JGSSデータダウンロードシステム",
        "metadata_prefix": "jpcoar_1.0",
    },
    "RUDA": {
        "url": "https://ruda.repo.nii.ac.jp/oai",
        "label": "立教大学データアーカイブ",
        "metadata_prefix": "jpcoar_2.0",
    },
    "CSRDA": {
        "url": "https://ssjda.iss.u-tokyo.ac.jp/Direct/oai2/",
        "label": "東京大学社会科学研究所データアーカイブ",
        "metadata_prefix": "oai_dc",
    },
}


def _get_text(element, tag: str, ns: dict = NS, lang: str = "ja") -> str:
    """Extract text from a DC element, preferring the specified language."""
    elements = element.findall(tag, ns)
    if not elements:
        return ""
    # Prefer the specified language
    for el in elements:
        el_lang = el.attrib.get("{http://www.w3.org/XML/1998/namespace}lang", "")
        if el_lang == lang:
            return (el.text or "").strip()
    # Fall back to first element
    return (elements[0].text or "").strip()


def _get_all_text(element, tag: str, ns: dict = NS, lang: str = "ja") -> List[str]:
    """Extract all text values from DC elements for a given tag."""
    elements = element.findall(tag, ns)
    results = []
    for el in elements:
        text = (el.text or "").strip()
        if text:
            el_lang = el.attrib.get("{http://www.w3.org/XML/1998/namespace}lang", "")
            if el_lang == lang or el_lang == "":
                results.append(text)
    return results


def _parse_oai_dc_record(record_el) -> Optional[Dict]:
    """Parse a single OAI-PMH record with oai_dc metadata.

    Returns a dict with ``fields`` keyed by Japanese METADATA_FIELDS names so
    downstream code can treat oai_dc, jpcoar, and DDI records uniformly.
    """
    header = record_el.find("oai:header", NS)
    if header is None:
        return None
    if header.attrib.get("status", "") == "deleted":
        return None

    identifier = header.findtext("oai:identifier", default="", namespaces=NS)

    metadata = record_el.find("oai:metadata", NS)
    if metadata is None:
        return None

    dc = metadata.find("oai_dc:dc", NS)
    if dc is None:
        return None

    title = _get_text(dc, "dc:title")
    if not title:
        return None

    description = _get_text(dc, "dc:description")
    creators = _get_all_text(dc, "dc:creator")
    subjects = _get_all_text(dc, "dc:subject")
    coverage = _get_text(dc, "dc:coverage")
    publisher = _get_text(dc, "dc:publisher")
    dc_type = _get_text(dc, "dc:type")
    rights = _get_text(dc, "dc:rights")
    date = _get_text(dc, "dc:date")
    identifiers = _get_all_text(dc, "dc:identifier")

    fields: Dict[str, str] = {}
    fields["タイトル"] = title
    if creators:
        fields["作成者"] = "; ".join(creators)
    if subjects:
        fields["トピック"] = "; ".join(subjects)
    if description:
        fields["概要"] = description
    if coverage:
        fields["対象時期"] = coverage
    if date:
        fields["調査日"] = date
    if publisher:
        fields["配布者"] = publisher
    if dc_type:
        fields["データタイプ"] = dc_type
    if rights:
        fields["権利情報"] = rights
    if identifiers:
        fields["URI"] = "; ".join(identifiers)

    text_parts = [f"{k}: {v[:500] if k == '概要' else v}" for k, v in fields.items()]

    return {
        "id": identifier,
        "title": title,
        "text": "\n".join(text_parts),
        "fields": fields,
    }


def _parse_jpcoar_record(record_el, ns: dict = JPCOAR_NS) -> Optional[Dict]:
    """Parse a single OAI-PMH record with JPCOAR (1.0 or 2.0) metadata.

    Returns a dict with ``fields`` keyed by Japanese METADATA_FIELDS names so
    downstream code can treat oai_dc, jpcoar, and DDI records uniformly.
    Pass ``ns=JPCOAR2_NS`` for jpcoar_2.0 endpoints (e.g. RUDA).
    """
    header = record_el.find("oai:header", ns)
    if header is None:
        return None
    if header.attrib.get("status", "") == "deleted":
        return None

    identifier = header.findtext("oai:identifier", default="", namespaces=ns)

    metadata = record_el.find("oai:metadata", ns)
    if metadata is None:
        return None

    jpcoar = metadata.find("jpcoar:jpcoar", ns)
    if jpcoar is None:
        return None

    title = _get_text(jpcoar, "dc:title", ns)
    if not title:
        return None

    # その他のタイトル
    alt_titles = _get_all_text(jpcoar, "dcterms:alternative", ns)

    # 作成者: jpcoar:creator > jpcoar:creatorName (複数)
    creators: List[str] = []
    for creator_el in jpcoar.findall("jpcoar:creator", ns):
        name_el = creator_el.find("jpcoar:creatorName", ns)
        if name_el is not None:
            txt = (name_el.text or "").strip()
            if txt:
                creators.append(txt)

    # 配布者: jpcoar:contributor[contributorType="Distributor"] > contributorName
    distributors: List[str] = []
    for contrib_el in jpcoar.findall("jpcoar:contributor", ns):
        if contrib_el.attrib.get("contributorType") != "Distributor":
            continue
        name_el = contrib_el.find("jpcoar:contributorName", ns)
        if name_el is not None:
            txt = (name_el.text or "").strip()
            if txt:
                distributors.append(txt)
    # publisher 要素が直接ある場合のフォールバック
    if not distributors:
        pub = _get_text(jpcoar, "dc:publisher", ns)
        if pub:
            distributors.append(pub)

    # 研究助成機関 / 研究費番号: jpcoar:fundingReference 配下
    funders: List[str] = []
    award_numbers: List[str] = []
    for fr_el in jpcoar.findall("jpcoar:fundingReference", ns):
        for fname_el in fr_el.findall("jpcoar:funderName", ns):
            txt = (fname_el.text or "").strip()
            if txt and txt not in funders:
                funders.append(txt)
        for aw_el in fr_el.findall("datacite:awardNumber", ns):
            txt = (aw_el.text or "").strip()
            if txt and txt not in award_numbers:
                award_numbers.append(txt)

    # トピック (multiple, prefer ja)
    subjects = _get_all_text(jpcoar, "jpcoar:subject", ns)

    # 概要・回収率: datacite:description (descriptionType は 1.0=Abstract, 2.0=abstract と
    # 大小文字差があるため小文字比較)
    abstract = ""
    response_rate = ""
    for el in jpcoar.findall("datacite:description", ns):
        txt = (el.text or "").strip()
        if not txt:
            continue
        dtype = el.attrib.get("descriptionType", "").strip().lower()
        if dtype == "abstract":
            if not abstract:  # 最初のAbstractを採用
                abstract = txt
        elif dtype == "response rate":
            if not response_rate:
                response_rate = txt
        elif not abstract and not dtype:
            abstract = txt  # type 未指定なら概要扱い（Abstractが見つかれば上書き）

    # 対象時期: dcterms:temporal
    temporal = "; ".join(_get_all_text(jpcoar, "dcterms:temporal", ns))

    # 調査日: datacite:date[dateType="Issued"]
    survey_date = ""
    for el in jpcoar.findall("datacite:date", ns):
        if el.attrib.get("dateType") == "Issued":
            survey_date = (el.text or "").strip()
            break

    # 対象地域: datacite:geoLocation/datacite:geoLocationPlace
    places: List[str] = []
    for geo_el in jpcoar.findall("datacite:geoLocation", ns):
        for pl_el in geo_el.findall("datacite:geoLocationPlace", ns):
            txt = (pl_el.text or "").strip()
            if txt:
                places.append(txt)

    # データタイプ: jpcoar:resourceType を優先、なければ dc:type
    rtype_el = jpcoar.find("jpcoar:resourceType", ns)
    dc_type = (rtype_el.text or "").strip() if rtype_el is not None else ""
    if not dc_type:
        dc_type = _get_text(jpcoar, "dc:type", ns)

    # 権利情報: dc:rights
    rights = "; ".join(_get_all_text(jpcoar, "dc:rights", ns))

    # バージョン情報: datacite:version
    version = _get_text(jpcoar, "datacite:version", ns)

    # アクセス権: dcterms:accessRights
    access_rights = _get_text(jpcoar, "dcterms:accessRights", ns)

    # URI: jpcoar:identifier (URI/HDL/DOI 優先)
    uris: List[str] = []
    for el in jpcoar.findall("jpcoar:identifier", ns):
        txt = (el.text or "").strip()
        if txt and txt not in uris:
            uris.append(txt)

    # データファイルURI: jpcoar:file/jpcoar:URI
    file_uris: List[str] = []
    for f_el in jpcoar.findall("jpcoar:file", ns):
        for u_el in f_el.findall("jpcoar:URI", ns):
            txt = (u_el.text or "").strip()
            if txt and txt not in file_uris:
                file_uris.append(txt)

    # 関連情報・関連文献: jpcoar:relation/jpcoar:relatedTitle
    relations: List[str] = []
    for rel_el in jpcoar.findall("jpcoar:relation", ns):
        for rt_el in rel_el.findall("jpcoar:relatedTitle", ns):
            txt = (rt_el.text or "").strip()
            if txt and txt not in relations:
                relations.append(txt)

    fields: Dict[str, str] = {}
    fields["タイトル"] = title
    if alt_titles:
        fields["その他のタイトル"] = "; ".join(alt_titles)
    if creators:
        fields["作成者"] = "; ".join(creators)
    if funders:
        fields["研究助成機関"] = "; ".join(funders)
    if award_numbers:
        fields["研究費番号"] = "; ".join(award_numbers)
    if distributors:
        fields["配布者"] = "; ".join(distributors)
    if rights:
        fields["権利情報"] = rights
    if version:
        fields["バージョン情報"] = version
    if uris:
        fields["URI"] = "; ".join(uris)
    if subjects:
        fields["トピック"] = "; ".join(subjects)
    if abstract:
        fields["概要"] = abstract
    if temporal:
        fields["対象時期"] = temporal
    if survey_date:
        fields["調査日"] = survey_date
    if places:
        fields["対象地域"] = "; ".join(places)
    if dc_type:
        fields["データタイプ"] = dc_type
    if response_rate:
        fields["回収率"] = response_rate
    if access_rights:
        fields["アクセス権"] = access_rights
    if file_uris:
        fields["データファイルURI"] = "; ".join(file_uris)
    if relations:
        fields["関連情報"] = "; ".join(relations)

    text_parts = [f"{k}: {v[:500] if k == '概要' else v}" for k, v in fields.items()]

    return {
        "id": identifier,
        "title": title,
        "text": "\n".join(text_parts),
        "fields": fields,
    }


def _find_ja_section(parent, tag: str, ns: dict = DDI_NS):
    """Find a child element preferring xml:lang='ja', falling back to first."""
    xml_lang = "{http://www.w3.org/XML/1998/namespace}lang"
    sections = parent.findall(tag, ns)
    for s in sections:
        if s.attrib.get(xml_lang, "") == "ja":
            return s
    return sections[0] if sections else None


def _parse_ddi_record(record_el) -> Optional[Dict]:
    """Parse a single OAI-PMH record with DDI Codebook 2.5 metadata."""
    header = record_el.find("oai:header", DDI_NS)
    if header is None:
        return None
    identifier = header.findtext("oai:identifier", default="", namespaces=DDI_NS)

    metadata = record_el.find("oai:metadata", DDI_NS)
    if metadata is None:
        return None
    codebook = metadata.find("ddi:codeBook", DDI_NS)
    if codebook is None:
        return None

    stdy_dscr = codebook.find("ddi:stdyDscr", DDI_NS)
    if stdy_dscr is None:
        return None

    # Cache Japanese sections (citation, stdyInfo, method, dataAccs, othrStdyMat)
    ja_sections = {}
    for section_tag in ("citation", "stdyInfo", "method", "dataAccs", "othrStdyMat"):
        ja_sections[section_tag] = _find_ja_section(
            stdy_dscr, f"ddi:{section_tag}", DDI_NS
        )

    # Extract each DDI field
    fields = {}
    text_parts = []
    for field_name, section, xpath, is_multi in _DDI_FIELD_DEFS:
        if section == "codebook":
            search_el = codebook
        else:
            search_el = ja_sections.get(section)
        if search_el is None:
            fields[field_name] = ""
            continue

        elements = search_el.findall(xpath, DDI_NS)
        if is_multi:
            values = []
            for el in elements:
                uri = el.attrib.get("URI", "")
                txt = (el.text or "").strip()
                values.append(uri if uri and not txt else txt)
            value = "; ".join(v for v in values if v)
        else:
            if elements:
                el = elements[0]
                uri = el.attrib.get("URI", "")
                txt = (el.text or "").strip()
                value = uri if uri and not txt else txt
            else:
                value = ""

        fields[field_name] = value
        if value:
            text_parts.append(f"{field_name}: {value}")

    title = fields.get("タイトル", "")
    if not title:
        return None

    return {
        "id": identifier,
        "title": title,
        "text": "\n".join(text_parts),
        "fields": fields,
    }


def fetch_ddi_record(
    base_url: str,
    identifier: str,
    timeout: int = 30,
) -> Optional[Dict]:
    """Fetch a single DDI record via OAI-PMH GetRecord."""
    params = {
        "verb": "GetRecord",
        "metadataPrefix": "ddi",
        "identifier": identifier,
    }
    try:
        resp = requests.get(base_url, params=params, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        logger.warning("OAI-PMH GetRecord failed for %s: %s", identifier, e)
        return None

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as e:
        logger.warning("Failed to parse DDI XML for %s: %s", identifier, e)
        return None

    error = root.find("oai:error", DDI_NS)
    if error is not None:
        logger.warning(
            "OAI-PMH GetRecord error for %s: %s (%s)",
            identifier, error.text, error.attrib.get("code", ""),
        )
        return None

    record_el = root.find("oai:GetRecord/oai:record", DDI_NS)
    if record_el is None:
        return None

    return _parse_ddi_record(record_el)


def fetch_jpcoar_record(
    base_url: str,
    identifier: str,
    metadata_prefix: str = "jpcoar_1.0",
    timeout: int = 30,
) -> Optional[Dict]:
    """Fetch a single jpcoar (1.0 / 2.0) record via OAI-PMH GetRecord."""
    params = {
        "verb": "GetRecord",
        "metadataPrefix": metadata_prefix,
        "identifier": identifier,
    }
    ns = JPCOAR2_NS if metadata_prefix == "jpcoar_2.0" else JPCOAR_NS
    try:
        resp = requests.get(base_url, params=params, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        logger.warning("OAI-PMH GetRecord failed for %s: %s", identifier, e)
        return None

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as e:
        logger.warning("Failed to parse jpcoar XML for %s: %s", identifier, e)
        return None

    error = root.find("oai:error", ns)
    if error is not None:
        logger.warning(
            "OAI-PMH GetRecord error for %s: %s (%s)",
            identifier, error.text, error.attrib.get("code", ""),
        )
        return None

    record_el = root.find("oai:GetRecord/oai:record", ns)
    if record_el is None:
        return None

    return _parse_jpcoar_record(record_el, ns=ns)


def fetch_oai_records(
    base_url: str,
    metadata_prefix: str = "oai_dc",
    max_records: Optional[int] = 100,
    timeout: int = 60,
) -> List[Dict]:
    """
    Fetch records from an OAI-PMH endpoint using ListRecords.

    Uses resumptionToken for pagination. If ``max_records`` is None, fetch all
    records exposed by the endpoint.
    """
    records = []
    params = {
        "verb": "ListRecords",
        "metadataPrefix": metadata_prefix,
    }

    if metadata_prefix == "jpcoar_2.0":
        parse_record = lambda rec_el: _parse_jpcoar_record(rec_el, ns=JPCOAR2_NS)
    elif metadata_prefix.startswith("jpcoar"):
        parse_record = lambda rec_el: _parse_jpcoar_record(rec_el, ns=JPCOAR_NS)
    else:
        parse_record = _parse_oai_dc_record

    while max_records is None or len(records) < max_records:
        try:
            time.sleep(0.5)
            resp = requests.get(base_url, params=params, timeout=timeout)
            resp.raise_for_status()
        except Exception as e:
            logger.warning("OAI-PMH request failed for %s: %s", base_url, e)
            break

        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError as e:
            logger.warning("Failed to parse OAI-PMH XML from %s: %s", base_url, e)
            break

        # Check for error
        error = root.find("oai:error", NS)
        if error is not None:
            logger.warning(
                "OAI-PMH error from %s: %s (%s)",
                base_url,
                error.text,
                error.attrib.get("code", ""),
            )
            break

        list_records = root.find("oai:ListRecords", NS)
        if list_records is None:
            break

        for record_el in list_records.findall("oai:record", NS):
            parsed = parse_record(record_el)
            if parsed:
                records.append(parsed)
            if max_records is not None and len(records) >= max_records:
                break

        # Check for resumptionToken
        token_el = list_records.find("oai:resumptionToken", NS)
        if token_el is None or not (token_el.text or "").strip():
            break
        params = {
            "verb": "ListRecords",
            "resumptionToken": token_el.text.strip(),
        }

    return records


@dataclass
class OaiPmhWrapper(BaseWrapper):
    """
    A wrapper to search social science data repositories via OAI-PMH.

    Fetches records from multiple OAI-PMH endpoints, caches them in a
    local vector store, and returns the most similar records for a given
    search query (e.g. a paper title).

    For repositories with fetch_mode="get_record" (e.g. JDCAT),
    specific records are fetched via GetRecord with DDI metadata prefix.
    The identifiers are specified via ddi_identifiers parameter.
    """

    name: ClassVar[str] = "oai_pmh"

    repositories: Dict = field(default_factory=lambda: dict(OAI_REPOSITORIES))
    """Which repositories to query. Keys are repo names, values are config dicts."""

    max_records_per_repo: int = 100
    """Maximum number of records to fetch from each repository."""

    request_timeout: int = 30
    """Timeout in seconds for each OAI-PMH HTTP request."""

    ddi_identifiers: List[str] = field(default_factory=list)
    """OAI-PMH identifiers to fetch via GetRecord (DDI format).
    Used for repositories with fetch_mode='get_record' (e.g. JDCAT).
    Example: ['oai:jdcat.jsps.go.jp:00013661']"""

    def external_search(
        self,
        text: str,
        expand: bool = False,
        limit: Optional[int] = None,
        **kwargs,
    ) -> List[Dict]:
        """
        Fetch records from all configured OAI-PMH repositories.

        - get_record repos (JDCAT): fetch specific DDI records by identifier
        - list_records repos (others): fetch all records via ListRecords
        """
        all_records = []
        for repo_name, repo_config in self.repositories.items():
            fetch_mode = repo_config.get("fetch_mode", "list_records")

            if fetch_mode == "get_record":
                # Fetch specific DDI records by identifier
                for ddi_id in self.ddi_identifiers:
                    logger.info(
                        "OAI_PMH_DDI_FETCH | repo=%s | identifier=%s",
                        repo_name, ddi_id,
                    )
                    rec = fetch_ddi_record(
                        base_url=repo_config["url"],
                        identifier=ddi_id,
                        timeout=self.request_timeout,
                    )
                    if rec:
                        rec["source_repository"] = repo_name
                        rec["id"] = f"{repo_name}:ddi:{rec['id']}"
                        all_records.append(rec)
                        logger.info(
                            "OAI_PMH_DDI_FETCH_DONE | repo=%s | id=%s | title=%s",
                            repo_name, ddi_id, rec.get("title", "")[:80],
                        )
                    else:
                        logger.warning(
                            "OAI_PMH_DDI_FETCH_FAILED | repo=%s | id=%s",
                            repo_name, ddi_id,
                        )
            else:
                # Fetch all records via ListRecords (oai_dc)
                logger.info(
                    "OAI_PMH_FETCH | repo=%s | url=%s",
                    repo_name,
                    repo_config["url"],
                )
                records = fetch_oai_records(
                    base_url=repo_config["url"],
                    metadata_prefix=repo_config["metadata_prefix"],
                    max_records=self.max_records_per_repo,
                    timeout=self.request_timeout,
                )
                for rec in records:
                    rec["source_repository"] = repo_name
                    rec["id"] = f"{repo_name}:{rec['id']}"

                logger.info(
                    "OAI_PMH_FETCH_DONE | repo=%s | records=%d",
                    repo_name,
                    len(records),
                )
                all_records.extend(records)

        logger.info("OAI_PMH_TOTAL_RECORDS | count=%d", len(all_records))
        return all_records
