"""生成結果を WEKO3 一括登録用 TSV（アイテムタイプ JDCat(22)）に変換する。

テンプレート（input_documents/def_item/weko_jdcat22_template.tsv）の見出し行を読み込み、
各列のパス（2行目）に対応する値を埋めた1行を追加して出力する。
繰り返し項目（例: タイトル[0]）に2件目以降の値がある場合は、[0] の列群を複製して
[1], [2], ... の列群を直前のインデックスの列群の直後に挿入する。
"""
from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path

from metadata_support.exporters.value_parsers import (
    detect_language,
    extract_doi,
    looks_like_organization,
    parse_date_range,
    pair_funding,
    parse_related_info,
    results_to_map,
    sanitize_cell,
    split_multi,
)
from metadata_support.exporters.vocab_maps import (
    CESSDA_TOPIC_VOCAB_NAME,
    CESSDA_TOPIC_VOCAB_URI_WEKO,
    DEF_ITEM_DIR,
    RESOURCE_TYPE_DATASET,
    RESOURCE_TYPE_DATASET_URI,
    lookup_access_right,
    lookup_cessda_topic,
)

logger = logging.getLogger(__name__)

TEMPLATE_PATH = DEF_ITEM_DIR / "weko_jdcat22_template.tsv"

# 見出し行のうち、列のパスと列ラベルが入っている行の位置
PATH_ROW = 1
LABEL_ROW = 2

_GROUP_RE = re.compile(r"^(\.metadata\.item_\d+)\[(\d+)\]")

# JDCat(22) の項目 ID
ITEM_TITLE = "item_1551264308487"            # タイトル
ITEM_ALT_TITLE = "item_1551264326373"        # その他のタイトル
ITEM_STUDY_ID = "item_1586157591881"         # 調査番号
ITEM_CREATOR = "item_1593074267803"          # 作成者
ITEM_PUBLISHER = "item_1551264917614"        # 出版者
ITEM_RIGHTS = "item_1551264629907"           # 権利情報
ITEM_FUND = "item_1592406896282"             # Fund Agency
ITEM_DISTRIBUTOR = "item_1592405734122"      # Distributor
ITEM_DEPOSITOR = "item_1606717399978"        # Depositor
ITEM_VERSION = "item_1551265075370"          # Version
ITEM_TEXT = "item_1592880868902"             # テキスト（引用上の注意）
ITEM_TOPIC_J = "item_1588259902813"          # Topic CESSDA J
ITEM_TOPIC_E = "item_1607048755062"          # Topic CESSDA E
ITEM_SUMMARY = "item_1551264846237"          # Summary DDI
ITEM_TIME_PERIOD = "item_1551265302120"      # Time Period
ITEM_COLL_DATE = "item_1586253152753"        # Date of Collection
ITEM_GEO_J = "item_1570068313185"            # Geographic Coverage J
ITEM_UNIT_J = "item_1588259962697"           # 観察単位（日本語統制語彙）
ITEM_UNIVERSE = "item_1586253249552"         # Universe
ITEM_DATA_TYPE_J = "item_1588260020969"      # Data Type J
ITEM_SAMPLING_J = "item_1588260065703"       # Sampling Procedure J
ITEM_COLL_METHOD_J = "item_1588260105004"    # Collection Method J
ITEM_SAMPLING_RATE = "item_1586253589529"    # Sampling Rate
ITEM_ACCESS_E = "item_1588260178185"         # Access E
ITEM_RELATED_STUDY = "item_1592405736602"    # Related Study
ITEM_RELATED_PUBL = "item_1592405735401"     # Related Publications
ITEM_RESOURCE_TYPE = "item_1591171125745"    # Resource Type
ITEM_CONTENT_FILE = "item_1603811082773"     # 制限公開用のコンテンツファイル


def load_template(path: Path | str | None = None) -> list[list[str]]:
    """テンプレート TSV の見出し行をセルのリストとして読み込む。"""
    p = Path(path) if path else TEMPLATE_PATH
    text = p.read_text(encoding="utf-8")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return [line.split("\t") for line in lines]


class _Values:
    """列パス→値の辞書を組み立てる補助クラス。"""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def set(self, path: str, value: str) -> None:
        value = sanitize_cell(value)
        if value:
            self.data[path] = value

    def item(self, item_id: str, idx: int, sub: str, value: str) -> None:
        self.set(f".metadata.{item_id}[{idx}].{sub}", value)


def _build_values(results: list[dict], conversion_date: date) -> dict[str, str]:
    m = results_to_map(results)
    v = _Values()

    # 【共通】固定値
    v.set(".publish_status", "private")
    v.set(".edit_mode", "Keep")
    v.set(".metadata.pubdate", conversion_date.isoformat())
    v.set(f".metadata.{ITEM_RESOURCE_TYPE}.resourcetype", RESOURCE_TYPE_DATASET)
    v.set(f".metadata.{ITEM_RESOURCE_TYPE}.resourceuri", RESOURCE_TYPE_DATASET_URI)

    # 1 タイトル（日本語のみ。英語タイトル [1] はアプリが生成しないため出力しない）
    if m.get("タイトル"):
        v.item(ITEM_TITLE, 0, "subitem_title", m["タイトル"])
        v.item(ITEM_TITLE, 0, "subitem_title_language", "ja")

    # 2 その他のタイトル
    for i, t in enumerate(split_multi(m.get("その他のタイトル"))):
        v.item(ITEM_ALT_TITLE, i, "subitem_alternative_title", t)
        v.item(ITEM_ALT_TITLE, i, "subitem_alternative_title_language", "ja")

    # 6 整理番号 → 調査番号[n]（付与機関=配布者）、12 引用上の注意中の DOI → 調査番号[n]（付与機関=DOI）
    distributors = split_multi(m.get("配布者"))
    study_ids: list[tuple[str, str]] = [
        (no, distributors[0] if distributors else "") for no in split_multi(m.get("整理番号"))
    ]
    doi = extract_doi(m.get("引用上の注意"))
    if doi:
        study_ids.append((doi, "DOI"))
    for i, (no, agency) in enumerate(study_ids):
        v.item(ITEM_STUDY_ID, i, "subitem_1586156939407", no)
        v.item(ITEM_STUDY_ID, i, "subitem_1586311767281", "ja")
        v.item(ITEM_STUDY_ID, i, "subitem_1591256665864", agency)

    # 3 作成者（組織名と判定できる場合のみ名前タイプ=Organizational）
    for i, name in enumerate(split_multi(m.get("作成者"))):
        v.item(ITEM_CREATOR, i, "creatorNames[0].creatorName", name)
        v.item(ITEM_CREATOR, i, "creatorNames[0].creatorNameLang", "ja")
        if looks_like_organization(name):
            v.item(ITEM_CREATOR, i, "creatorNames[0].creatorNameType", "Organizational")

    # 7 編集者 → 出版者
    for i, name in enumerate(split_multi(m.get("編集者"))):
        v.item(ITEM_PUBLISHER, i, "subitem_publisher", name)
        v.item(ITEM_PUBLISHER, i, "subitem_publisher_language", "ja")

    # 10 権利情報 / 32 利用条件（同文なら統合、異なれば権利情報[n]として追加）
    rights: list[str] = []
    for text in (m.get("権利情報"), m.get("利用条件")):
        text = sanitize_cell(text or "")
        if text and text not in rights:
            rights.append(text)
    for i, text in enumerate(rights):
        v.item(ITEM_RIGHTS, i, "subitem_rights", text)
        v.item(ITEM_RIGHTS, i, "subitem_rights_language", "ja")

    # 4 研究助成機関 / 5 研究費番号 → Fund Agency[n]
    for i, (funder, grant) in enumerate(pair_funding(m.get("研究助成機関"), m.get("研究費番号"))):
        sub_name = "subitem_1591282476079[0]"
        v.item(ITEM_FUND, i, f"{sub_name}.subitem_1592367160950", funder)
        if funder:
            v.item(ITEM_FUND, i, f"{sub_name}.subitem_1592462961468", detect_language(funder))
        v.item(ITEM_FUND, i, "subitem_1591282477038.subitem_1592367183821", grant)

    # 8 配布者 → Distributor[n]
    for i, name in enumerate(distributors):
        v.item(ITEM_DISTRIBUTOR, i, "subitem_1591320918354", "Distributor")
        v.item(ITEM_DISTRIBUTOR, i, "subitem_1592369405220", name)
        v.item(ITEM_DISTRIBUTOR, i, "subitem_1592369407829", "ja")

    # 9 寄託者 → Depositor[n]
    for i, name in enumerate(split_multi(m.get("寄託者"))):
        v.item(ITEM_DEPOSITOR, i, "subitem_1602143328410", name)
        v.item(ITEM_DEPOSITOR, i, "subitem_1602143330068", "ja")

    # 11 バージョン情報 → Version[0]
    if m.get("バージョン情報"):
        v.item(ITEM_VERSION, 0, "subitem_1591254914934", m["バージョン情報"])
        v.item(ITEM_VERSION, 0, "subitem_1591254915406", "ja")

    # 12 引用上の注意 → テキスト[0]
    if m.get("引用上の注意"):
        v.item(ITEM_TEXT, 0, "subitem_text_language", "ja")
        v.item(ITEM_TEXT, 0, "subitem_text_value", m["引用上の注意"])

    # 14 トピック → Topic CESSDA J / E
    e_idx = 0
    for i, label in enumerate(split_multi(m.get("トピック"))):
        topic = lookup_cessda_topic(label)
        v.item(ITEM_TOPIC_J, i, "subitem_1591258818994", topic.ja if topic else label)
        v.item(ITEM_TOPIC_J, i, "subitem_1591258819866", "ja")
        if topic is None:
            # CESSDA 語彙に無い語は統制語彙情報と英語を付与しない
            continue
        v.item(ITEM_TOPIC_J, i, "subitem_1591258820315", CESSDA_TOPIC_VOCAB_NAME)
        v.item(ITEM_TOPIC_J, i, "subitem_1591258820754", CESSDA_TOPIC_VOCAB_URI_WEKO)
        v.item(ITEM_TOPIC_J, i, "subitem_1591258821194", "Other")
        v.item(ITEM_TOPIC_E, e_idx, "subitem_1591259686465", topic.en)
        v.item(ITEM_TOPIC_E, e_idx, "subitem_1591259687137", "en")
        v.item(ITEM_TOPIC_E, e_idx, "subitem_1591259687617", CESSDA_TOPIC_VOCAB_NAME)
        v.item(ITEM_TOPIC_E, e_idx, "subitem_1591259688049", CESSDA_TOPIC_VOCAB_URI_WEKO)
        v.item(ITEM_TOPIC_E, e_idx, "subitem_1591259688513", "Other")
        e_idx += 1

    # 15 概要 → Summary DDI[0]
    if m.get("概要"):
        v.item(ITEM_SUMMARY, 0, "subitem_1551255577890", m["概要"])
        v.item(ITEM_SUMMARY, 0, "subitem_1551255592625", "ja")
        v.item(ITEM_SUMMARY, 0, "subitem_1551255637472", "Abstract")

    # 16 対象時期 → Time Period[0]=start, [1]=end
    start, end = parse_date_range(m.get("対象時期"))
    for i, (val, kind) in enumerate(p for p in ((start, "start"), (end, "end")) if p[0]):
        v.item(ITEM_TIME_PERIOD, i, "subitem_1592383206099", val)
        v.item(ITEM_TIME_PERIOD, i, "subitem_1592383207234", kind)

    # 17 調査日 → Date of Collection[0]=start, [1]=end
    start, end = parse_date_range(m.get("調査日"))
    for i, (val, kind) in enumerate(p for p in ((start, "start"), (end, "end")) if p[0]):
        v.item(ITEM_COLL_DATE, i, "subitem_1594611548575", val)
        v.item(ITEM_COLL_DATE, i, "subitem_1594611552366", kind)

    # 18 対象地域 → Geographic Coverage J
    for i, text in enumerate(split_multi(m.get("対象地域"))):
        v.item(ITEM_GEO_J, i, "subitem_1603960211179", text)
        v.item(ITEM_GEO_J, i, "subitem_1603960230666", "ja")

    # 19 観察単位 → 観察単位（日本語統制語彙）。英語統制語彙は対応表が無いため空欄
    for i, text in enumerate(split_multi(m.get("観察単位"))):
        v.item(ITEM_UNIT_J, i, "subitem_1591260766940", text)
        v.item(ITEM_UNIT_J, i, "subitem_1591260767605", "ja")

    # 20 母集団 → Universe[0]
    if m.get("母集団"):
        v.item(ITEM_UNIVERSE, 0, "subitem_1596608974429", m["母集団"])
        v.item(ITEM_UNIVERSE, 0, "subitem_1596608975087", "ja")

    # 21 データタイプ / 24 サンプリング方法 / 25 調査方法（日本語のみ）
    for name, item_id, sub_value, sub_lang in (
        ("データタイプ", ITEM_DATA_TYPE_J, "subitem_1591178723921", "subitem_1591178724426"),
        ("サンプリング方法", ITEM_SAMPLING_J, "subitem_1591178872873", "subitem_1591178873359"),
        ("調査方法", ITEM_COLL_METHOD_J, "subitem_1587459190791", "subitem_1587459200372"),
    ):
        for i, text in enumerate(split_multi(m.get(name))):
            v.item(item_id, i, sub_value, text)
            v.item(item_id, i, sub_lang, "ja")

    # 29 回収率 → Sampling Rate[0]
    if m.get("回収率"):
        v.item(ITEM_SAMPLING_RATE, 0, "subitem_1596609826487", m["回収率"])
        v.item(ITEM_SAMPLING_RATE, 0, "subitem_1596609827068", "ja")

    # 30 アクセス権 → Access E（COAR の英語ラベルと URI）。Access J は空欄
    a_idx = 0
    for label in split_multi(m.get("アクセス権")):
        access = lookup_access_right(label)
        if access is None:
            logger.warning("WEKO_TSV_UNKNOWN_ACCESS_RIGHT | value=%s", label)
            continue
        v.item(ITEM_ACCESS_E, a_idx, "subitem_1522650717957", "en")
        v.item(ITEM_ACCESS_E, a_idx, "subitem_1522650727486", access.label_en)
        v.item(ITEM_ACCESS_E, a_idx, "subitem_1522651041219", access.uri)
        a_idx += 1

    # 33 関連情報 → Related Study[n]
    for i, (title, url) in enumerate(parse_related_info(m.get("関連情報"))):
        title_sub = "subitem_1591282133676[0]"
        v.item(ITEM_RELATED_STUDY, i, f"{title_sub}.subitem_1592383512395", title)
        if title:
            v.item(ITEM_RELATED_STUDY, i, f"{title_sub}.subitem_1592383527694", "ja")
        if url:
            v.item(ITEM_RELATED_STUDY, i, "subitem_1592384097524.subitem_1592384631145", url)
            v.item(ITEM_RELATED_STUDY, i, "subitem_1592384097524.subitem_1592384631850", "URI")

    # 34 関連文献 → Related Publications[n]（文献情報全体をタイトルに入力）
    for i, text in enumerate(split_multi(m.get("関連文献"))):
        title_sub = "subitem_1591282556350[0]"
        v.item(ITEM_RELATED_PUBL, i, f"{title_sub}.subitem_1592383876755", text)
        v.item(ITEM_RELATED_PUBL, i, f"{title_sub}.subitem_1592383877507", "ja")

    # 13 URI → 制限公開用のコンテンツファイル[n].本文URL（アクセス=open_access）
    for i, url in enumerate(split_multi(m.get("URI"))):
        v.item(ITEM_CONTENT_FILE, i, "accessrole", "open_access")
        v.item(ITEM_CONTENT_FILE, i, "url.label", url)
        v.item(ITEM_CONTENT_FILE, i, "url.url", url)

    return v.data


def _expand_columns(header: list[list[str]], values: dict[str, str]) -> list[list[str]]:
    """繰り返し項目の列群を値の件数に合わせて複製した見出し行を返す。"""
    max_index: dict[str, int] = {}
    for path in values:
        gm = _GROUP_RE.match(path)
        if gm:
            key, idx = gm.group(1), int(gm.group(2))
            max_index[key] = max(max_index.get(key, 0), idx)

    paths = header[PATH_ROW]
    n_cols = len(paths)
    new_cols: list[int | tuple[int, int]] = []  # 元の列番号、または (元の列番号, 複製先インデックス)
    col = 0
    while col < n_cols:
        gm = _GROUP_RE.match(paths[col])
        if not gm or gm.group(2) != "0":
            new_cols.append(col)
            col += 1
            continue
        key = gm.group(1)
        block_start = col
        while col < n_cols and paths[col].startswith(f"{key}[0]"):
            col += 1
        block = list(range(block_start, col))
        new_cols.extend(block)
        for k in range(1, max_index.get(key, 0) + 1):
            new_cols.extend((c, k) for c in block)

    expanded: list[list[str]] = []
    for row_no, row in enumerate(header):
        if len(row) != n_cols:
            # 1行目（#ItemType 行）は列数が異なるためそのまま出力する
            expanded.append(list(row))
            continue
        new_row: list[str] = []
        for spec in new_cols:
            if isinstance(spec, int):
                new_row.append(row[spec])
                continue
            c, k = spec
            cell = row[c]
            if row_no == PATH_ROW:
                key = _GROUP_RE.match(cell).group(1)
                cell = cell.replace(f"{key}[0]", f"{key}[{k}]", 1)
            elif row_no == LABEL_ROW:
                cell = cell.replace("[0]", f"[{k}]", 1)
            new_row.append(cell)
        expanded.append(new_row)
    return expanded


def build_weko_tsv(
    results: list[dict],
    conversion_date: date | None = None,
    template_path: Path | str | None = None,
) -> str:
    """生成結果を WEKO3 JDCat(22) 一括登録用 TSV 文字列（UTF-8 BOM なし・LF 改行）に変換する。"""
    conversion_date = conversion_date or date.today()
    header = load_template(template_path)
    values = _build_values(results, conversion_date)
    rows = _expand_columns(header, values)
    paths = rows[PATH_ROW]
    unknown = set(values) - set(paths)
    if unknown:
        # 実装ミスで列が見つからない値は出力されないため警告する
        logger.warning("WEKO_TSV_UNMAPPED_PATHS | paths=%s", sorted(unknown))
    data_row = [values.get(p, "") for p in paths]
    rows.append(data_row)
    return "\n".join("\t".join(r) for r in rows) + "\n"
