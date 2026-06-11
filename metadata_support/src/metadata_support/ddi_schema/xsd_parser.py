"""DDI-Codebook 2.5 xsd から 34 項目の英語説明を機械抽出する。

xsd 内の各 <xs:element name="..."> は以下の均質なテンプレートを持つため、
<xhtml:div class="description">...</xhtml:div> を正規表現で抽出できる。
"""
from __future__ import annotations

import re
from typing import Any

# (no, name, ddi_path, leaf_element, 現行 METADATA_FIELDS の desc)
FIELDS: list[tuple[int, str, str, str, str]] = [
    (1,  "タイトル",           "stdyDscr->citation->titlStmt->titl",        "titl",          "調査名"),
    (2,  "その他のタイトル",    "stdyDscr->citation->titlStmt->altTitl",     "altTitl",       "タイトルの別名・副題・英訳タイトルなど"),
    (3,  "作成者",             "stdyDscr->citation->rspStmt->AuthEnty",     "AuthEnty",      "調査主体/調査代表者/作成者/著者"),
    (4,  "研究助成機関",        "stdyDscr->citation->prodStmt->fundAg",      "fundAg",        "調査研究に対して資金を提供した機関"),
    (5,  "研究費番号",          "stdyDscr->citation->prodStmt->grantNo",     "grantNo",       "助成金・研究費の識別番号"),
    (6,  "整理番号",           "stdyDscr->citation->titlStmt->IDNo",        "IDNo",          "論文や調査票に付与された一意の識別番号"),
    (7,  "編集者",             "stdyDscr->citation->prodStmt->producer",    "producer",      "調査したデータの加工・編集に責任を持つ組織"),
    (8,  "配布者",             "stdyDscr->citation->distStmt->distrbtr",    "distrbtr",      "調査したデータの配布・提供を行う組織"),
    (9,  "寄託者",             "stdyDscr->citation->distStmt->depositr",    "depositr",      "調査したデータをアーカイブに寄託した組織または個人"),
    (10, "権利情報",           "stdyDscr->citation->prodStmt->copyright",   "copyright",     "調査したデータのライセンス"),
    (11, "バージョン情報",      "stdyDscr->citation->verStmt->version",      "version",       "調査したデータセットの版番号・改訂履歴"),
    (12, "引用上の注意",       "stdyDscr->citation->biblCit",               "biblCit",       "入力した論文や調査票を引用する際の注意点"),
    (13, "URI",               "holdings",                                   "holdings",      "ランディングページURI"),
    (14, "トピック",           "stdyDscr->stdyInfo->subject->topcClas",     "topcClas",      "調査したプロジェクトや調査内容のトピック"),
    (15, "概要",              "stdyDscr->stdyInfo->abstract",              "abstract",      "調査概要/資料概要"),
    (16, "対象時期",           "stdyDscr->stdyInfo->sumDscr->timePrd",      "timePrd",       "調査を開始した日付と終了した日付"),
    (17, "調査日",             "stdyDscr->stdyInfo->sumDscr->collDate",     "collDate",      "調査を開始した日付と終了した日付"),
    (18, "対象地域",           "stdyDscr->stdyInfo->sumDscr->geogCover",    "geogCover",     "調査地域（都道府県全ての場合は日本（全国）とする）"),
    (19, "観察単位",           "stdyDscr->stdyInfo->sumDscr->anlyUnit",     "anlyUnit",      "個人、世帯、企業、イベント、地域などの単位"),
    (20, "母集団",             "stdyDscr->stdyInfo->sumDscr->universe",     "universe",      "調査の対象となる集団全体の定義"),
    (21, "データタイプ",        "stdyDscr->stdyInfo->sumDscr->dataKind",     "dataKind",      "調査で収集されたデータの種類・形式"),
    (22, "時間的設計",          "stdyDscr->method->dataColl->timeMeth",      "timeMeth",      "縦断か横断かなどの時間的な調査設計"),
    (23, "データ収集者",        "stdyDscr->method->dataColl->dataCollector", "dataCollector", "実際のデータ収集作業を担当した組織"),
    (24, "サンプリング方法",     "stdyDscr->method->dataColl->sampProc",      "sampProc",      "データのサンプリング方法"),
    (25, "調査方法",           "stdyDscr->method->dataColl->collMode",      "collMode",      "データ収集の方法・モード"),
    (26, "調査機器",           "stdyDscr->method->dataColl->resInstru",     "resInstru",     "回答を収集する機器、媒体"),
    (27, "ウェイト",           "stdyDscr->method->dataColl->weight",        "weight",        "標本の代表性を補正するための重み付け情報"),
    (28, "クリーニング処理",    "stdyDscr->method->dataColl->cleanOps",      "cleanOps",      "データの加工処理内容"),
    (29, "回収率",             "stdyDscr->method->anlyInfo->respRate",      "respRate",      "配布数に対する有効回収数の割合"),
    (30, "アクセス権",         "stdyDscr->dataAccs->setAvail->avlStatus",   "avlStatus",     "データのアクセス権の状態"),
    (31, "データファイルURI",   "stdyDscr->dataAccs->setAvail->accsPlac",    "accsPlac",      "データファイルの取得先を示すURL"),
    (32, "利用条件",           "stdyDscr->dataAccs->useStmt->conditions",   "conditions",    "データを利用する際に遵守すべき条件・規約"),
    (33, "関連情報",           "stdyDscr->othrStdyMat->relMat",             "relMat",        "当該調査研究に関連する補足資料・データセット"),
    (34, "関連文献",           "stdyDscr->othrStdyMat->relPubl",            "relPubl",       "当該調査研究に関連する公刊物・論文"),
]


def extract_english_descriptions(xsd_text: str) -> dict[str, dict[str, Any]]:
    """xsd 全文から各項目の英語タイトル・説明文を辞書で返す。キーは日本語項目名。"""
    results: dict[str, dict[str, Any]] = {}
    for no, name, path, leaf, current_desc in FIELDS:
        elem_pat = re.compile(
            rf'<xs:element\s+name="{re.escape(leaf)}"[^>]*>(.*?)</xs:element>',
            re.DOTALL,
        )
        m = elem_pat.search(xsd_text)
        record: dict[str, Any] = {
            "no": no,
            "ddi_path": path,
            "leaf": leaf,
            "current_desc": current_desc,
            "en_title": "",
            "en_description": "",
        }
        if not m:
            record["error"] = "element declaration not found"
            results[name] = record
            continue
        block = m.group(1)
        title_m = re.search(
            r'<xhtml:h1[^>]*class="element_title"[^>]*>(.*?)</xhtml:h1>',
            block, re.DOTALL,
        )
        desc_m = re.search(
            r'<xhtml:div[^>]*class="description"[^>]*>(.*?)</xhtml:div>',
            block, re.DOTALL,
        )
        if title_m:
            record["en_title"] = _normalize(title_m.group(1))
        if desc_m:
            record["en_description"] = _normalize(desc_m.group(1))
        results[name] = record
    return results


def _normalize(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
