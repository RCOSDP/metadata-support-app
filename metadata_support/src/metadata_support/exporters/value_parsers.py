"""アプリが生成した項目値（日本語文字列）を登録形式向けに分解・正規化する関数群。

アプリの生成値は以下の慣例に従う。
- 空文字または「該当なし」は不明（空欄扱い）
- 複数値は「;」（通常は「; 」）で連結
- 対象時期・調査日は ISO 8601 の区間「YYYY-MM-DD/YYYY-MM-DD」
  （「開始; 終了;」形式や「2015/1/31」のような表記も受け付ける）
- 関連情報は「タイトル (URL)」（「名称 URL」形式も受け付ける）
"""
from __future__ import annotations

import re

# 不明を表す値
UNKNOWN_VALUES = {"", "該当なし"}

_WS_CTRL_RE = re.compile(r"[\t\r\n]+")
_URL_RE = re.compile(r"https?://[^\s;；()（）<>\"']+")
_DOI_RE = re.compile(
    r"(?:https?://(?:dx\.)?doi\.org/|doi:\s*)(10\.\d{4,9}/[^\s;；()（）<>\"'、，。]+)",
    re.IGNORECASE,
)
_TRAILING_PUNCT = ".,。、，)）]」』"


def clean_value(value: object) -> str:
    """値を文字列化して前後の空白を除去する。不明値（空・「該当なし」）は空文字を返す。"""
    if value is None:
        return ""
    text = str(value).strip()
    if text in UNKNOWN_VALUES:
        return ""
    return text


def sanitize_cell(text: str) -> str:
    """セル内のタブ・改行を半角空白に置換し、前後の空白を除去する。"""
    if not text:
        return ""
    return _WS_CTRL_RE.sub(" ", str(text)).strip()


def split_multi(value: object) -> list[str]:
    """「;」区切りの複数値を分解する。空要素・「該当なし」は除外し、重複は1つにまとめる。"""
    text = clean_value(value)
    if not text:
        return []
    items: list[str] = []
    for part in re.split(r"[;；]", text):
        part = sanitize_cell(part)
        if part and part not in UNKNOWN_VALUES and part not in items:
            items.append(part)
    return items


# ──────────────────────────────
# 日付
# ──────────────────────────────
_DATE_PATTERNS = [
    re.compile(r"^(\d{4})$"),
    re.compile(r"^(\d{4})[-/.](\d{1,2})$"),
    re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$"),
    re.compile(r"^(\d{4})年$"),
    re.compile(r"^(\d{4})年(\d{1,2})月$"),
    re.compile(r"^(\d{4})年(\d{1,2})月(\d{1,2})日$"),
]


def normalize_date(text: str) -> str:
    """日付表記を ISO 8601（YYYY / YYYY-MM / YYYY-MM-DD）に正規化する。

    解釈できない場合は入力をそのまま（前後空白除去のみ）返す。
    """
    s = sanitize_cell(text).replace(" ", "")
    for pat in _DATE_PATTERNS:
        m = pat.match(s)
        if not m:
            continue
        parts = m.groups()
        year = parts[0]
        if len(parts) == 1:
            return year
        month = int(parts[1])
        if not 1 <= month <= 12:
            break
        if len(parts) == 2:
            return f"{year}-{month:02d}"
        day = int(parts[2])
        if not 1 <= day <= 31:
            break
        return f"{year}-{month:02d}-{day:02d}"
    return sanitize_cell(text)


def is_iso_date(text: str) -> bool:
    """ISO 8601 の年・年月・年月日のいずれかの形式かを判定する。"""
    return bool(re.match(r"^\d{4}(-\d{2}(-\d{2})?)?$", text or ""))


def _split_slash_range(s: str) -> list[str]:
    """「/」を含む文字列を、区間区切りの「/」と日付内の「/」を区別して分割する。"""
    parts = s.split("/")
    if len(parts) == 2:
        # 「2015/2016」「2015-01-31/2015-07-26」は区間、「2015/1」は年月とみなす
        # 「2015-01-31/..」「../2015」（ISO 8601 の開区間）も区間とみなす
        left, right = parts
        if re.match(r"^\d{4}", right) or ".." in (left, right):
            return [left, right]
        return [s]
    if len(parts) == 3:
        # 「2015/1/31」は単一の日付
        if all(re.match(r"^\d+$", p) for p in parts):
            return [s]
        # 「2015-01/2015/7」などの混在は、4桁年で始まる要素の位置で区切る
    if len(parts) == 4 and all(re.match(r"^\d+$", p) for p in parts):
        # 「2015/1/2015/7」（年月/年月）
        return ["/".join(parts[:2]), "/".join(parts[2:])]
    if len(parts) == 6:
        # 「2015/1/31/2015/7/26」（年月日/年月日）
        return ["/".join(parts[:3]), "/".join(parts[3:])]
    # その他は4桁年で始まる要素を区切り位置として再構成する
    groups: list[list[str]] = []
    for p in parts:
        if re.match(r"^\d{4}", p) or not groups:
            groups.append([p])
        else:
            groups[-1].append(p)
    return ["/".join(g) for g in groups]


def parse_date_range(value: object) -> tuple[str, str]:
    """期間を表す値を (開始, 終了) に分解し、それぞれ ISO 8601 に正規化する。

    - 「2015-01-31/2015-07-26」（ISO 8601 区間）
    - 「2015/1/31; 2015-07-26;」（「開始; 終了;」形式）
    - 「2015-01-31～2015-07-26」
    - 「2015」や「2015-01-31」（単一の値は開始=終了とみなす）
    複数の期間・日付が「;」で並ぶ場合は、最初の値を開始、最後の値を終了とする。
    """
    text = clean_value(value)
    if not text:
        return "", ""
    parts: list[str] = []
    for item in re.split(r"[;；]", text):
        item = item.strip()
        if not item or item in UNKNOWN_VALUES:
            continue
        if re.search(r"[～〜~]", item):
            parts.extend(p.strip() for p in re.split(r"[～〜~]", item))
        elif "/" in item:
            parts.extend(p.strip() for p in _split_slash_range(item))
        else:
            parts.append(item)
    if not parts:
        return "", ""
    start = normalize_date(parts[0]) if parts[0] not in ("..", "") else ""
    if len(parts) == 1:
        return start, start
    end = normalize_date(parts[-1]) if parts[-1] not in ("..", "") else ""
    return start, end


# ──────────────────────────────
# URL・DOI・関連情報
# ──────────────────────────────
def _strip_url(url: str) -> str:
    return url.rstrip(_TRAILING_PUNCT)


def extract_doi(text: object) -> str:
    """文字列中の DOI（https://doi.org/... または doi:...）を抽出し「10.xxxx/...」形式で返す。"""
    s = clean_value(text)
    m = _DOI_RE.search(s)
    if not m:
        return ""
    return _strip_url(m.group(1))


def doi_to_url(doi: str) -> str:
    """DOI を https://doi.org/ 形式の URL にする。"""
    return f"https://doi.org/{doi}" if doi else ""


def parse_related_info(value: object) -> list[tuple[str, str]]:
    """関連情報を (名称, URL) のリストに分解する。

    「名称 (URL)」「名称（URL）」「名称 URL」「URL」のいずれの形式も受け付ける。
    URL が無い要素は (名称, "") とする。
    """
    entries: list[tuple[str, str]] = []
    for item in split_multi(value):
        m = re.match(r"^(.*?)\s*[(（]\s*(https?://[^\s)）]+)\s*[)）]\s*$", item)
        if m:
            title, url = m.group(1).strip(), _strip_url(m.group(2))
        else:
            um = _URL_RE.search(item)
            if um:
                url = _strip_url(um.group(0))
                title = (item[: um.start()] + item[um.end():]).strip(" 　:：-")
            else:
                title, url = item, ""
        entries.append((title, url))
    return entries


# ──────────────────────────────
# 研究費番号
# ──────────────────────────────
_KAKEN_RE = re.compile(r"^(?:JP)?(\d{2}[0-9A-Z]{4,8})$")


def kaken_grant_url(grant_no: str) -> str:
    """科研費の課題番号（例: JP25000001）から KAKEN の課題 URL を生成する。

    先頭の「JP」を除いた番号が数字2桁で始まる英数字（例: 25000001, 19H05634）の場合のみ生成し、
    それ以外（JST の JPMJ... など）は空文字を返す。
    """
    m = _KAKEN_RE.match((grant_no or "").strip().upper())
    if not m:
        return ""
    return f"https://kaken.nii.ac.jp/grant/KAKENHI-PROJECT-{m.group(1)}/"


def pair_funding(funders_value: str | None, grants_value: str | None) -> list[tuple[str, str]]:
    """研究助成機関と研究費番号を組にする。

    件数が同じなら順に対応させ、助成機関が1件で研究費番号が複数ある場合は各番号に同じ機関を付ける。
    それ以外は位置で対応させ、足りない側は空欄とする。
    """
    funders = split_multi(funders_value)
    grants = split_multi(grants_value)
    if len(funders) == 1 and len(grants) > 1:
        funders = funders * len(grants)
    n = max(len(funders), len(grants))
    return [
        (funders[i] if i < len(funders) else "", grants[i] if i < len(grants) else "")
        for i in range(n)
    ]


# ──────────────────────────────
# 文字種・組織名の判定
# ──────────────────────────────
_JA_CHAR_RE = re.compile(r"[぀-ヿ㐀-鿿ｦ-ﾟ]")

# 組織名とみなす語（作成者・寄託者の名前タイプ判定に使用）
_ORG_KEYWORDS = (
    "研究会", "委員会", "研究所", "研究室", "研究班", "研究グループ", "研究センター", "センター",
    "大学", "学会", "協会", "機構", "財団", "法人", "株式会社", "有限会社", "（株）", "(株)",
    "省", "庁", "局", "役所", "自治体", "都庁", "県庁", "市役所", "アーカイブ", "プロジェクト",
    "事務局", "調査会", "会社", "銀行", "組合", "連合", "連盟", "協議会", "審議会", "社",
    "University", "Institute", "Committee", "Association", "Society", "Center", "Centre",
    "Foundation", "Agency", "Ministry", "Council", "Group", "Project", "Team", "Archive",
    "Inc.", "Ltd", "Corporation",
)


def has_japanese(text: str) -> bool:
    """日本語の文字（かな・漢字）を含むかを判定する。"""
    return bool(_JA_CHAR_RE.search(text or ""))


def detect_language(text: str) -> str:
    """値の言語コードを返す（日本語文字を含めば ja、それ以外は en）。"""
    return "ja" if has_japanese(text) else "en"


def looks_like_organization(name: str) -> bool:
    """名前が組織名らしいかを判定する（判定できない場合は False）。"""
    return any(k in (name or "") for k in _ORG_KEYWORDS)


# ──────────────────────────────
# 生成結果の参照
# ──────────────────────────────
def results_to_map(results: list[dict]) -> dict[str, str]:
    """アプリの生成結果リストを {項目名: 値} の辞書にする（不明値は空文字）。"""
    return {r["name"]: clean_value(r.get("value")) for r in results}
