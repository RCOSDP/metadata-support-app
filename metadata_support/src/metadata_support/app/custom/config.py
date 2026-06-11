import os
import re

from metadata_support.extract.bedrock_extractor import _BedrockModel
from metadata_support.ddi_schema import build_field_groups, build_metadata_fields

# working_dir
HOME_DIR = "/workspaces/nii-metadata-support"
OAI_DB_BASE_PATH = os.environ.get("OAI_DB_PATH", "./oai_repo_db")

# Models
MODELS = {
    "Claude Haiku 4.5": {
        "model_id": "bedrock/jp.anthropic.claude-haiku-4-5-20251001-v1:0",
        "region": "ap-northeast-1",
    },
    "Claude Sonnet 4.5": {
        "model_id": "bedrock/jp.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "region": "ap-northeast-1",
    },
    "Claude Sonnet 4.6": {
        "model_id": "bedrock/jp.anthropic.claude-sonnet-4-6",
        "region": "ap-northeast-1",
    },
}
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
chunk_model = _BedrockModel(
            model_name="bedrock/jp.anthropic.claude-haiku-4-5-20251001-v1:0",
            aws_region="ap-northeast-1",
        )
vocab_model = _BedrockModel(
            model_name="bedrock/jp.anthropic.claude-sonnet-4-6",
            aws_region="ap-northeast-1",
        )

USE_LLM_SUMMARIZED_DESC = True
FORCE_REBUILD_DESC = False

METADATA_FIELDS = build_metadata_fields(
    use_llm_summary=USE_LLM_SUMMARIZED_DESC,
    force_rebuild=FORCE_REBUILD_DESC,
)

FIELD_NAMES_STR = "\n".join(
    f"- **{field['name']}**: {field['desc']}" for field in METADATA_FIELDS
)

_FIELD_DESC_MAP = {f["name"]: f["desc"] for f in METADATA_FIELDS}

FIELD_GROUPS_FLG = False

if FIELD_GROUPS_FLG:
    FIELD_GROUPS = [
        {
            "name": "書誌・権利",
            "fields": [
                "タイトル", "その他のタイトル", "作成者", "概要",
                "研究助成機関", "研究費番号", "編集者", "配布者", "寄託者",
                "権利情報", "バージョン情報", "整理番号", "URI", "引用上の注意",
                "アクセス権", "データファイルURI", "利用条件",
            ],
        },
        {
            "name": "調査設計・方法",
            "fields": [
                "対象時期", "調査日", "対象地域", "観察単位",
                "母集団", "時間的設計", "データ収集者", "サンプリング方法", "調査方法", "調査機器",
                "ウェイト", "クリーニング処理", "回収率",
            ],
        },
        {
            "name": "トピック・データタイプ",
            "fields": ["トピック", "データタイプ"],
        },
        {
            "name": "外部情報",
            "fields": ["関連情報", "関連文献"],
        },
    ]
    for _group in FIELD_GROUPS:
        _group["fields_str"] = "\n".join(
            f"- **{name}**: {_FIELD_DESC_MAP[name]}" for name in _group["fields"]
        )
else:
    FIELD_GROUPS = build_field_groups(METADATA_FIELDS)

# 統制語彙
CONTROLLED_VOCABULARY = {
    "トピック": [
                "人口", "人口センサス", "人口移動", "罹病と死亡", "経済", "消費と消費者行動", "経済状況と経済指標", "経済政策、歳出と歳入",
                "経済体制と経済発展", "所得、財産、投資・貯蓄", "教育", "義務教育と就学前教育", "教育政策", "高等教育", "生涯・継続教育", "職業教育と職業訓練",
                "健康", "食事と栄養", "薬物乱用、アルコール、喫煙", "健康一般とウェルビーイング", "ヘルスケアサービスと政策", "薬と治療", "労働衛生",
                "フィットネスと運動", "公衆衛生", "リプロダクティブヘルス", "徴候と症状、病気の状態", "疾病、障害、健康状態", "創傷と損傷", "歴史",
                "住宅と土地利用", "住宅", "土地利用と計画", "労働と雇用", "職員研修", "雇用", "労働と雇用政策", "労使関係・争議", "退職", "失業", "労働条件",
                "法、犯罪、法制度","犯罪と法の執行","立法と法制度","メディア、コミュニケーション、言語","情報社会","言語と言語学","メディア","広報（PR)",
                "自然環境","エネルギーと天然資源","環境と保全","自然景観","植物と動物","政治","紛争、安全保障、平和","選挙","政府、政治制度、組織",
                "国際政治と国際機構","政治行動と政治的態度","政治的イデオロギー","心理","科学と技術","バイオテクノロジー","情報技術","社会階層と社会集団",
                "子ども","高齢者","エリートとリーダーシップ","平等、不平等、社会的排除","家族生活と結婚","ジェンダーと性別役割","マイノリティー",
                "社会移動と職業移動","若者","社会福祉政策と制度","社会福祉制度・構造","特定の社会サービスの利用と供給","社会と文化","地域社会、都市生活、農村生活",
                "文化的活動と参加","文化的アイデンティティーとナショナルアイデンティティー","余暇、観光、スポーツ","宗教と価値観","社会行動と社会的態度",
                "社会変動","社会状況と社会指標","生活時間","貿易、産業、市場","農林水産業","ビジネス・産業経営と組織","外国貿易","運輸と旅行","その他"
            ],
    "データタイプ": [
                "量的調査: ミクロデータ", "量的調査: 集計データ、統計表", "量的調査: その他",
                "公的統計: ミクロデータ", "公的統計: 集計データ、統計表", "公的統計: その他",
                "質的調査: 音声データ", "質的調査: 映像データ", "質的調査: 文書データ",
                "史資料: テキスト", "史資料: 画像", "史資料: 音声", "史資料: 映像", "史資料: その他",
                "出版物: テキスト", "出版物: 画像", "出版物: 音声", "出版物: 映像", "出版物: その他",
                "事項","データベース","その他"
            ],
    "サンプリング方法": [
                "母集団/ 全数調査","確率","確率: 単純無作為抽出","確率: 系統抽出","確率: 層別抽出","確率: 層別抽出: 比例割当法",
                "確率: 層別抽出: 非比例割当法","確率: 集落抽出","確率: 集落抽出: 単純無作為","確率: 集落抽出: 層別無作為",
                "確率: 多段抽出","非確率","非確率: 利用可能性","非確率: 有意抽出","非確率: クオータ抽出",
                "非確率: 回答者支援抽出","混合確率と非確率","その他"
            ],
    "調査方法": [
                "インタビュー","個別面接法","個別面接法：コンピューター支援型（CAPI/CAMI）","個別面接法： 紙と鉛筆 （PAPI）",
                "電話法","電話法：コンピュータ支援 （CATI）","電子メール・インタビュー","Webベースのインタビュー",
                "自記式調査票","自記式調査票：電子メール","自記式調査票：紙","自記式調査票：メッセージング（SMS/MMS）",
                "自記式調査票：Webベース（CAWI）","自記式調査票：コンピュータ支援型（CASI）","フォーカスグループ",
                "対面フォーカスグループ","電話フォーカスグループ","オンライン・フォーカスグループ",
                "自記式の文章／日記","自記式の文章／日記：電子メール","自記式の文章／日記：紙","自記式の文章／日記：Webベース",
                "観察","フィールド観察","参与観察","非参与型のフィールド観察","実験室観察",
                "参与型実験室観察","非参与型実験室観察","コンピューターによる観察",
                "実験","実験室実験","フィールド／介入実験","Webベース実験",
                "記録","内容コーディング","トランスクリプション","編集／統合","要約","集計","シミュレーション",
                "測定・テスト","教育測定・テスト","肉体的／物理的測定・テスト","心理測定・テスト","その他"
            ],
    "アクセス権": ["オープンアクセス","制約付きアクセス","メタデータのみ","エンバーゴ期間中"]
}
CONTROLLED_VOCAB_STR = "\n".join(
    f"- **{field_name}**: {', '.join(vocab_list)}"
    for field_name, vocab_list in CONTROLLED_VOCABULARY.items()
)
VERBATIM_FIELDS = {"トピック", "データタイプ", "サンプリング方法", "調査方法", "アクセス権"}

# 調査票の質問部分を除去
_QUESTION_MARKER_RE = re.compile(r'(?:^|\n)\s*[問目闘]\s*\d{1,2}[.．]?\s', re.MULTILINE)
_QUESTION_THRESHOLD = 5  # 質問マーカーがこの数以上なら調査票と判定

# その他
EXTRACT_MAX_WORKERS = 30
CONFIDENCE_THRESHOLD = 3

FIELDS_WITH_DESC = {f["name"] for f in METADATA_FIELDS}
