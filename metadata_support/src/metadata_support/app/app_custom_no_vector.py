import html
import logging
import time
import os
import streamlit as st
import json
import fitz
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from metadata_support.extract.bedrock_extractor import (
    KNOWN_PURPOSES,
    BedrockExtractor,
    calculate_cost,
    calculate_cost_by_purpose,
    reset_token_usage,
)
from metadata_support.wrappers.literature.oai_pmh_wrapper import OAI_REPOSITORIES

from metadata_support.app.custom.config import (
    HOME_DIR,
    OAI_DB_BASE_PATH,
    MODELS,
    chunk_model,
    vocab_model,
    METADATA_FIELDS,
    FIELD_GROUPS,
    CONTROLLED_VOCABULARY,
    VERBATIM_FIELDS,
    EXTRACT_MAX_WORKERS,
    CONFIDENCE_THRESHOLD,
    FIELDS_WITH_DESC,
)
from metadata_support.app.custom.bm25_preload import BM25PreloadStore
from metadata_support.app.custom.text_extractor import PDFTextExtractor
from metadata_support.app.custom.chunk_pipeline import ChunkPipeline
from metadata_support.app.custom.output import OutputWriter
from metadata_support.app.custom.llm import LLMProcessor
from metadata_support.wrappers.literature.jdcat_search import (
    hits_to_ddi_records,
    search_jdcat_all_pages,
)

logger = logging.getLogger(__name__)

# JDCAT キーワード検索の設定
JDCAT_SEARCH_SIZE = 20

# 用途ラベル → ログ表示用ラベル
PURPOSE_LABELS = {
    "survey_name": "調査名抽出",
    "annotate": "項目付与",
    "rerank_external": "外部レコード再ランキング",
    "vocab_convert": "統制語彙変換",
    "field_extract": "項目抽出",
    "unknown": "その他",
}

def _format_usage(usage: dict | None) -> str:
    """LLM 呼び出し1回分の usage を 1行の文字列にする (ログ表記用)。"""
    if not usage:
        return "(LLM未使用)"
    return (
        f"input={usage.get('input_tokens', 0):,}tok | "
        f"output={usage.get('output_tokens', 0):,}tok | "
        f"cost=${usage.get('total_cost_usd', 0.0):.4f} | "
        f"model={usage.get('model_id', '-')}"
    )

llm_processor = LLMProcessor(
    chunk_model=chunk_model,
    vocab_model=vocab_model,
    verbatim_fields=VERBATIM_FIELDS,
)

# =========================
# アプリ起動時にJGSSDDS/RUDA/CSRDAの全データをローカルJSONに事前格納し、BM25インデックスを構築
# =========================
OAI_PRELOAD_REPOS = ["JGSSDDS", "RUDA", "CSRDA"]
OAI_PRELOAD_DB_PATH = os.path.join(OAI_DB_BASE_PATH, "preloaded")

@st.cache_resource(show_spinner="外部リポジトリのデータを取得中...")
def _preload_oai_repos():
    """JGSSDDS/RUDA/CSRDAの全レコードを取得しローカルJSONに格納、BM25インデックスを構築する。

    リポジトリ単位でキャッシュを保存するため、一部リポジトリのフェッチが
    失敗した場合でも次回起動時に未取得分のみ再フェッチされる。
    """
    from metadata_support.wrappers.literature.oai_pmh_wrapper import fetch_oai_records
    store = BM25PreloadStore(path=OAI_PRELOAD_DB_PATH)
    for repo_name in OAI_PRELOAD_REPOS:
        if store.has_repo_data(repo_name):
            store.load_repo(repo_name)
            continue
        repo_config = OAI_REPOSITORIES[repo_name]
        logger.info("OAI_PRELOAD_FETCH | repo=%s | max=ALL", repo_name)
        records = fetch_oai_records(
            base_url=repo_config["url"],
            metadata_prefix=repo_config["metadata_prefix"],
            max_records=None,
        )
        for rec in records:
            rec["source_repository"] = repo_name
            rec["id"] = f"{repo_name}:{rec['id']}"
        logger.info("OAI_PRELOAD_FETCH_DONE | repo=%s | records=%d", repo_name, len(records))
        if records:
            store.save_repo(repo_name, records)
            store.add_records(records)
        else:
            logger.warning(
                "OAI_PRELOAD_FETCH_EMPTY | repo=%s | キャッシュ未作成、次回起動時に再フェッチ",
                repo_name,
            )
    logger.info("OAI_PRELOAD_TOTAL | records=%d", len(store))
    store.build_index()
    return store

oai_preloaded_store = _preload_oai_repos()

# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="Metadata Generator", layout="wide")

st.markdown("""
<style>
    .main-title { background: #ffffff; color: #333333; padding: 15px 25px; border-radius: 8px; border: 1px solid #e0e0e0; margin-bottom: 20px; font-size: 24px; font-weight: bold; }
    .field-label { font-weight: bold; color: #333333; font-size: 15px; margin-top: 12px; margin-bottom: 2px; }
    .field-value { padding: 6px 12px; background: #ffffff; border-left: 3px solid #999999; margin-bottom: 8px; font-size: 14px; color: #333333; }
    .status-processing { background: #ffffff; border: 1px solid #999999; border-radius: 5px; padding: 10px 15px; text-align: center; font-weight: bold; color: #333333; margin: 10px 0; }
    .status-done { background: #ffffff; border: 1px solid #333333; border-radius: 5px; padding: 10px 15px; text-align: center; font-weight: bold; color: #333333; margin: 10px 0; }
    div[data-testid="stButton"] > button[kind="primary"] { background-color: #2e7d32; border-color: #2e7d32; color: #ffffff; }
    div[data-testid="stButton"] > button[kind="primary"]:hover { background-color: #1b5e20; border-color: #1b5e20; }
    div[data-testid="stDownloadButton"] > button { background-color: #1565c0; border-color: #1565c0; color: #ffffff; }
    div[data-testid="stDownloadButton"] > button:hover { background-color: #0d47a1; border-color: #0d47a1; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Social Science Metadata Generator</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader("PDFファイルをアップロード", type=["pdf"], accept_multiple_files=True)

# =========================
# サイドバー
# =========================
with st.sidebar:
    st.header("生成前準備")

    st.subheader("LLMモデル")
    selected_model = st.selectbox("LLMモデル：", list(MODELS.keys()), index=0)

    st.subheader("外部データの利用")
    use_oai_search = st.checkbox(
        "外部データから類似データを検索して例示に使用",
        value=False,
        help="タイトルチャンクを用いて外部リポジトリから類似メタデータを検索し、LLMプロンプトの例示として追加します。",
    )
    selected_repos = []
    if use_oai_search:
        selected_repos = st.multiselect(
            "検索対象リポジトリ：",
            list(OAI_REPOSITORIES.keys()),
            default=list(OAI_REPOSITORIES.keys()),
            format_func=lambda k: f"{k} ({OAI_REPOSITORIES[k]['label']})",
        )

# =========================
# メイン処理
# =========================
if uploaded_files:

    model_config = MODELS[selected_model]
    extractor = BedrockExtractor(
        model_name=model_config["model_id"],
        aws_region=model_config["region"],
    )
    llm_processor.extractor = extractor

    run_button = st.button("メタデータ生成開始", type="primary", use_container_width=True)

    if "all_results" not in st.session_state:
        st.session_state.all_results = {}

    if run_button:
        status_placeholder = st.empty()
        status_placeholder.markdown('<div class="status-processing">処理中</div>', unsafe_allow_html=True)

        start_time = time.time()
        llm_processor.start_time = start_time
        reset_token_usage()

        # =======================
        # テキスト抽出
        # =======================
        all_pdf_bytes, all_full_texts, combined_text = PDFTextExtractor(uploaded_files).extract()

        # ===========================================
        # チャンク分割 + フィールド判定（ChunkPipeline で一括実行）
        # ===========================================
        _pipeline_result = ChunkPipeline(
            all_full_texts=all_full_texts,
            llm_processor=llm_processor,
            field_groups=FIELD_GROUPS,
            batch_size=50,
            max_workers=20,
            excluded_group_names={"外部情報"},
        ).run()
        split_chunks_map = _pipeline_result["split_chunks_map"]
        split_text_logs = _pipeline_result["split_text_logs"]
        split_to_file = _pipeline_result["split_to_file"]
        split_position_info = _pipeline_result["split_position_info"]
        file_ordered_chunks = _pipeline_result["file_ordered_chunks"]
        file_list = _pipeline_result["file_list"]
        NUM_SPLITS = _pipeline_result["num_splits"]
        total_chunks_step1 = _pipeline_result["total_chunks"]
        all_chunks = _pipeline_result["all_chunks"]
        split_logs = _pipeline_result["split_logs"]
        llm_processor.num_splits = NUM_SPLITS

        # 調査名抽出（各PDFの先頭5000文字から、JDCATベクトル検索のクエリに使用）
        SURVEY_NAME_HEAD_CHARS = 5000
        survey_names: list[str] = []
        survey_name_info = None
        if use_oai_search and "JDCAT" in selected_repos:
            first_parts = []
            for uploaded_file in sorted(uploaded_files, key=lambda f: f.name):
                pdf_bytes = all_pdf_bytes.get(uploaded_file.name)
                if not pdf_bytes:
                    continue
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                raw_text = ""
                for page in doc:
                    raw_text += page.get_text()
                    if len(raw_text) >= SURVEY_NAME_HEAD_CHARS:
                        break
                doc.close()
                first_parts.append(raw_text[:SURVEY_NAME_HEAD_CHARS])
            if first_parts:
                combined_first_chunk = "\n\n".join(first_parts)
                source_label = ", ".join(f.name for f in uploaded_files)
                try:
                    survey_name_info = llm_processor.extract_survey_name(
                        combined_first_chunk, source_label
                    )
                    survey_names = list(survey_name_info.get("names", []))
                except Exception as e:
                    logger.error("SURVEY_NAME_EXTRACT_ERROR | error=%s", e)
            logger.info("SURVEY_NAME | count=%d | names=%s",
                        len(survey_names), "; ".join(survey_names)[:200])

        # チャンク分割結果をDataFrameに変換
        chunks_df = pd.DataFrame(
            [{"ドキュメント内通番": c.get("doc_chunk_no", 0), "ソースファイル": c.get("source_file", ""), "該当項目": c["field"], "テキスト": c["chunk_text"], "確信度": c["confidence"], "種別": "原文"} for c in all_chunks]
        )
        logger.info("CHUNKS_DF | shape=%s\n%s", chunks_df.shape, chunks_df.to_string(max_rows=10))

        # ログ出力（フィルタ前の全チャンクを記録）
        log_timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_output_dir = f"{HOME_DIR}/output/{log_timestamp}"
        os.makedirs(log_output_dir, exist_ok=True)
        filenames_for_log = ", ".join(f.name for f in uploaded_files)

        log2_path = os.path.join(log_output_dir, "log2_chunks.txt")
        with open(log2_path, "w", encoding="utf-8") as lf:
            lf.write(f"入力ファイル名: {filenames_for_log}\n")
            lf.write(f"チャンク総数（フィルタ前）: {len(chunks_df)}\n")
            lf.write("=" * 80 + "\n\n")
            for i, row in chunks_df.iterrows():
                lf.write(f"--- チャンク {i + 1} (ドキュメント内通番: {int(row['ドキュメント内通番'])}, 該当項目: {row['該当項目']}, 確信度: {row['確信度']:.2f}) ---\n")
                lf.write(row["テキスト"])
                lf.write("\n\n")

        # 「該当項目なし」「関連文献」を除外
        before_count = len(chunks_df)
        chunks_df = chunks_df[
            ~chunks_df["該当項目"].isin(["該当項目なし", "関連情報", "関連文献"])
        ].reset_index(drop=True)
        logger.info("CHUNK_FILTER | before=%d | after=%d | removed=%d", before_count, len(chunks_df), before_count - len(chunks_df))

        # =============================
        # 統制語彙変換（該当項目のみ）
        # =============================
        vocab_rows = []
        vocab_logs = []
        vocab_final_values = {}  # 統制語彙項目はLLMを通さず、この値をそのままメタデータとして出力
        vocab_targets = chunks_df[chunks_df["該当項目"].isin(CONTROLLED_VOCABULARY.keys())]
        if not vocab_targets.empty:
            # 同じ項目の確信度が最大値のチャンクテキストのみをまとめて変換
            field_combined = {}
            field_max_confidence = {}
            for field_name, group in vocab_targets.groupby("該当項目"):
                # トピックは確信度2以上を全て使用、データタイプは確信度関係なく最大値のみ、それ以外は閾値以上の最大値のみ
                if field_name == "トピック":
                    filtered = group[group["確信度"] >= 2]
                elif field_name == "データタイプ":
                    filtered = group
                else:
                    filtered = group[group["確信度"] >= CONFIDENCE_THRESHOLD]
                if filtered.empty:
                    continue
                if field_name == "トピック":
                    selected = filtered
                else:
                    max_conf = filtered["確信度"].max()
                    selected = filtered[filtered["確信度"] == max_conf]
                field_combined[field_name] = "\n---\n".join(selected["テキスト"].tolist())
                field_max_confidence[field_name] = filtered["確信度"].max()

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {}
                for field_name, combined in field_combined.items():
                    vocab_list = CONTROLLED_VOCABULARY[field_name]
                    future = executor.submit(llm_processor.convert_to_vocab, field_name, combined, vocab_list)
                    futures[future] = field_name

                for future in as_completed(futures):
                    field_name = futures[future]
                    result = future.result()
                    converted = result["converted"]
                    vocab_logs.append({
                        "field": field_name,
                        "original": field_combined[field_name],
                        "prompt": result["prompt"],
                        "response": result["response"],
                        "usage": result.get("usage"),
                    })
                    if converted and converted != "該当なし":
                        vocab_final_values[field_name] = converted
                        vocab_rows.append({
                            "ドキュメント内通番": 0,
                            "ソースファイル": "",
                            "該当項目": field_name,
                            "テキスト": converted,
                            "確信度": field_max_confidence[field_name],
                            "種別": "変換済み",
                        })
                        logger.info("VOCAB_CONVERT | field=%s | original=%s | converted=%s",
                                    field_name, field_combined[field_name][:80], converted)

        if vocab_rows:
            vocab_df = pd.DataFrame(vocab_rows)
            chunks_df = pd.concat([chunks_df, vocab_df], ignore_index=True)
            logger.info("VOCAB_CONVERT_DONE | added=%d | total=%d", len(vocab_rows), len(chunks_df))
            logger.warning("ELAPSED | VOCAB_CONVERT | elapsed=%.1fs", time.time() - start_time)
        else:
            logger.info("VOCAB_CONVERT_DONE | no conversions needed")
            logger.warning("ELAPSED | VOCAB_CONVERT | elapsed=%.1fs", time.time() - start_time)

        # =============================
        # チャンクを該当項目ごとにマッピング
        # =============================

        # field_chunks_map: {項目名: [(テキスト, 通番), ...]}
        field_chunks_map = {}
        for field_name, group in chunks_df.groupby("該当項目"):
            # 統制語彙がある項目は変換済みチャンクのみ使用
            if field_name in CONTROLLED_VOCABULARY:
                group = group[group["種別"] == "変換済み"]
            # 確信度がCONFIDENCE_THRESHOLD以上のチャンクのうち、最大値のものを全て使用
            filtered = group[group["確信度"] >= CONFIDENCE_THRESHOLD]
            if not filtered.empty:
                max_conf = filtered["確信度"].max()
                filtered = filtered[filtered["確信度"] == max_conf]
            field_chunks_map[field_name] = list(zip(filtered["テキスト"].tolist(), filtered["ドキュメント内通番"].astype(int).tolist(), filtered["ソースファイル"].tolist()))
        # 調査機器には調査方法の原文チャンクを使用
        method_original = chunks_df[
            (chunks_df["該当項目"] == "調査方法") & (chunks_df["種別"] == "原文")
        ]
        method_filtered = method_original[method_original["確信度"] >= CONFIDENCE_THRESHOLD]
        if not method_filtered.empty:
            method_max = method_filtered["確信度"].max()
            method_filtered = method_filtered[method_filtered["確信度"] == method_max]
        field_chunks_map["調査機器"] = list(zip(method_filtered["テキスト"].tolist(), method_filtered["ドキュメント内通番"].astype(int).tolist(), method_filtered["ソースファイル"].tolist()))

        # 調査日と対象時期は両方のチャンクを統合して共通テキストにする
        date_fields = ["調査日", "対象時期"]
        date_chunks = chunks_df[
            (chunks_df["該当項目"].isin(date_fields)) & (chunks_df["種別"] == "原文")
        ]
        date_filtered = date_chunks[date_chunks["確信度"] >= CONFIDENCE_THRESHOLD]
        if not date_filtered.empty:
            date_max = date_filtered["確信度"].max()
            date_filtered = date_filtered[date_filtered["確信度"] == date_max]
        shared_date_items = list(zip(date_filtered["テキスト"].tolist(), date_filtered["ドキュメント内通番"].astype(int).tolist(), date_filtered["ソースファイル"].tolist()))
        for f in date_fields:
            field_chunks_map[f] = shared_date_items

        logger.info("FIELD_CHUNKS_MAP | threshold=%d | strategy=max_confidence | fields=%s", CONFIDENCE_THRESHOLD, {k: len(v) for k, v in field_chunks_map.items()})

        # =============================
        # 該当項目マッチング → field_chunk_texts 構築
        # =============================
        def _build_chunk_with_context(chunk_text, doc_chunk_no, source_file):
            """マッチしたチャンクに同一ファイル内の前後2チャンクの全文を文脈として付与する。"""
            ordered = file_ordered_chunks.get(source_file, [""])
            parts = [f"【該当チャンク】\n{chunk_text}"]
            # 前方文脈（同一ファイル内、最大2チャンク）
            prev_ctx = []
            for offset in range(2, 0, -1):
                prev_no = doc_chunk_no - offset
                if 1 <= prev_no < len(ordered):
                    prev_ctx.append(f"[前方文脈 チャンク{prev_no}]\n{ordered[prev_no]}")
            # 後方文脈（同一ファイル内、最大2チャンク）
            next_ctx = []
            for offset in range(1, 3):
                next_no = doc_chunk_no + offset
                if 1 <= next_no < len(ordered):
                    next_ctx.append(f"[後方文脈 チャンク{next_no}]\n{ordered[next_no]}")
            if prev_ctx or next_ctx:
                parts.append("＜前後の文脈＞")
                parts.extend(prev_ctx)
                parts.extend(next_ctx)
            return "\n".join(parts)

        field_chunk_texts = {}
        for field in METADATA_FIELDS:
            field_name = field["name"]
            matched_items = field_chunks_map.get(field_name, [])
            # 「その他のタイトル」は「タイトル」のチャンクも統合して判定に使う
            if field_name == "その他のタイトル":
                seen = set()
                merged = []
                for item in field_chunks_map.get("タイトル", []) + matched_items:
                    key = (item[0], item[1], item[2])
                    if key in seen:
                        continue
                    seen.add(key)
                    merged.append(item)
                matched_items = merged
            if matched_items:
                texts_with_ctx = []
                for chunk_text, doc_no, src_file in matched_items:
                    if doc_no > 0 and src_file:
                        texts_with_ctx.append(_build_chunk_with_context(chunk_text, doc_no, src_file))
                    else:
                        texts_with_ctx.append(chunk_text)
                field_chunk_texts[field_name] = "\n\n---\n\n".join(texts_with_ctx)
                logger.info("FIELD_MATCH | field=%s | chunks=%d", field_name, len(matched_items))
            else:
                field_chunk_texts[field_name] = ""
                logger.info("FIELD_MATCH | field=%s | chunks=0 (no match)", field_name)

        metadata_obj = {}
        all_field_prompts = {}
        all_field_responses = {}
        all_field_usage = {}

        # 編集者・配布者・寄託者はテキスト内にキーワードがなければLLM呼び出しをスキップ
        KEYWORD_REQUIRED_FIELDS = {
            "編集者": ["編集"],
            "配布者": ["配布"],
            "寄託者": ["寄託"],
        }
        for fname, ftxt in list(field_chunk_texts.items()):
            if fname in KEYWORD_REQUIRED_FIELDS:
                keywords = KEYWORD_REQUIRED_FIELDS[fname]
                if not any(kw in ftxt for kw in keywords):
                    field_chunk_texts[fname] = ""
                    logger.info("FIELD_SKIP_NO_KEYWORD | field=%s | keywords=%s", fname, keywords)

        field_desc_map = {f["name"]: f["desc"] for f in METADATA_FIELDS}

        # =============================
        # Phase 1: タイトル・その他のタイトルを先に抽出
        # =============================
        PHASE1_FIELDS = ["タイトル", "その他のタイトル"]
        llm_processor.oai_background_text = None
        llm_processor.external_records = []
        llm_processor.current_survey_name = ""
        with ThreadPoolExecutor(max_workers=max(1, len(PHASE1_FIELDS))) as executor:
            phase1_futures = {
                executor.submit(
                    llm_processor.extract_field,
                    fname,
                    field_desc_map[fname] if fname in FIELDS_WITH_DESC else "",
                    field_chunk_texts.get(fname, ""),
                ): fname
                for fname in PHASE1_FIELDS
            }
            for future in as_completed(phase1_futures):
                field_name, value, prompt_text, raw_response, usage = future.result()
                metadata_obj[field_name] = value
                all_field_prompts[field_name] = prompt_text
                all_field_responses[field_name] = raw_response
                all_field_usage[field_name] = usage

        # =============================
        # OAI-PMH 外部リポジトリ検索
        # =============================
        oai_background_text = None
        jdcat_keyword_search_log = []
        jdcat_vector_search_log = []
        preload_bm25_search_log = []
        external_records = []  # 項目別フィルタ用の構造化レコード
        jdcat_has_hits = False  # JDCAT で external_records に追加できたかのフラグ

        # 外部データの対象時期・調査日は "A; B" 形式を "A/B" 形式に変換
        DATE_RANGE_FIELDS = {"対象時期", "調査日"}
        def _normalize_external_fields(fields):
            normalized = {}
            for k, v in (fields or {}).items():
                if not v:
                    continue
                if k in DATE_RANGE_FIELDS and isinstance(v, str) and ";" in v:
                    parts = [p.strip() for p in v.split(";") if p.strip()]
                    normalized[k] = "/".join(parts) if parts else v
                else:
                    normalized[k] = v
            return normalized

        if use_oai_search and selected_repos:
            try:
                example_parts = []

                # JDCAT (DDI) / JGSSDDS (jpcoar) / RUDA (jpcoar) / CSRDA (oai_dc) いずれも
                # ``rec["fields"]`` が日本語項目名の dict として揃っている前提
                def _rec_to_json(rec, source):
                    """レコードをメタデータ項目名のJSON文字列に変換する。"""
                    fields = {k: v for k, v in (rec.get("fields") or {}).items() if v}
                    for fname in fields:
                        if isinstance(fields[fname], str):
                            fields[fname] = re.sub(r'[\r\n]+', ' ', fields[fname]).strip()
                    fields["source_repository"] = source
                    return json.dumps(fields, ensure_ascii=False, indent=2)

                # 検索キーワード = Phase1の タイトル/その他のタイトル + LLM抽出の調査名
                keyword_strings = []
                seen_kw = set()
                for field in PHASE1_FIELDS:
                    value = str(metadata_obj.get(field, "")).strip()
                    if not value or value == "該当なし":
                        continue
                    for k in value.split(";"):
                        k = k.strip()
                        if k and k not in seen_kw:
                            seen_kw.add(k)
                            keyword_strings.append(k)
                for sname in survey_names:
                    if sname and sname not in seen_kw:
                        seen_kw.add(sname)
                        keyword_strings.append(sname)
                        logger.info("KEYWORD_FROM_SURVEY_NAME | name=%s", sname[:100])

                # 候補プール: JDCAT + BM25 の検索結果を統合（idベースで重複除去）
                # 各要素: {"source", "id", "title", "rec", "origin"("JDCAT"/"BM25"), 任意で "bm25_score"}
                candidate_pool = []
                pool_ids = set()

                # ───── JDCAT: キーワード検索（常に実行） ─────
                if "JDCAT" in selected_repos and keyword_strings:
                    for kw in keyword_strings:
                        hits = search_jdcat_all_pages(kw, size=JDCAT_SEARCH_SIZE)
                        records = hits_to_ddi_records(hits, source_survey=kw) if hits else []
                        jdcat_keyword_search_log.append({
                            "keyword": kw,
                            "hits": [{"id": r.get("id", ""), "title": r.get("title", "")} for r in records],
                        })
                        for rec in records:
                            rid = rec.get("id", "")
                            if not rid or rid in pool_ids or not rec.get("title"):
                                continue
                            pool_ids.add(rid)
                            candidate_pool.append({
                                "source": "JDCAT",
                                "origin": "JDCAT",
                                "id": rid,
                                "title": rec.get("title", ""),
                                "rec": rec,
                            })
                        logger.info("JDCAT_KEYWORD_HITS | kw=%s | hits=%d", kw[:60], len(records))

                # ───── JGSSDDS/RUDA/CSRDA: BM25 検索（常に実行） ─────
                preload_repos = [r for r in selected_repos if r in OAI_PRELOAD_REPOS]
                BM25_TOP_K_PER_QUERY = 3
                if preload_repos:
                    preload_query_strings = list(keyword_strings)
                    if not preload_query_strings:
                        preload_query_strings = [combined_text[:500]]

                    per_query_log = []
                    for q in preload_query_strings:
                        results = oai_preloaded_store.search(
                            q, repos=preload_repos, limit=BM25_TOP_K_PER_QUERY,
                        )
                        per_query_log.append({
                            "query": q,
                            "hits": [
                                {"id": rec.get("id", ""), "title": rec.get("title", ""),
                                 "score": float(score), "source": rec.get("source_repository", "")}
                                for rec, score in results
                            ],
                        })
                        for rec, score in results:
                            rid = rec.get("id", "")
                            if not rid or rid in pool_ids or not rec.get("title"):
                                continue
                            pool_ids.add(rid)
                            candidate_pool.append({
                                "source": rec.get("source_repository", "BM25"),
                                "origin": "BM25",
                                "id": rid,
                                "title": rec.get("title", ""),
                                "rec": rec,
                                "bm25_score": float(score),
                            })
                        logger.info(
                            "OAI_PMH_BM25_SEARCH | query=%s | hits=%d",
                            q[:80], len(results),
                        )
                    preload_bm25_search_log.append({
                        "queries": preload_query_strings,
                        "per_query_hits": per_query_log,
                    })

                # ───── 統合 LLM 再ランキング（JDCAT + BM25 候補プールから上位 N 件） ─────
                EXTERNAL_TOP_N = 1
                rerank_result = {"selections": [], "prompt": "", "response": ""}
                if candidate_pool:
                    rerank_result = llm_processor.rerank_external_records(
                        survey_names=keyword_strings or [combined_text[:200]],
                        candidates=candidate_pool,
                        top_n=EXTERNAL_TOP_N,
                    )
                    selections = rerank_result.get("selections", [])

                    top_hits = []
                    seen_top_ids = set()
                    for rank, sel in enumerate(selections, 1):
                        idx = sel.get("idx")
                        if idx is None or idx < 0 or idx >= len(candidate_pool):
                            continue
                        cand = candidate_pool[idx]
                        rec = cand["rec"]
                        rid = cand["id"]
                        if rid and rid in seen_top_ids:
                            continue
                        if rid:
                            seen_top_ids.add(rid)
                        source = cand["source"]
                        rec_title = cand["title"]
                        reason = sel.get("reason", "")
                        top_hits.append({
                            "id": rid,
                            "title": rec_title,
                            "source": source,
                            "origin": cand["origin"],
                            "rank": rank,
                            "reason": reason,
                        })
                        example_parts.append(
                            f"【例: {source} (タイトル: {rec_title[:60]})】\n{_rec_to_json(rec, source)}"
                        )
                        external_records.append({
                            "source": source,
                            "label": f"タイトル: {rec_title[:80]}",
                            "fields": _normalize_external_fields(rec.get("fields") or {}),
                        })
                        if cand["origin"] == "JDCAT":
                            jdcat_has_hits = True
                        logger.info(
                            "EXTERNAL_LLM_RERANK | rank=%d | source=%s | id=%s | title=%s | reason=%s",
                            rank, source, rid, rec_title[:60], reason[:80],
                        )
                    jdcat_vector_search_log.append({
                        "query": ", ".join(keyword_strings),
                        "candidate_pool_size": len(candidate_pool),
                        "top_hits": top_hits,
                        "llm_prompt": rerank_result.get("prompt", ""),
                        "llm_response": rerank_result.get("response", ""),
                        "usage": rerank_result.get("usage"),
                    })

                if example_parts:
                    oai_background_text = (
                        "以下は外部リポジトリから取得したメタデータ例です。"
                        "抽出の参考にしてください。\n\n"
                        + "\n\n---\n\n".join(example_parts)
                    )
            except Exception as e:
                logger.error("OAI_PMH_SEARCH_ERROR | error=%s", e)

        # =============================
        # 外部レコードから URI・関連情報・トピック 用の値を準備
        #   - 関連情報    : LLMを使わず、外部データのタイトル+URIから直接生成
        #   - URI/トピック: Phase 2 のLLM抽出結果と外部データの値をマージする（後段で実施）
        # =============================
        EXTERNAL_DERIVED_FIELDS = {"関連情報"}
        external_uri_parts: list[str] = []
        external_rel_parts: list[str] = []
        external_topic_parts: list[str] = []
        for r in external_records:
            f = r.get("fields") or {}
            ext_title = (f.get("タイトル") or "").strip()
            uri_raw = (f.get("URI") or "").strip()
            topic_raw = (f.get("トピック") or "").strip()
            if uri_raw:
                for u in (s.strip() for s in uri_raw.split(";")):
                    if not u:
                        continue
                    if u not in external_uri_parts:
                        external_uri_parts.append(u)
                    if ext_title:
                        entry = f"{ext_title} ({u})"
                        if entry not in external_rel_parts:
                            external_rel_parts.append(entry)
            if topic_raw:
                for t in (s.strip() for s in topic_raw.split(";")):
                    if t and t not in external_topic_parts:
                        external_topic_parts.append(t)
        metadata_obj["関連情報"] = "; ".join(external_rel_parts) if external_rel_parts else "該当なし"
        all_field_prompts["関連情報"] = "(LLM未使用 / 外部リポジトリ取得データから生成)"
        all_field_responses["関連情報"] = metadata_obj["関連情報"]
        all_field_usage["関連情報"] = None
        logger.info(
            "EXTERNAL_DERIVED | 関連情報=%s | external_uri_count=%d | external_topic_count=%d",
            metadata_obj["関連情報"][:120], len(external_uri_parts), len(external_topic_parts),
        )

        # =============================
        # Phase 2: タイトル以外のフィールドを並列抽出（外部リポジトリ例を背景に使う）
        # =============================
        llm_processor.oai_background_text = oai_background_text
        llm_processor.external_records = external_records
        # Phase 1 で生成したタイトル/その他のタイトルを「調査名」としてプロンプトに渡し、
        # 複数の参考例から対応する調査を識別する手がかりにする
        phase1_title = str(metadata_obj.get("タイトル", "")).strip()
        phase1_other_title = str(metadata_obj.get("その他のタイトル", "")).strip()
        survey_name_values = []
        if phase1_title and phase1_title != "該当なし":
            survey_name_values.append(phase1_title)
        if phase1_other_title and phase1_other_title != "該当なし":
            for v in phase1_other_title.split(";"):
                v = v.strip()
                if v and v not in survey_name_values:
                    survey_name_values.append(v)
        llm_processor.current_survey_name = "\n".join(survey_name_values)
        remaining_field_texts = {
            k: v for k, v in field_chunk_texts.items()
            if k not in PHASE1_FIELDS and k not in EXTERNAL_DERIVED_FIELDS
        }
        with ThreadPoolExecutor(max_workers=EXTRACT_MAX_WORKERS) as executor:
            futures = {
                executor.submit(llm_processor.extract_field, fname, field_desc_map[fname] if fname in FIELDS_WITH_DESC else "", ftxt): fname
                for fname, ftxt in remaining_field_texts.items()
            }
            for future in as_completed(futures):
                field_name, value, prompt_text, raw_response, usage = future.result()
                metadata_obj[field_name] = value
                all_field_prompts[field_name] = prompt_text
                all_field_responses[field_name] = raw_response
                all_field_usage[field_name] = usage

        # URI / トピック: LLM抽出結果に外部レコードの値をマージ
        def _merge_with_external(field_name: str, external_parts: list[str]) -> None:
            llm_value = str(metadata_obj.get(field_name, "")).strip()
            merged: list[str] = []
            if llm_value and llm_value != "該当なし":
                for v in (s.strip() for s in llm_value.split(";")):
                    if v and v not in merged:
                        merged.append(v)
            for v in external_parts:
                if v not in merged:
                    merged.append(v)
            metadata_obj[field_name] = "; ".join(merged) if merged else "該当なし"
            logger.info(
                "FIELD_MERGE | field=%s | llm=%s | external=%d | merged=%s",
                field_name, llm_value[:80], len(external_parts),
                metadata_obj[field_name][:120],
            )

        _merge_with_external("URI", external_uri_parts)
        _merge_with_external("トピック", external_topic_parts)

        # 「その他のタイトル」が「タイトル」と重複している値を除去
        title_value = str(metadata_obj.get("タイトル", "")).strip()
        other_title_value = str(metadata_obj.get("その他のタイトル", "")).strip()
        if title_value and other_title_value:
            other_titles = [t.strip() for t in other_title_value.split(";") if t.strip()]
            filtered = [t for t in other_titles if t != title_value]
            if not filtered:
                metadata_obj["その他のタイトル"] = "該当なし"
                logger.info("OTHER_TITLE_DEDUP | all duplicates removed (title=%s)", title_value[:60])
            elif len(filtered) != len(other_titles):
                metadata_obj["その他のタイトル"] = "; ".join(filtered)
                logger.info("OTHER_TITLE_DEDUP | removed duplicates of title=%s", title_value[:60])

        # ── ファイル①: 設定パラメータログ ──
        total_cost_usd, cost_breakdown = calculate_cost()
        _total_cost_by_purpose, purpose_breakdown = calculate_cost_by_purpose()
        # 用途は KNOWN_PURPOSES の順で並べ、未知の purpose があれば末尾に追加
        ordered_purposes = [p for p in KNOWN_PURPOSES if p in purpose_breakdown]
        for p in purpose_breakdown.keys():
            if p not in ordered_purposes:
                ordered_purposes.append(p)
        log1_path = os.path.join(log_output_dir, "log1_settings.txt")
        with open(log1_path, "w", encoding="utf-8") as lf:
            lf.write(f"LLMモデル: {selected_model} ({model_config['model_id']})\n")
            lf.write(f"テキスト抽出方法: pymupdf\n")
            lf.write(f"チャンク分割方法: LLM意味単位分割（{NUM_SPLITS}分割）\n")
            lf.write(f"検索方法: ベクトル検索なし（該当項目マッチング）\n")
            lf.write(f"入力ファイル名: {filenames_for_log}\n")
            lf.write(f"チャンク分割数: {len(all_chunks)}\n")
            lf.write("\n" + "=" * 40 + "\n")
            lf.write("【LLM トークン使用量・料金（モデル別）】\n")
            lf.write("=" * 40 + "\n")
            total_input = sum(b["input_tokens"] for b in cost_breakdown.values())
            total_output = sum(b["output_tokens"] for b in cost_breakdown.values())
            total_calls = sum(b["calls"] for b in cost_breakdown.values())
            for model_id, b in cost_breakdown.items():
                lf.write(f"\n[{model_id}]\n")
                lf.write(f"  呼び出し回数: {b['calls']}\n")
                lf.write(f"  入力トークン: {b['input_tokens']:,}\n")
                lf.write(f"  出力トークン: {b['output_tokens']:,}\n")
                lf.write(f"  入力料金: ${b['input_cost_usd']:.4f}\n")
                lf.write(f"  出力料金: ${b['output_cost_usd']:.4f}\n")
                lf.write(f"  小計: ${b['total_cost_usd']:.4f}\n")
            lf.write("\n" + "=" * 40 + "\n")
            lf.write("【LLM トークン使用量・料金（用途別）】\n")
            lf.write("=" * 40 + "\n")
            for p in ordered_purposes:
                b = purpose_breakdown[p]
                label = PURPOSE_LABELS.get(p, p)
                lf.write(f"\n[{label} (purpose={p})]\n")
                lf.write(f"  呼び出し回数: {b['calls']}\n")
                lf.write(f"  入力トークン: {b['input_tokens']:,}\n")
                lf.write(f"  出力トークン: {b['output_tokens']:,}\n")
                lf.write(f"  入力料金: ${b['input_cost_usd']:.4f}\n")
                lf.write(f"  出力料金: ${b['output_cost_usd']:.4f}\n")
                lf.write(f"  小計: ${b['total_cost_usd']:.4f}\n")
                if len(b.get("by_model") or {}) > 1:
                    for mid, mb in b["by_model"].items():
                        lf.write(
                            f"    └ {mid}: calls={mb['calls']}, "
                            f"in={mb['input_tokens']:,}, out={mb['output_tokens']:,}, "
                            f"$={mb['total_cost_usd']:.4f}\n"
                        )
            lf.write("\n" + "-" * 40 + "\n")
            lf.write(f"合計呼び出し回数: {total_calls}\n")
            lf.write(f"合計入力トークン: {total_input:,}\n")
            lf.write(f"合計出力トークン: {total_output:,}\n")
            lf.write(f"合計料金: ${total_cost_usd:.4f} USD\n")
        logger.info(
            "LLM_COST | total_calls=%d | input_tokens=%d | output_tokens=%d | total_cost_usd=%.4f",
            total_calls, total_input, total_output, total_cost_usd,
        )
        for p, b in purpose_breakdown.items():
            logger.info(
                "LLM_COST_BY_PURPOSE | purpose=%s | calls=%d | input=%d | output=%d | cost_usd=%.4f",
                p, b["calls"], b["input_tokens"], b["output_tokens"], b["total_cost_usd"],
            )

        # ── ファイル③: 該当項目マッチング結果 ──
        log3_path = os.path.join(log_output_dir, "log3_field_matching.txt")
        with open(log3_path, "w", encoding="utf-8") as lf:
            for field in METADATA_FIELDS:
                field_name = field["name"]
                matched = field_chunks_map.get(field_name, [])
                lf.write(f"【項目: {field_name}】 マッチ数: {len(matched)}\n")
                for rank, (chunk_text, doc_no, src_file) in enumerate(matched, 1):
                    lf.write(f"  チャンク {rank} (通番{doc_no}, {src_file}): {chunk_text[:200]}\n")
                lf.write("=" * 80 + "\n\n")

        # ── ファイル④: LLMプロンプト ──
        log4_path = os.path.join(log_output_dir, "log4_llm_prompt.txt")
        with open(log4_path, "w", encoding="utf-8") as lf:
            for field in METADATA_FIELDS:
                field_name = field["name"]
                llm_prompt = all_field_prompts.get(field_name, "")
                usage = all_field_usage.get(field_name)
                lf.write(f"【項目: {field_name}】 {_format_usage(usage)}\n")
                lf.write("-" * 40 + "\n")
                lf.write("【プロンプト】\n")
                lf.write(llm_prompt)
                lf.write("\n" + "-" * 40 + "\n")
                lf.write("【LLM生レスポンス】\n")
                raw_resp = all_field_responses.get(field_name, "")
                lf.write(raw_resp if raw_resp else "(空)")
                lf.write("\n" + "-" * 40 + "\n")
                lf.write("【抽出結果】\n")
                answer = metadata_obj.get(field_name, "")
                lf.write(str(answer) if answer else "(空)")
                lf.write("\n" + "=" * 80 + "\n\n")

        # ── ファイル⑤: チャンク分割LLMプロンプト（Step1: テキスト分割 + Step2: 項目付与） ──
        log5_path = os.path.join(log_output_dir, "log5_split_prompt.txt")
        with open(log5_path, "w", encoding="utf-8") as lf:
            lf.write("=" * 80 + "\n")
            lf.write("Step 1: テキスト分割\n")
            lf.write("=" * 80 + "\n\n")
            for idx in range(NUM_SPLITS):
                log_entry = split_text_logs.get(idx, {"prompt": "", "response": ""})
                prompt_text = log_entry["prompt"]
                resp_text = log_entry["response"]
                # Step1 は機械的分割 (LLM未使用)
                lf.write(f"【テキスト分割 {idx + 1}/{NUM_SPLITS}】 (LLM未使用 / 句点分割)\n")
                lf.write("-" * 40 + "\n")
                lf.write("【プロンプト】\n")
                lf.write(prompt_text)
                lf.write("\n" + "-" * 40 + "\n")
                lf.write("【回答】\n")
                lf.write(resp_text if resp_text else "(空)")
                lf.write("\n" + "=" * 80 + "\n\n")
            lf.write("=" * 80 + "\n")
            lf.write("Step 2: 項目付与\n")
            lf.write("=" * 80 + "\n\n")
            for key in sorted(split_logs.keys()):
                idx, group_name, batch_start = key
                log_entry = split_logs[key]
                prompt_text = log_entry["prompt"]
                resp_text = log_entry["response"]
                usage = log_entry.get("usage")
                lf.write(
                    f"【項目付与 split={idx + 1}/{NUM_SPLITS} | グループ: {group_name} | "
                    f"バッチ開始: {batch_start}】 {_format_usage(usage)}\n"
                )
                lf.write("-" * 40 + "\n")
                lf.write("【プロンプト】\n")
                lf.write(prompt_text)
                lf.write("\n" + "-" * 40 + "\n")
                lf.write("【回答】\n")
                lf.write(resp_text if resp_text else "(空)")
                lf.write("\n" + "=" * 80 + "\n\n")

        # ── ファイル⑥: 統制語彙変換LLMプロンプト ──
        log6_path = os.path.join(log_output_dir, "log6_vocab_prompt.txt")
        with open(log6_path, "w", encoding="utf-8") as lf:
            for i, vlog in enumerate(vocab_logs, 1):
                prompt_text = vlog["prompt"]
                usage = vlog.get("usage")
                lf.write(
                    f"【変換 {i}】 項目: {vlog['field']} | 元テキスト: {vlog['original'][:100]} | "
                    f"{_format_usage(usage)}\n"
                )
                lf.write("-" * 40 + "\n")
                lf.write("【プロンプト】\n")
                lf.write(prompt_text)
                lf.write("\n" + "-" * 40 + "\n")
                lf.write("【回答】\n")
                lf.write(vlog["response"] if vlog["response"] else "(空)")
                lf.write("\n" + "=" * 80 + "\n\n")

        # ── ファイル⑦: JDCAT キーワード検索 & ベクトル検索 ──
        log7_path = os.path.join(log_output_dir, "log7_jdcat_search.txt")
        with open(log7_path, "w", encoding="utf-8") as lf:
            lf.write(f"入力ファイル名: {filenames_for_log}\n")
            lf.write("=" * 80 + "\n\n")
            lf.write("[キーワード検索 (JDCAT)]\n\n")
            if jdcat_keyword_search_log:
                for entry in jdcat_keyword_search_log:
                    lf.write(f"キーワード: {entry['keyword']}\n")
                    lf.write(f"ヒット数: {len(entry['hits'])}\n")
                    for i, h in enumerate(entry['hits'], 1):
                        lf.write(f"  {i}. id={h['id']} | title={h['title']}\n")
                    lf.write("\n")
            else:
                lf.write("(キーワード検索は実行されませんでした)\n\n")
            lf.write("=" * 80 + "\n\n")
            lf.write("[BM25検索 (JGSSDDS/RUDA/CSRDA)]\n\n")
            if preload_bm25_search_log:
                for entry in preload_bm25_search_log:
                    queries = entry.get("queries") or []
                    lf.write(f"使用クエリ数: {len(queries)}\n")
                    for q in queries:
                        lf.write(f"  - {q}\n")
                    per_query_hits = entry.get("per_query_hits") or []
                    if per_query_hits:
                        lf.write("各クエリのヒット:\n")
                        for h in per_query_hits:
                            hits_list = h.get("hits") or []
                            lf.write(f"  - query={h['query'][:60]} | hits={len(hits_list)}\n")
                            for hit in hits_list:
                                lf.write(
                                    f"      [{hit.get('source','')}] score={hit.get('score',0):.4f} "
                                    f"| id={hit.get('id','')} | title={hit.get('title','')[:80]}\n"
                                )
                    lf.write("\n")
            else:
                lf.write("(BM25検索は実行されませんでした)\n\n")
            lf.write("=" * 80 + "\n\n")
            lf.write("[統合LLM再ランキング (JDCAT + JGSSDDS/RUDA/CSRDA)]\n\n")
            if jdcat_vector_search_log:
                for entry in jdcat_vector_search_log:
                    lf.write(f"判定に使用した調査名候補: {entry['query']}\n")
                    pool_size = entry.get("candidate_pool_size")
                    if pool_size is not None:
                        lf.write(f"候補プールのサイズ: {pool_size}\n")
                    hits = entry.get("top_hits") or []
                    if hits:
                        lf.write(f"LLMが選出した上位{len(hits)}件:\n")
                        for h in hits:
                            rank = h.get("rank", "-")
                            lf.write(f"  {rank}. [{h.get('origin','')}/{h.get('source','')}] id={h['id']}\n")
                            lf.write(f"     title={h['title']}\n")
                            reason = h.get("reason", "")
                            if reason:
                                lf.write(f"     reason={reason}\n")
                    else:
                        lf.write("上位件: なし\n")
                    llm_prompt = entry.get("llm_prompt", "")
                    llm_response = entry.get("llm_response", "")
                    if llm_prompt:
                        usage = entry.get("usage")
                        lf.write(f"LLM使用量: {_format_usage(usage)}\n")
                        lf.write("-" * 40 + "\n")
                        lf.write("【LLMプロンプト】\n")
                        lf.write(llm_prompt)
                        lf.write("\n" + "-" * 40 + "\n")
                        lf.write("【LLM回答】\n")
                        lf.write(llm_response if llm_response else "(空)")
                        lf.write("\n")
                    lf.write("\n")
            else:
                lf.write("(候補プールが空のため、再ランキングは実行されませんでした)\n\n")

        # ── ファイル⑧: 調査名抽出（機械的 or LLM） ──
        log8_path = os.path.join(log_output_dir, "log8_survey_name.txt")
        with open(log8_path, "w", encoding="utf-8") as lf:
            lf.write(f"入力ファイル名: {filenames_for_log}\n")
            lf.write("=" * 80 + "\n\n")
            if survey_name_info:
                method_label = {
                    "mechanical": "機械的抽出（「調査名」ラベルを検出）",
                    "llm": "LLM抽出（機械的抽出不可のためフォールバック）",
                    "fallback": "抽出失敗（入力先頭200文字をフォールバック）",
                }.get(survey_name_info.get("method", ""), survey_name_info.get("method", ""))
                names_list = survey_name_info.get("names", [])
                lf.write(f"抽出方法: {method_label}\n")
                lf.write(f"抽出件数: {len(names_list)}\n")
                if names_list:
                    lf.write("抽出結果:\n")
                    for n in names_list:
                        lf.write(f"  - {n}\n")
                else:
                    lf.write("抽出結果: (なし)\n")
                lf.write("-" * 40 + "\n")
                lf.write("【入力テキスト（各ファイルの1チャンク目を結合）】\n")
                lf.write(survey_name_info.get("input", ""))
                lf.write("\n" + "-" * 40 + "\n")
                if survey_name_info.get("prompt"):
                    lf.write(f"LLM使用量: {_format_usage(survey_name_info.get('usage'))}\n")
                    lf.write("-" * 40 + "\n")
                    lf.write("【LLMプロンプト】\n")
                    lf.write(survey_name_info["prompt"])
                    lf.write("\n" + "-" * 40 + "\n")
                    lf.write("【LLM回答】\n")
                    lf.write(survey_name_info.get("response") or "(空)")
                    lf.write("\n")
            else:
                lf.write("(調査名抽出は実行されませんでした。use_oai_searchがOFFか、JDCATが選択されていません)\n")

        # LLM応答を結果リストに変換
        results = []
        for field in METADATA_FIELDS:
            value = metadata_obj.get(field["name"], "")
            if value is None:
                value = ""
            results.append({
                "no": field["no"],
                "name": field["name"],
                "ddi": field["ddi"],
                "value": str(value),
            })

        elapsed = time.time() - start_time
        print("METADATA_EXTRACTION_COMPLETE | filename=%s | elapsed_seconds=%.2f" % (filenames_for_log, elapsed))
        print("=" * 60)
        print("LLM トークン使用量・料金（モデル別）")
        print("=" * 60)
        for model_id, b in cost_breakdown.items():
            print(f"[{model_id}]")
            print(f"  呼び出し回数: {b['calls']}")
            print(f"  入力トークン: {b['input_tokens']:,}  (${b['input_cost_usd']:.4f})")
            print(f"  出力トークン: {b['output_tokens']:,}  (${b['output_cost_usd']:.4f})")
            print(f"  小計: ${b['total_cost_usd']:.4f}")
        print("=" * 60)
        print("LLM トークン使用量・料金（用途別）")
        print("=" * 60)
        for p in ordered_purposes:
            b = purpose_breakdown[p]
            label = PURPOSE_LABELS.get(p, p)
            print(f"[{label}]")
            print(f"  呼び出し回数: {b['calls']}")
            print(f"  入力トークン: {b['input_tokens']:,}  (${b['input_cost_usd']:.4f})")
            print(f"  出力トークン: {b['output_tokens']:,}  (${b['output_cost_usd']:.4f})")
            print(f"  小計: ${b['total_cost_usd']:.4f}")
        print("-" * 60)
        print(f"合計: 呼び出し={total_calls}回 / 入力={total_input:,}tok / 出力={total_output:,}tok / 料金=${total_cost_usd:.4f} USD")
        print("=" * 60)
        status_placeholder.markdown('<div class="status-done">処理完了</div>', unsafe_allow_html=True)

        filenames = ", ".join(f.name for f in uploaded_files)
        st.session_state.all_results[filenames] = results

        logger.info("LOG_FILES_WRITTEN | log1=%s | log2=%s | log3=%s | log4=%s | log5=%s | log6=%s | log7=%s | log8=%s", log1_path, log2_path, log3_path, log4_path, log5_path, log6_path, log7_path, log8_path)

    # ══════════════════════════════════════════════
    # 結果表示
    # ══════════════════════════════════════════════
    for filename, results in st.session_state.get("all_results", {}).items():
        st.markdown(f"---\n### {filename}")

        csv_data = OutputWriter.create_csv(results)
        st.download_button(
            "結果をCSVファイルでダウンロード",
            data=csv_data.encode("utf-8-sig"),
            file_name=f"metadata_{filename.replace('.pdf', '')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        for r in results:
            st.markdown(f'<div class="field-label">{html.escape(r["name"])}：</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="field-value">{html.escape(r["value"])}</div>', unsafe_allow_html=True)
