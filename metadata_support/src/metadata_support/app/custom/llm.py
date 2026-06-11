import logging
import re
import time

from metadata_support.app.custom.json_utils import JsonUtils
from metadata_support.extract.bedrock_extractor import (
    get_last_call_usage,
    purpose_scope,
)

logger = logging.getLogger(__name__)


def _empty_usage(purpose: str = "unknown") -> dict:
    """LLM呼び出しが行われなかった/失敗した場合の空 usage を返す。"""
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "input_cost_usd": 0.0,
        "output_cost_usd": 0.0,
        "total_cost_usd": 0.0,
        "model_id": "",
        "purpose": purpose,
    }


class LLMProcessor:
    """チャンク分割・項目付与・統制語彙変換・項目抽出を担うLLM処理。

    実行時に値が決まる要素（extractor, num_splits, start_time, oai_background_text）は
    呼び出し前にインスタンス属性として設定する。
    """

    def __init__(self, chunk_model, vocab_model, verbatim_fields):
        self.chunk_model = chunk_model
        self.vocab_model = vocab_model
        self.verbatim_fields = verbatim_fields
        # 実行時に呼び出し側で設定
        self.extractor = None
        self.num_splits = 0
        self.start_time = 0.0
        self.oai_background_text = None
        # 外部リポジトリから取得したレコード一覧（項目別にフィルタして参考例として使用）
        # 各要素: {"source": "JDCAT"|..., "label": "任意の補足", "fields": {"項目名": "値", ...}}
        self.external_records = []
        # 先行フェーズで生成した「調査名」（タイトル + その他のタイトル）。
        # 参考例との照合ガイドとしてプロンプトに含める。
        self.current_survey_name = ""

    # 1チャンク目から調査名を抽出（複数候補に対応、戻り値の "names" は list[str]）
    def extract_survey_name(self, chunk_text, fname=""):
        # 機械的抽出: テキスト中の「調査名: ...」表現を全件取得し、重複除去で順序保持
        mech_matches = re.findall(
            r"調査名[\s:：「『【（(《〈\"'　]+([^\n\r]+)",
            chunk_text,
        )
        mech_names = []
        for raw in mech_matches:
            n = raw.strip().rstrip("　 、。,，").strip("「」『』\"' 　")
            if n and n not in mech_names:
                mech_names.append(n)
        if mech_names:
            logger.info("SURVEY_NAME_MECHANICAL | file=%s | count=%d | names=%s",
                        fname, len(mech_names), "; ".join(mech_names)[:200])
            return {
                "names": mech_names,
                "method": "mechanical",
                "input": chunk_text,
                "prompt": "",
                "response": "",
                "usage": _empty_usage("survey_name"),
            }

        logger.info("SURVEY_NAME_EXTRACT_START | file=%s | text_len=%d", fname, len(chunk_text))
        prompt = f"""以下のテキストから調査名を取得してください。

## ルール
- 取得する調査名をJDCATでキーワード検索した場合に、上位に類似の調査が出てくるような調査名を取得してください。
- 〇〇調査、〇〇プロジェクトなどといった表現があれば、調査名の可能性が高いです。
- 候補が複数ある場合は、すべて1行に1件ずつ出力してください。
- 調査名のみを出力してください。前置き・説明・ラベル・記号・括弧書きは一切不要です。
- 調査名が見つからない場合は、「該当なし」とだけ出力してください。

## テキスト
{chunk_text}
"""
        max_retries = 3
        last_resp_text = ""
        last_usage = _empty_usage("survey_name")
        for attempt in range(max_retries):
            try:
                resp = self.chunk_model.prompt(prompt, purpose="survey_name")
                resp_text = resp.text().strip()
                last_resp_text = resp_text
                last_usage = resp.usage
                # マークダウン見出し・空行・前置きラベルをスキップして調査名行を抽出
                names: list[str] = []
                for line in resp_text.splitlines():
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    if s.startswith("-") or s.startswith("*"):
                        s = s.lstrip("-* 　").strip()
                    s = s.strip("「」『』\"' 　")
                    if not s or s == "該当なし":
                        continue
                    if s not in names:
                        names.append(s)
                if not names:
                    raise ValueError("empty survey name")
                logger.info("SURVEY_NAME_EXTRACT_DONE | file=%s | count=%d | names=%s",
                            fname, len(names), "; ".join(names)[:200])
                return {
                    "names": names,
                    "method": "llm",
                    "input": chunk_text,
                    "prompt": prompt,
                    "response": resp_text,
                    "usage": last_usage,
                }
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = 5 * (2 ** attempt)
                    logger.warning("SURVEY_NAME_EXTRACT_RETRY | file=%s | attempt=%d/%d | delay=%ds | error=%s",
                                   fname, attempt + 1, max_retries, delay, e)
                    time.sleep(delay)
                else:
                    logger.error("SURVEY_NAME_EXTRACT_ERROR | file=%s | error=%s", fname, e)
                    return {
                        "names": [chunk_text.strip()[:200]],
                        "method": "fallback",
                        "input": chunk_text,
                        "prompt": prompt,
                        "response": last_resp_text,
                        "usage": last_usage,
                    }

    # JDCAT + JGSSDDS/RUDA/CSRDA の候補を統合してLLMで再ランキング
    def rerank_external_records(self, survey_names, candidates, top_n=5):
        """JDCAT(キーワード検索)+JGSSDDS/RUDA/CSRDA(BM25検索)の候補プールから、
        調査名に最も対応するものを上位N件選出する。

        candidates: list[dict] - {"source", "id", "title", ...} のリスト
        戻り値: {"selections": [{"idx": int, "reason": str}], "prompt", "response"}
        """
        if not candidates:
            return {"selections": [], "prompt": "", "response": "", "usage": _empty_usage("rerank_external")}
        names_text = "\n".join(f"- {n}" for n in survey_names if n)
        candidates_text = "\n".join(
            f"[{i}] (source={c.get('source','?')}) {c.get('title', '')}"
            for i, c in enumerate(candidates)
        )
        prompt = f"""あなたは社会科学分野の調査データカタログを扱う専門家です。
以下の「調査名候補」で示された調査に、最も対応する可能性が高い候補レコードを選出してください。
候補は複数の外部リポジトリ（JDCAT / JGSSDDS / RUDA / CSRDA）から取得されています。

## 調査名候補
{names_text}

## 候補レコード一覧
{candidates_text}

## 指示
- 「調査名候補」が指す調査と、最も対応する可能性が高いレコードを {top_n} 件まで選んでください。
- 同じシリーズ・同じプロジェクトのレコードも候補に含めてよいですが、全く関係ない調査は含めないでください。
- 該当レコードが無い場合は空の配列を返してください。
- 候補が {top_n} 件未満なら、該当するもののみ返してください（無理に {top_n} 件にする必要はありません）。
- 対応度が高い順にソートしてください。

## 出力形式（JSONのみ。前置き・説明文は不要）
{{"selections": [{{"idx": 候補番号, "reason": "選んだ理由の短い説明"}}, ...]}}
"""
        logger.info("RERANK_EXTERNAL_START | candidates=%d | top_n=%d", len(candidates), top_n)
        try:
            resp = self.chunk_model.prompt(prompt, purpose="rerank_external")
            resp_text = resp.text().strip()
            usage = resp.usage
        except Exception as e:
            logger.error("RERANK_EXTERNAL_ERROR | error=%s", e)
            return {"selections": [], "prompt": prompt, "response": "", "usage": _empty_usage("rerank_external")}

        try:
            data, _ = JsonUtils.repair_and_parse(resp_text, "rerank_external")
        except Exception as e:
            logger.error("RERANK_EXTERNAL_PARSE_ERROR | error=%s | resp=%s", e, resp_text[:200])
            return {"selections": [], "prompt": prompt, "response": resp_text, "usage": usage}

        raw_sels = data.get("selections") if isinstance(data, dict) else data
        if not isinstance(raw_sels, list):
            return {"selections": [], "prompt": prompt, "response": resp_text, "usage": usage}

        selections = []
        seen_idx = set()
        for item in raw_sels:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("idx"))
            except (TypeError, ValueError):
                continue
            if idx < 0 or idx >= len(candidates) or idx in seen_idx:
                continue
            seen_idx.add(idx)
            selections.append({
                "idx": idx,
                "reason": str(item.get("reason", "")).strip(),
            })
            if len(selections) >= top_n:
                break
        logger.info("RERANK_EXTERNAL_DONE | selected=%d", len(selections))
        return {"selections": selections, "prompt": prompt, "response": resp_text, "usage": usage}

    # JDCATキーワード検索結果をLLMで再ランキング（Top-N選出）
    def rerank_jdcat_records(self, survey_names, candidates, top_n=5):
        """JDCAT候補レコードから、調査名に最も対応するものを上位N件選出する。

        survey_names: list[str] - 調査の名称候補（タイトル・その他のタイトル・調査名など）
        candidates: list[dict] - {"id", "title", ...} のリスト
        top_n: int - 選出する件数

        戻り値: list[dict] = [{"idx": int, "id": str, "title": str, "reason": str|None}]
        （LLMが選択した順）
        """
        if not candidates:
            return {"selections": [], "prompt": "", "response": "", "usage": _empty_usage("rerank_external")}
        names_text = "\n".join(f"- {n}" for n in survey_names if n)
        candidates_text = "\n".join(
            f"[{i}] {c.get('title', '')}" for i, c in enumerate(candidates)
        )
        prompt = f"""あなたは社会科学分野の調査データカタログを扱う専門家です。
以下の「調査名候補」で示された調査に、最も対応する可能性が高い JDCAT レコードを選出してください。

## 調査名候補
{names_text}

## JDCAT 候補レコード一覧
{candidates_text}

## 指示
- 「調査名候補」が指す調査と、最も対応する可能性が高いレコードを {top_n} 件まで選んでください。
- 同じシリーズ・同じプロジェクトのレコードも候補に含めてよいですが、全く関係ない調査は含めないでください。
- 該当レコードが無い場合は空の配列を返してください。
- 候補が {top_n} 件未満なら、該当するもののみ返してください（無理に {top_n} 件にする必要はありません）。
- 対応度が高い順にソートしてください。

## 出力形式（JSONのみ。前置き・説明文は不要）
{{"selections": [{{"idx": 候補番号, "reason": "選んだ理由の短い説明"}}, ...]}}
"""
        logger.info("RERANK_START | candidates=%d | top_n=%d", len(candidates), top_n)
        try:
            resp = self.chunk_model.prompt(prompt, purpose="rerank_external")
            resp_text = resp.text().strip()
            usage = resp.usage
        except Exception as e:
            logger.error("RERANK_ERROR | error=%s", e)
            return {"selections": [], "prompt": prompt, "response": "", "usage": _empty_usage("rerank_external")}

        # JSONパース（JsonUtils経由で修復も試みる）
        try:
            data, _ = JsonUtils.repair_and_parse(resp_text, "rerank_jdcat")
        except Exception as e:
            logger.error("RERANK_PARSE_ERROR | error=%s | resp=%s", e, resp_text[:200])
            return {"selections": [], "prompt": prompt, "response": resp_text, "usage": usage}

        # 期待構造: {"selections": [{"idx": N, "reason": "..."}]}
        raw_sels = data.get("selections") if isinstance(data, dict) else data
        if not isinstance(raw_sels, list):
            return {"selections": [], "prompt": prompt, "response": resp_text, "usage": usage}

        selections = []
        seen_idx = set()
        for item in raw_sels:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("idx"))
            except (TypeError, ValueError):
                continue
            if idx < 0 or idx >= len(candidates) or idx in seen_idx:
                continue
            seen_idx.add(idx)
            selections.append({
                "idx": idx,
                "id": candidates[idx].get("id", ""),
                "title": candidates[idx].get("title", ""),
                "reason": str(item.get("reason", "")).strip(),
            })
            if len(selections) >= top_n:
                break
        logger.info("RERANK_DONE | selected=%d", len(selections))
        return {"selections": selections, "prompt": prompt, "response": resp_text, "usage": usage}

    # テキストをチャンク分割
    def split_text(self, fname, ftext):
        """ファイルテキストを句点（。）で機械的にチャンク分割する。
        句点がない場合は改行で分割にフォールバック。
        """
        logger.info("SPLIT_TEXT_START | file=%s | text_len=%d", fname, len(ftext))
        ftext = re.sub(r"\n{2,}", "\n", ftext)
        raw_sentences = re.split(r'(?<=。)', ftext)
        chunks = [s.strip() for s in raw_sentences if s.strip()]
        if len(chunks) <= 1:
            chunks = [s.strip() for s in ftext.split('\n') if s.strip()]
            logger.info("SPLIT_TEXT_FALLBACK_NEWLINE | file=%s | chunks=%d", fname, len(chunks))
        logger.info("SPLIT_TEXT_DONE | file=%s | chunks=%d", fname, len(chunks))
        return {"fname": fname, "chunks": chunks}

    # 各チャンクに該当する項目を付与
    def annotate_chunks(self, split_idx, chunks, group_name, group_fields_str, position_info="",
                        prev_split_chunks=None, next_split_chunks=None):
        logger.info("ANNOTATE_START | split=%d | group=%s | chunks=%d", split_idx, group_name, len(chunks))
        # 前splitの末尾チャンク + 現split + 次splitの先頭チャンクを連結した文脈リストを構築
        prev_ctx_chunks = (prev_split_chunks or [])[-2:]  # 前splitの末尾2チャンク
        next_ctx_chunks = (next_split_chunks or [])[:2]    # 次splitの先頭2チャンク
        # 全文脈チャンク: prev_ctx + chunks + next_ctx
        full_chunks = prev_ctx_chunks + list(chunks) + next_ctx_chunks
        offset = len(prev_ctx_chunks)  # chunksの先頭が full_chunks 内で何番目か

        # 各チャンクに前後2チャンクの全文を文脈として付与
        chunks_text_parts = []
        for i, c in enumerate(chunks):
            fi = offset + i  # full_chunks 内のインデックス
            part = f"【チャンク{i+1}】（判定対象）\n{c}"
            # 前方文脈（full_chunks から最大2チャンク）
            prev_ctx = []
            for p in range(max(0, fi - 2), fi):
                prev_ctx.append(f"[前方文脈]\n{full_chunks[p]}")
            # 後方文脈（full_chunks から最大2チャンク）
            next_ctx = []
            for n in range(fi + 1, min(len(full_chunks), fi + 3)):
                next_ctx.append(f"[後方文脈]\n{full_chunks[n]}")
            if prev_ctx or next_ctx:
                part += "\n＜前後の文脈＞"
                if prev_ctx:
                    part += "\n" + "\n".join(prev_ctx)
                if next_ctx:
                    part += "\n" + "\n".join(next_ctx)
            chunks_text_parts.append(part)
        chunks_text = "\n\n".join(chunks_text_parts)
        prompt_text = f"""あなたは社会科学分野の論文や調査票を分析する専門家です。
以下のチャンクは、論文または調査報告書をセマンティックに分割したテキストです。
各チャンクに対して、該当するメタデータ項目名と確信度を付与してください。
各チャンクには前後の文脈（同一ドキュメント内の前後最大2チャンクの全文）が付与されています。
判定対象チャンクの内容に加え、前後の文脈も考慮して該当項目を判定してください。

## メタデータ項目一覧
{group_fields_str}

## ルール
1. 1つのチャンクが複数の項目に該当する場合は、該当する項目の数だけエントリを作成してください。
2. 確信度が3でも4でもない場合は field を「該当項目なし」としてください。
3. field には項目名のみを記載してください。括弧内の説明は含めないでください。
   （例: ○「タイトル」 ✕「タイトル（調査名・タイトル）」）
4. 出力には chunk_id と field と confidence のみを含めてください。テキストは出力しないでください（後段で機械的に付与します）。
5. 各チャンクに対して、該当項目への確信度を以下の基準で判定し、対応する数値を記載してください。
   - 4（確実）: 前後の文脈を見なくてもチャンク内のテキストだけを見れば、該当項目が判断できる
   - 3（高い）: 前後の文脈を見れば、該当項目が判断できる

## 位置情報
{position_info}

## 出力形式（JSON配列のみで回答してください。前置きや説明文は不要です）
[
  {{"chunk_id": 1, "field": "該当項目名", "confidence": 4}},
  {{"chunk_id": 1, "field": "別の該当項目名", "confidence": 3}},
  ...
]

## チャンク一覧
{chunks_text}"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = self.chunk_model.prompt(prompt_text, purpose="annotate")
                resp_text = resp.text().strip()
                usage = resp.usage
                chunks_data, resp_text = JsonUtils.repair_and_parse(
                    resp_text, f"annotate|split={split_idx}|group={group_name}"
                )
                results = []
                for item in chunks_data:
                    field = item.get("field", "該当項目なし")
                    field = re.sub(r"[（(].+?[）)]$", "", field).strip()
                    chunk_id = item.get("chunk_id", 0)
                    # chunk_idに対応する元テキストを機械的に付与（LLMはtextを出力しない）
                    if not isinstance(chunk_id, (int, float)) or not (1 <= int(chunk_id) <= len(chunks)):
                        logger.warning("ANNOTATE_INVALID_CHUNK_ID | split=%d | group=%s | chunk_id=%s",
                                       split_idx, group_name, chunk_id)
                        continue
                    text = chunks[int(chunk_id) - 1]
                    confidence = item.get("confidence", 0.0)
                    try:
                        confidence = float(confidence)
                    except (TypeError, ValueError):
                        confidence = 0.0
                    if not text:
                        continue
                    results.append({
                        "id": f"split{split_idx}_{group_name}_{len(results)}",
                        "name": f"split{split_idx}_{group_name}_chunk{len(results)}",
                        "text": f"該当項目: {field}\n\n{text}",
                        "field": field,
                        "chunk_text": text,
                        "confidence": confidence,
                        "local_chunk_id": int(chunk_id) if isinstance(chunk_id, (int, float)) else 0,
                    })
                logger.info("ANNOTATE_DONE | split=%d/%d | group=%s | chunks=%d",
                            split_idx + 1, self.num_splits, group_name, len(results))
                logger.warning("ELAPSED | ANNOTATE | split=%d/%d | group=%s | elapsed=%.1fs",
                               split_idx + 1, self.num_splits, group_name, time.time() - self.start_time)
                return {"results": results, "prompt": prompt_text, "response": resp_text, "usage": usage}
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = 5 * (2 ** attempt)
                    logger.warning("ANNOTATE_RETRY | split=%d | group=%s | attempt=%d/%d | delay=%ds | error=%s",
                                   split_idx, group_name, attempt + 1, max_retries, delay, e)
                    time.sleep(delay)
                else:
                    logger.error("ANNOTATE_ERROR | split=%d | group=%s | error=%s", split_idx, group_name, e)
                    return {"results": [], "prompt": prompt_text, "response": "", "usage": _empty_usage("annotate")}

    # チャンクテキストを統制語彙に変換
    def convert_to_vocab(self, field_name, text, vocab_list):
        logger.info("VOCAB_CONVERT_START | field=%s | text_len=%d | vocab_count=%d",
                    field_name, len(text), len(vocab_list))
        vocab_str = "\n".join(f"- {v}" for v in vocab_list)
        extra_rules = ""
        if field_name == "データタイプ":
            extra_rules = (
                "\n最も近いデータタイプを1つ選択してください。"
                "\n調査・データセット本体のデータタイプを選んでください。"
                "比較や参照のために言及されているデータは除いてください。"
            )
        prompt = f"""以下のテキストは「{field_name}」に関する記述です。
下記の統制語彙リストの中から、テキストの内容に最も近いものを選んでください。
複数該当する場合は「;」（セミコロン）区切りで列挙してください。
統制語彙のみを回答し、前置きや説明文は不要です。

## Rules
必ず統制語彙のリストの中にある語彙をなにも変えずに選んでください。{extra_rules}

## 統制語彙リスト
{vocab_str}

## テキスト
{text}"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = self.vocab_model.prompt(prompt, purpose="vocab_convert")
                resp_text = resp.text().strip()
                logger.info("VOCAB_CONVERT_DONE | field=%s | converted=%s", field_name, resp_text[:100])
                logger.warning("ELAPSED | VOCAB_CONVERT | field=%s | elapsed=%.1fs",
                               field_name, time.time() - self.start_time)
                return {"converted": resp_text, "prompt": prompt, "response": resp_text, "usage": resp.usage}
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = 5 * (2 ** attempt)
                    logger.warning("VOCAB_CONVERT_RETRY | field=%s | attempt=%d/%d | delay=%ds | error=%s",
                                   field_name, attempt + 1, max_retries, delay, e)
                    time.sleep(delay)
                else:
                    logger.error("VOCAB_CONVERT_ERROR | field=%s | error=%s", field_name, e)
                    return {"converted": None, "prompt": prompt, "response": "", "usage": _empty_usage("vocab_convert")}

    # チャンクテキストから特定のメタデータ項目を抽出
    def extract_field(self, field_name, field_desc, chunk_texts):
        logger.info("FIELD_EXTRACT_START | field=%s | text_len=%d", field_name, len(chunk_texts))
        rules = []
        # LLMに渡す抽出対象名（「その他のタイトル」は「タイトル」として抽出）
        extract_as = "タイトル" if field_name == "その他のタイトル" else field_name
        if field_name == "タイトル":
            rules.append(
                "複数のタイトル候補がある場合は、調査全体を最も適切に表す正式なタイトルを1つだけ選んでください。"
                "セミコロン区切りで複数回答しないでください。"
            )
        if field_name in self.verbatim_fields:
            rules.append(f"{field_name}はテキストの内容をそのまま生成してください。")
        if field_name == "観察単位":
            rules.append("何を1つの観測対象として測定・記録しているかの質問に対して、シンプルに単位を回答する内容を生成してください。")
        if field_name == "対象地域":
            rules.append(
                "47都道府県すべて、または日本全域を対象としている場合は「日本（全国）」とのみ回答してください。"
                "個別の都道府県名を列挙しないでください。"
            )
        if field_name == "母集団":
            rules.append(
                "まず縦断的調査か横断的調査かを判断し、その上で具体的にどのような時間的設計"
                "（例: 単一時点の横断調査、パネル調査、繰り返し横断調査、コホート調査 など）"
                "であるかを回答してください。"
            )
        if field_name in ("対象時期", "調査日"):
            rules.append("複数の期間がある場合は、適切な期間を1つ選択して、抽出してください。")
            rules.append("「/」を用いて必ずISO標準の日付形式で期間を記載してください。（YYYY-MM-DD/YYYY-MM-DD）")
        # 参考例には外部レコードの全フィールドを含める
        # （LLMは関連項目を相互参照しながら当該項目の値を判断できる）
        # 複数レコードが参照例にある場合は、各参照例にタイトル（ラベル）を付けて識別できるようにする
        field_background = None
        if self.external_records:
            parts = []
            multi = len(self.external_records) > 1
            for rec in self.external_records:
                fields = rec.get("fields") or {}
                if not fields:
                    continue
                rec_lines: list[str] = []
                if multi:
                    label = rec.get("label") or ""
                    if label:
                        rec_lines.append(f"【{label}】")
                for k, v in fields.items():
                    if v is None:
                        continue
                    val_str = str(v).strip()
                    if not val_str:
                        continue
                    rec_lines.append(f"{k}: {val_str}")
                if rec_lines:
                    parts.append("\n".join(rec_lines))
            if parts:
                field_background = "\n\n---\n\n".join(parts)
        rules = rules or None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with purpose_scope("field_extract"):
                    output = self.extractor.extract(
                        text=chunk_texts,
                        target_class=extract_as,
                        target_class_desc=field_desc,
                        background_text=field_background,
                        rules=rules,
                        survey_name=self.current_survey_name,
                    )
                # extractor.extract 内で行われた最後の LLM 呼び出しの実トークン/料金を取得
                usage = get_last_call_usage() or _empty_usage("field_extract")
                obj = output.object if isinstance(output.object, dict) else {}
                value = obj.get(extract_as, "")
                prompt_text = output.annotations.get("prompt", "")
                raw_response = output.annotations.get("response", "")
                logger.info("FIELD_EXTRACT_DONE | field=%s | value=%s", field_name, str(value)[:100])
                logger.warning("ELAPSED | FIELD_EXTRACT | field=%s | elapsed=%.1fs",
                               field_name, time.time() - self.start_time)
                return field_name, value, prompt_text, raw_response, usage
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = 5 * (2 ** attempt)
                    logger.warning("FIELD_EXTRACT_RETRY | field=%s | attempt=%d/%d | delay=%ds | error=%s",
                                   field_name, attempt + 1, max_retries, delay, e)
                    time.sleep(delay)
                else:
                    logger.error("FIELD_EXTRACT_ERROR | field=%s | error=%s", field_name, e)
                    return field_name, "", "", "", _empty_usage("field_extract")
