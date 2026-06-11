"""分割済みチャンクにメタデータ項目（フィールド）を付与するクラス。"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class FieldAnnotator:
    """`llm_processor.annotate_chunks` を並列実行して、各チャンクに該当項目を付与する。

    入力:
      - split_chunks_map: {split_idx: [chunk, ...]}
      - split_position_info: [pos_info, ...]（split_idx 順）
      - split_to_file: [fname, ...]（split_idx 順）
      - field_groups: [{"name": str, "fields_str": str, ...}, ...]
      - llm_processor: `annotate_chunks(split_idx, batch_chunks, group_name,
                                       group_fields_str, position_info,
                                       prev_ctx=None, next_ctx=None)` を持つオブジェクト
      - excluded_group_names: アノテーション対象から除外するグループ名（デフォルト: {"外部情報"}）

    出力:
      - all_chunks: list[dict] - 各要素は annotate 結果に doc_chunk_no/source_file を付加したもの
      - split_logs: dict[(split_idx, group_name, batch_start), {"prompt", "response"}]
    """

    def __init__(
        self,
        split_chunks_map: dict[int, list[str]],
        split_position_info: list[str],
        split_to_file: list[str],
        field_groups: list[dict],
        llm_processor,
        *,
        batch_size: int = 50,
        max_workers: int = 20,
        excluded_group_names=None,
    ):
        self.split_chunks_map = split_chunks_map
        self.split_position_info = split_position_info
        self.split_to_file = split_to_file
        self.field_groups = field_groups
        self.llm_processor = llm_processor
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.excluded_group_names = set(excluded_group_names or {"外部情報"})

        self.all_chunks: list[dict] = []
        self.split_logs: dict[tuple, dict] = {}

    def annotate(self) -> dict:
        """全分割チャンクにメタデータ項目を付与して結果を返す。

        戻り値: {"all_chunks": list[dict], "split_logs": dict}
        """
        active_groups = [
            g for g in self.field_groups if g["name"] not in self.excluded_group_names
        ]
        if not active_groups or not self.split_chunks_map:
            logger.info("LLM_CHUNK_SPLIT_DONE | total_chunks=0")
            return self._result()

        split_results: dict[tuple, list] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for idx, chunks in self.split_chunks_map.items():
                if not chunks:
                    continue
                for batch_start in range(0, len(chunks), self.batch_size):
                    batch_end = min(batch_start + self.batch_size, len(chunks))
                    batch_chunks = chunks[batch_start:batch_end]
                    prev_ctx = chunks[max(0, batch_start - 2):batch_start]
                    next_ctx = chunks[batch_end:batch_end + 2]
                    pos_info = (
                        self.split_position_info[idx]
                        if idx < len(self.split_position_info) else ""
                    )
                    for group in active_groups:
                        future = executor.submit(
                            self.llm_processor.annotate_chunks,
                            idx, batch_chunks, group["name"], group["fields_str"],
                            pos_info,
                            prev_ctx if prev_ctx else None,
                            next_ctx if next_ctx else None,
                        )
                        futures[future] = (idx, group["name"], batch_start)

            for future in as_completed(futures):
                key = futures[future]
                result = future.result()
                split_results[key] = result["results"]
                self.split_logs[key] = {
                    "prompt": result["prompt"],
                    "response": result["response"],
                    "usage": result.get("usage"),
                }

        # 結果を集約し、バッチオフセットを加算して doc_chunk_no を算出
        for (idx, _group_name, batch_start), results in sorted(split_results.items()):
            for chunk in results:
                chunk["doc_chunk_no"] = batch_start + chunk.get("local_chunk_id", 0)
                chunk["source_file"] = (
                    self.split_to_file[idx] if idx < len(self.split_to_file) else ""
                )
                self.all_chunks.append(chunk)

        logger.info("LLM_CHUNK_SPLIT_DONE | total_chunks=%d", len(self.all_chunks))
        return self._result()

    def _result(self) -> dict:
        return {
            "all_chunks": self.all_chunks,
            "split_logs": self.split_logs,
        }
