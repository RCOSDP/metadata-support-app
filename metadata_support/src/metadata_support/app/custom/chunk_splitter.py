"""ファイル毎のテキストを句点（。）でチャンク分割するクラス。"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class ChunkSplitter:
    """`llm_processor.split_text` を並列実行してチャンク関連の各種マッピングを構築する。

    出力:
      - split_chunks_map: {split_idx: [chunk, ...]}
      - split_text_logs:  {split_idx: {"prompt": str, "response": str}}
      - split_to_file:    [fname, ...]  （split_idx 順）
      - split_position_info: [pos_info, ...]  （split_idx 順）
      - file_ordered_chunks: {fname: ["", chunk1, chunk2, ...]}  # 1-basedアクセス用にダミー先頭
      - num_splits: int
      - total_chunks: int
    """

    def __init__(self, all_full_texts: dict[str, str], llm_processor):
        self.all_full_texts = all_full_texts
        self.llm_processor = llm_processor
        self.file_list = list(all_full_texts.items())

        self.split_chunks_map: dict[int, list[str]] = {}
        self.split_text_logs: dict[int, dict] = {}
        self.split_to_file: list[str] = []
        self.split_position_info: list[str] = []
        self.file_ordered_chunks: dict[str, list[str]] = {}
        self.num_splits: int = 0
        self.total_chunks: int = 0

    def split(self):
        """全ファイルを並列に分割し、結果を返す。

        戻り値: dict - 上記属性の辞書
        """
        if not self.file_list:
            logger.info("SPLIT_TEXT_ALL_DONE | splits=0 | total_chunks=0 | num_files=0")
            return self._result()

        max_workers = max(1, len(self.file_list))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.llm_processor.split_text, fname, ftext): i
                for i, (fname, ftext) in enumerate(self.file_list)
            }
            file_results: dict[int, dict] = {}
            for future in as_completed(futures):
                idx = futures[future]
                file_results[idx] = future.result()

        for split_idx in range(len(self.file_list)):
            result = file_results[split_idx]
            fname = result["fname"]
            chunks = result["chunks"]

            self.split_chunks_map[split_idx] = chunks
            self.split_text_logs[split_idx] = {"prompt": f"(句点分割: {fname})", "response": ""}
            self.split_to_file.append(fname)
            pos_info = (
                f"このテキストはファイル「{fname}」から句点（。）で分割した"
                f"{len(chunks)}個のチャンクです。"
            )
            self.split_position_info.append(pos_info)
            self.file_ordered_chunks[fname] = [""] + chunks

        self.num_splits = len(self.split_chunks_map)
        self.total_chunks = sum(len(v) for v in self.split_chunks_map.values())
        logger.info(
            "SPLIT_TEXT_ALL_DONE | splits=%d | total_chunks=%d | num_files=%d",
            self.num_splits, self.total_chunks, len(self.all_full_texts),
        )
        return self._result()

    def _result(self) -> dict:
        return {
            "split_chunks_map": self.split_chunks_map,
            "split_text_logs": self.split_text_logs,
            "split_to_file": self.split_to_file,
            "split_position_info": self.split_position_info,
            "file_ordered_chunks": self.file_ordered_chunks,
            "num_splits": self.num_splits,
            "total_chunks": self.total_chunks,
            "file_list": self.file_list,
        }
