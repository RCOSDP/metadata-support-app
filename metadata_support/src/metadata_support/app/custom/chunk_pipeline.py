"""チャンク分割 + フィールド判定を連続実行する上位オーケストレータ。

個別利用したい場合は `ChunkSplitter` / `FieldAnnotator` を直接呼ぶ。
"""
import logging

from metadata_support.app.custom.chunk_splitter import ChunkSplitter
from metadata_support.app.custom.field_annotator import FieldAnnotator

logger = logging.getLogger(__name__)


class ChunkPipeline:
    """全ファイルのテキストを句点分割し、各チャンクにフィールドを付与する一連のパイプライン。

    入力:
      - all_full_texts: {ファイル名: 正規化済みテキスト}
      - llm_processor: `split_text` / `annotate_chunks` を持つ LLMProcessor
      - field_groups: FIELD_GROUPS
      - batch_size, max_workers, excluded_group_names: FieldAnnotator 用の調整値

    出力（`run()` の戻り値）:
      ChunkSplitter と FieldAnnotator の結果を統合した dict。含まれるキー:
      - split_chunks_map, split_text_logs, split_to_file, split_position_info,
        file_ordered_chunks, num_splits, total_chunks, file_list
      - all_chunks, split_logs
    """

    def __init__(
        self,
        all_full_texts: dict[str, str],
        llm_processor,
        field_groups: list[dict],
        *,
        batch_size: int = 50,
        max_workers: int = 20,
        excluded_group_names=None,
    ):
        self.all_full_texts = all_full_texts
        self.llm_processor = llm_processor
        self.field_groups = field_groups
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.excluded_group_names = excluded_group_names or {"外部情報"}

    def run(self) -> dict:
        split_result = ChunkSplitter(self.all_full_texts, self.llm_processor).split()

        annotate_result = FieldAnnotator(
            split_chunks_map=split_result["split_chunks_map"],
            split_position_info=split_result["split_position_info"],
            split_to_file=split_result["split_to_file"],
            field_groups=self.field_groups,
            llm_processor=self.llm_processor,
            batch_size=self.batch_size,
            max_workers=self.max_workers,
            excluded_group_names=self.excluded_group_names,
        ).annotate()

        return {**split_result, **annotate_result}
