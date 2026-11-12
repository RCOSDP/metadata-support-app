import csv
import io
import logging

from metadata_support.exporters import build_cdif_json_text, build_weko_tsv

logger = logging.getLogger(__name__)


class OutputWriter:
    """CSV・TSV・JSON出力を担うクラス。"""

    @staticmethod
    def create_csv(results: list) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["No", "項目名", "DDI-Cパス", "生成結果"])
        for r in results:
            writer.writerow([r["no"], r["name"], r["ddi"], r["value"]])
        return output.getvalue()

    @staticmethod
    def create_tsv(results: list) -> str:
        """WEKO3 一括登録用 TSV（アイテムタイプ JDCat(22)）を作成する。"""
        return build_weko_tsv(results)

    @staticmethod
    def create_json(results: list) -> str:
        """CDIF Core 1.1 準拠の JSON-LD を作成する。"""
        return build_cdif_json_text(results)
