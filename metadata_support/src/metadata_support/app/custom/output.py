import csv
import io
import logging

logger = logging.getLogger(__name__)


class OutputWriter:
    """CSV出力を担うクラス。"""

    @staticmethod
    def create_csv(results: list) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["No", "項目名", "DDI-Cパス", "生成結果"])
        for r in results:
            writer.writerow([r["no"], r["name"], r["ddi"], r["value"]])
        return output.getvalue()
