"""Basic Extractor that is purely example driven."""

import json
import logging
import re
import yaml

from dataclasses import dataclass
from typing import List
from pydantic import ConfigDict
from metadata_support.formatters.format_utils import remove_formatting
from .extractor import AnnotatedObject, Extractor

logger = logging.getLogger(__name__)


@dataclass
class BasicExtractor(Extractor):
    """
    Extractor that is purely example driven.
    """

    model_config = ConfigDict(protected_namespaces=())

    serialization_format: str = "json"
    model_name: str = "gpt-4o"

    def extract(
        self,
        text: str,
        target_class: str,
        target_class_desc: str = None,
        examples: List[AnnotatedObject] = None,
        background_text: str = None,
        rules: List[str] = None,
        min_examples=1,
        survey_name: str = None,
        **kwargs,
    ) -> AnnotatedObject:
        """参考例の有無でプロンプトを切り替えるディスパッチャ。"""
        if background_text:
            return self.extract_with_background(
                text=text,
                target_class=target_class,
                target_class_desc=target_class_desc,
                background_text=background_text,
                rules=rules,
                survey_name=survey_name,
            )
        return self.extract_without_background(
            text=text,
            target_class=target_class,
            target_class_desc=target_class_desc,
            rules=rules,
        )

    def extract_with_background(
        self,
        text: str,
        target_class: str,
        target_class_desc: str = None,
        background_text: str = None,
        rules: List[str] = None,
        survey_name: str = None,
    ) -> AnnotatedObject:
        """参考例（外部リポジトリ由来）がある場合のプロンプト。

        survey_name: 先行フェーズで生成した「調査名」（タイトル + その他のタイトル）。
        参考例のうちどの調査に対応するものかを識別するために提示する。
        """
        logger.debug(f"Basic extractor (with background): {text}, target_class={target_class}")
        desc_part = f"（{target_class_desc}）" if target_class_desc else ""
        prompt = f"""## 指示
以下のTextは論文または調査報告書から検索された抜粋です。
参考例は外部リポジトリから取得した類似調査のメタデータ全項目で、各参考例には対応する調査のタイトルが付与されています。
下記の「調査名」は今回のTextに対応する調査の名称です。調査名と最も近いタイトルを持つ参考例を優先して参照し、
参考例の中から「{target_class}」{desc_part}に該当する項目の値を抽出して生成してください。

## 生成ルール
- 該当する情報が複数ある場合は「;」（セミコロン）で区切って1つの文字列として回答してください。
- 文の途中で切れた不完全な文章ではなく、意味が通じる簡潔な表現で回答してください。
- Textが無い場合や該当する情報がない場合は該当なしと回答してください。
- Textがある場合は必ず生成してください。Textがあるのに該当なしと回答するのは誤りです。
- 回答はJSON形式のみとし、前置きや説明文は含めないでください。
- Textに情報が無く、参考例に「{target_class}」の項目がある場合は必ずその値をそのまま生成してください。参考例はこの場合以外では利用しません。
- 参考例が複数ある場合は、下記「調査名」と最も近いタイトルを持つ参考例を優先してください。"""
        if rules:
            for rule in rules:
                prompt += f"- {rule}\n\n"
        if survey_name:
            prompt += f"## 調査名\n{survey_name}\n\n"
        prompt += f"## 参考例\n{background_text}\n\n"
        prompt += f"""
## 出力形式
{{ "{target_class}": "生成した値" }}

"""
        prompt += f"\n## Text:\n{text}\n\n"
        model = self.model
        logger.info(f"Prompt: {prompt}")
        response = model.prompt(prompt)
        raw_response = response.text()
        ao = self.deserialize(raw_response)
        ao.annotations["prompt"] = prompt
        ao.annotations["response"] = raw_response
        return ao

    def extract_without_background(
        self,
        text: str,
        target_class: str,
        target_class_desc: str = None,
        rules: List[str] = None,
    ) -> AnnotatedObject:
        """参考例がない場合のプロンプト。Textのみから抽出する。"""
        logger.debug(f"Basic extractor (no background): {text}, target_class={target_class}")
        desc_part = f"（{target_class_desc}）" if target_class_desc else ""
        prompt = f"""## 指示
以下のTextは論文または調査報告書から検索された抜粋です。
この中から「{target_class}」{desc_part}に該当する情報を生成してください。

## 生成ルール
- 該当する情報が複数ある場合は「;」（セミコロン）で区切って1つの文字列として回答してください。
- 文の途中で切れた不完全な文章ではなく、意味が通じる簡潔な表現で回答してください。
- Textが無い場合や該当する情報がない場合は該当なしと回答してください。
- Textがある場合は必ず生成してください。Textがあるのに該当なしと回答するのは誤りです。
- 回答はJSON形式のみとし、前置きや説明文は含めないでください。
"""
        if rules:
            for rule in rules:
                prompt += f"- {rule}\n"
        prompt += f"""
## 出力形式
{{ "{target_class}": "抽出した値" }}

"""
        prompt += f"\n## Text:\n{text}\n\n"
        model = self.model
        logger.info(f"Prompt: {prompt}")
        response = model.prompt(prompt)
        raw_response = response.text()
        ao = self.deserialize(raw_response)
        ao.annotations["prompt"] = prompt
        ao.annotations["response"] = raw_response
        return ao

    def serialize(self, ao: AnnotatedObject) -> str:
        if isinstance(ao.object, str):
            return ao.object
        if self.serialization_format == "yaml":
            return yaml.dump(ao.object, sort_keys=False)
        else:
            return json.dumps(ao.object, ensure_ascii=False)

    def deserialize(self, text: str, format=None, **kwargs) -> AnnotatedObject:
        if format is None:
            format = self.serialization_format
        if format == "yaml":
            return self.deserialize_yaml(text, **kwargs)
        logger.debug(f"Parsing {text}")
        text = remove_formatting(text=text, expect_format="json")
        try:
            obj = json.loads(text)
            if isinstance(obj, str):
                if self.raise_error_if_unparsable:
                    raise ValueError(f"Could not parse {text}")
                else:
                    obj = {}
            return AnnotatedObject(object=obj)
        except Exception as e:
            match = re.search(r"\{.*\}", text)
            if match:
                if match.group() != text:
                    return self.deserialize(match.group())
            if self.raise_error_if_unparsable:
                raise e
            else:
                logger.warning(f"Could not parse {text}")
                return AnnotatedObject(object={})

    def deserialize_yaml(self, text: str, multiple=False) -> AnnotatedObject:
        logger.debug(f"Parsing YAML: {text}")
        text = remove_formatting(text=text, expect_format="yaml")
        try:
            if multiple:
                obj = yaml.safe_load_all(text)
            else:
                obj = yaml.safe_load(text)
            if isinstance(obj, str):
                if self.raise_error_if_unparsable:
                    raise ValueError(f"Could not parse {text}")
                else:
                    obj = {}
            return AnnotatedObject(object=obj)
        except Exception as e:
            logger.warning(f"Could not parse {text} due to {e}")
            return AnnotatedObject(object={})
