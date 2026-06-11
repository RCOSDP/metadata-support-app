"""Chat with a KB."""

import logging
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterator, List, Optional, Union

from metadata_support.extract import Extractor

logger = logging.getLogger(__name__)


@dataclass
class BaseWrapper(ABC):  # noqa: B024
    """
    A virtual store that implements a view over some remote or external source.
    """

    source_locator: Optional[Union[str, Path]] = None

    local_store: Any = None
    """Adapter to local knowledge base used to cache results."""

    extractor: Extractor = None

    name: ClassVar[str] = "__dbview__"

    default_embedding_model = "openai:"

    default_object_type = "Publication"

    prefix: ClassVar[str] = None

    search_limit_multiplier: ClassVar[int] = 3

    max_text_length = 3000
    text_overlap = 200

    def objects(
        self, collection: str = None, object_ids: "Optional[List[str]]" = None, **kwargs
    ) -> Iterator[Dict]:
        if object_ids:
            yield from self.objects_by_ids(object_ids, **kwargs)
        else:
            raise NotImplementedError

    def objects_by_ids(self, object_ids: List[str]) -> List[Dict]:
        raise NotImplementedError

    def external_search(self, text: str, expand: bool = True, **kwargs) -> List[Dict]:
        raise NotImplementedError

    def extract_concepts_from_text(self, text: str, **kwargs):
        model = self.extractor.model
        response = model.prompt(
            text, system="generate a semi-colon separated list of the most relevant terms"
        )
        terms = response.text().split(";")
        return terms

    def split_objects(self, objects: List[Dict], text_field="text", id_field="id") -> List[Dict]:
        new_objects = []
        for obj in objects:
            if len(obj[text_field]) > self.max_text_length:
                obj_id = obj[id_field]
                text = obj[text_field]
                n = 0
                while text:
                    new_obj = obj.copy()
                    n += 1
                    new_obj[id_field] = f"{obj_id}#{n}"
                    new_obj[text_field] = text[: self.max_text_length +
                                               self.text_overlap]
                    new_objects.append(new_obj)
                    text = text[self.max_text_length:]
            else:
                new_objects.append(obj)
        return new_objects

    def create_curie(self, local_id: str) -> str:
        return f"{self.prefix}:{local_id}"

    def wrap_object(self, obj: Dict) -> Iterator[Dict]:
        raise NotImplementedError

    def unwrap_object(self, obj: Dict[str, Any], store: Any, **kwargs) -> Any:
        raise NotImplementedError
