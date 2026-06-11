"""
CurateGPT Extractors.

These handle connections to (remote or local) LLMs,
and can also extract structured objects from text.

* Base class: :class:`Extractor`
"""

from .basic_extractor import BasicExtractor
from .extractor import AnnotatedObject, Extractor

__all__ = [
    "BasicExtractor",
    "AnnotatedObject",
    "Extractor",
]
