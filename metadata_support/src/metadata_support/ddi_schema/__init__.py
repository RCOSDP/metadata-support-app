"""DDI-Codebook 2.5 XSD から METADATA_FIELDS 用の説明文を抽出・整形するモジュール。"""
from metadata_support.ddi_schema.applier import ApplyResult, apply_descriptions, format_report
from metadata_support.ddi_schema.description_generator import generate_japanese_descriptions
from metadata_support.ddi_schema.runtime import (
    apply_to_metadata_fields,
    build_field_groups,
    build_metadata_fields,
    load_or_build_descriptions,
)
from metadata_support.ddi_schema.xsd_parser import FIELDS, extract_english_descriptions

__all__ = [
    "FIELDS",
    "extract_english_descriptions",
    "generate_japanese_descriptions",
    "apply_descriptions",
    "ApplyResult",
    "format_report",
    "apply_to_metadata_fields",
    "build_field_groups",
    "build_metadata_fields",
    "load_or_build_descriptions",
]
