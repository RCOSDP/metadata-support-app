"""
CurateGPT: A framework semi-assisted curation of knowledge bases.

Architecture
============

* :mod:`.store`: json object stores that allow for embedding based search
* :mod:`.wrappers`: wraps external APIs and data sources for ingest
* :mod:`.extract`: extraction of json objects from LLMs
* :mod:`.agents`: agents that chain together search and generate components
* :mod:`.formatters`: formats data objects for presentation to humans and machine agents
* :mod:`.app`: streamlit application


"""

import importlib
import importlib_metadata

try:
    __version__ = importlib_metadata.version(__name__)
except importlib_metadata.PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0"  # pragma: no cover

__all__ = ["Extractor", "BasicExtractor"]

_LAZY_IMPORTS = {
    "BasicExtractor": ("metadata_support.extract", "BasicExtractor"),
    "Extractor": ("metadata_support.extract", "Extractor"),
}


def __getattr__(name):
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'metadata_support' has no attribute {name!r}")
    module = importlib.import_module(target[0])
    value = getattr(module, target[1])
    globals()[name] = value
    return value
