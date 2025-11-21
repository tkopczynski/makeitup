"""Data generation package for creating synthetic datasets using LLM."""

__version__ = "0.1.0"

from data_generation.api import generate_dataset
from data_generation.core.output_formats import SUPPORTED_FORMATS, write_dataframe

__all__ = [
    "__version__",
    "generate_dataset",
    "write_dataframe",
    "SUPPORTED_FORMATS",
]
