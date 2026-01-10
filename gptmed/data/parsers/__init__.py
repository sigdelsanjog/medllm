"""
Data parsers for MedQuAD dataset.

This module contains parsers to extract and process medical Q&A pairs
from various XML sources.
"""

from .medquad_parser import MedQuADParser

__all__ = ["MedQuADParser"]
