"""
TED Talk Summarization Package

Provides extractive summarization using BERT embeddings and PageRank.
"""

from .models import BERTSummarizer, DomainTunedSummarizer, create_summarizer

__version__ = "0.1.0"
__all__ = ["BERTSummarizer", "DomainTunedSummarizer", "create_summarizer"]
