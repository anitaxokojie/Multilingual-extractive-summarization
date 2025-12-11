"""
Production-ready summarization models for TED Talk transcripts.

This module implements BERT-based extractive summarization with domain-specific
optimizations for TED Talk content structure.
"""

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from typing import List, Optional


class BERTSummarizer:
    """
    Basic BERT-based extractive summarizer using sentence embeddings and PageRank.
    
    Uses multilingual sentence transformers to generate semantic embeddings,
    then applies PageRank algorithm to identify central sentences.
    """
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-mpnet-base-v2'):
        """
        Initialize the summarizer with a pre-trained sentence transformer model.
        
        Args:
            model_name: HuggingFace model identifier for sentence embeddings
        """
        self.model = SentenceTransformer(model_name)
        
        # Load spaCy models for sentence segmentation
        try:
            self.nlp_en = spacy.load("en_core_web_lg")
            self.nlp_es = spacy.load("es_core_news_lg")
        except OSError:
            print("⚠️  spaCy models not found. Installing...")
            print("Run: python -m spacy download en_core_web_lg")
            print("Run: python -m spacy download es_core_news_lg")
            raise
    
    def _get_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Generate sentence embeddings using the transformer model."""
        return self.model.encode(sentences)
    
    def _cosine_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity matrix between all sentence pairs.
        
        Args:
            embeddings: Matrix of sentence embeddings (n_sentences, embedding_dim)
            
        Returns:
            Similarity matrix (n_sentences, n_sentences)
        """
        # Normalize embeddings to unit vectors
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm_embeddings = embeddings / (norm + 1e-9)
        
        # Cosine similarity is just dot product of normalized vectors
        return np.dot(norm_embeddings, norm_embeddings.T)
    
    def _pagerank(self, similarity_matrix: np.ndarray, 
                  damping: float = 0.85, 
                  max_iter: int = 100) -> np.ndarray:
        """
        Apply PageRank algorithm to identify central sentences.
        
        Args:
            similarity_matrix: Sentence similarity matrix
            damping: Probability of following edges (vs random jump)
            max_iter: Maximum iterations for convergence
            
        Returns:
            PageRank scores for each sentence
        """
        n = len(similarity_matrix)
        pr = np.ones(n) / n  # Uniform initialization
        
        # Normalize similarity matrix (row-stochastic)
        row_sums = similarity_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        norm_matrix = similarity_matrix / row_sums
        
        # Power iteration
        for _ in range(max_iter):
            pr_new = (1 - damping) / n + damping * (norm_matrix.T @ pr)
            
            # Check convergence
            if np.allclose(pr, pr_new, rtol=1e-6):
                break
            pr = pr_new
        
        return pr
    
    def summarize(self, text: str, 
                  language: str = 'en', 
                  num_sentences: int = 3) -> str:
        """
        Generate extractive summary by selecting top-ranked sentences.
        
        Args:
            text: Input text to summarize
            language: Language code ('en' or 'es')
            num_sentences: Number of sentences to extract
            
        Returns:
            Summary as concatenated sentences
        """
        # Select appropriate language model
        nlp = self.nlp_en if language == 'en' else self.nlp_es
        
        # Segment into sentences
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]
        
        # Edge case: if text is already short
        if len(sentences) <= num_sentences:
            return text
        
        # Generate embeddings
        embeddings = self._get_embeddings(sentences)
        
        # Calculate similarity graph
        sim_matrix = self._cosine_similarity(embeddings)
        
        # Apply PageRank
        scores = self._pagerank(sim_matrix)
        
        # Select top sentences
        top_indices = np.argsort(scores)[::-1][:num_sentences]
        top_indices = sorted(top_indices)  # Maintain original order
        
        summary = ' '.join([sentences[i] for i in top_indices])
        return summary


class DomainTunedSummarizer(BERTSummarizer):
    """
    Enhanced summarizer with domain-specific optimizations for TED Talks.
    
    Applies boosting factors based on:
    - Position (intro/conclusion)
    - Title relevance
    - Domain keywords
    """
    
    # TED-specific keywords that often signal main points
    TED_KEYWORDS = [
        'idea', 'innovation', 'discovery', 'research', 'experiment',
        'future', 'change', 'global', 'technology', 'science',
        'society', 'challenge', 'solution', 'problem', 'world'
    ]
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-mpnet-base-v2'):
        super().__init__(model_name)
        
        # Tuning parameters (empirically optimized)
        self.intro_boost = 1.2      # First 10% of talk
        self.conclusion_boost = 1.3  # Last 10% of talk
        self.keyword_boost = 1.4     # Sentences with domain keywords
        self.title_boost = 1.5       # Sentences similar to title
    
    def summarize(self, text: str,
                  language: str = 'en',
                  num_sentences: int = 3,
                  title: Optional[str] = None) -> str:
        """
        Generate domain-tuned summary with structural awareness.
        
        Args:
            text: Input text to summarize
            language: Language code ('en' or 'es')
            num_sentences: Number of sentences to extract (default: 5)
            title: Optional talk title for relevance boosting
            
        Returns:
            Summary as concatenated sentences
        """
        nlp = self.nlp_en if language == 'en' else self.nlp_es
        
        # Segment sentences
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]
        
        if len(sentences) <= num_sentences:
            return text
        
        # Generate embeddings
        embeddings = self._get_embeddings(sentences)
        sim_matrix = self._cosine_similarity(embeddings)
        
        # Apply domain-specific boosts
        boost_factors = np.ones(len(sentences))
        
        # 1. Positional boosting (TED speakers telegraph thesis at start/end)
        intro_end = max(1, int(len(sentences) * 0.1))
        boost_factors[:intro_end] *= self.intro_boost
        
        conclusion_start = min(len(sentences) - 1, int(len(sentences) * 0.9))
        boost_factors[conclusion_start:] *= self.conclusion_boost
        
        # 2. Keyword boosting (domain relevance)
        for i, sentence in enumerate(sentences):
            if any(kw in sentence.lower() for kw in self.TED_KEYWORDS):
                boost_factors[i] *= self.keyword_boost
        
        # 3. Title alignment boosting (if title provided)
        if title:
            title_embedding = self._get_embeddings([title])[0]
            title_norm = title_embedding / np.linalg.norm(title_embedding)
            
            for i, embedding in enumerate(embeddings):
                sent_norm = embedding / np.linalg.norm(embedding)
                similarity = np.dot(title_norm, sent_norm)
                
                # Boost if semantically similar to title
                if similarity > 0.5:
                    boost_factors[i] *= self.title_boost
        
        # Apply boosts to similarity matrix (multiplicative on both dimensions)
        for i in range(len(sim_matrix)):
            sim_matrix[i, :] *= boost_factors
            sim_matrix[:, i] *= boost_factors[i]
        
        # Run PageRank on boosted graph
        scores = self._pagerank(sim_matrix)
        
        # Extract top sentences
        top_indices = np.argsort(scores)[::-1][:num_sentences]
        top_indices = sorted(top_indices)
        
        summary = ' '.join([sentences[i] for i in top_indices])
        return summary


# Factory function for convenience
def create_summarizer(tuned: bool = True) -> BERTSummarizer:
    """
    Factory function to create summarizer instances.
    
    Args:
        tuned: If True, returns DomainTunedSummarizer; else basic BERTSummarizer
        
    Returns:
        Summarizer instance
    """
    if tuned:
        return DomainTunedSummarizer()
    return BERTSummarizer()
