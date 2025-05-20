import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import spacy

class BERTSummarizer:
    """
    A summarizer using BERT sentence embeddings for extractive summarization
    """
    def __init__(self, model_name='paraphrase-multilingual-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
        # Load spaCy models
        try:
            self.nlp_en = spacy.load("en_core_web_sm")
            self.nlp_es = spacy.load("es_core_news_sm")
        except OSError:
            # If models aren't installed, suggest installation
            print("Please install spaCy models with:")
            print("python -m spacy download en_core_web_sm")
            print("python -m spacy download es_core_news_sm")
            raise
    
    def _get_sentence_embeddings(self, sentences):
        """Get embeddings for a list of sentences"""
        return self.model.encode(sentences)
    
    def _cosine_similarity(self, embeddings):
        """Calculate cosine similarity between sentence embeddings"""
        # Normalize embeddings
        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Calculate similarity matrix
        similarity_matrix = np.dot(norm_embeddings, norm_embeddings.T)
        return similarity_matrix
    
    def _pagerank(self, similarity_matrix, damping=0.85, max_iter=100, tol=1e-6):
        """Simplified PageRank implementation"""
        n = len(similarity_matrix)
        # Initialize pagerank scores
        pr = np.ones(n) / n
        
        # Normalize similarity matrix by row sums
        row_sums = similarity_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        norm_similarity = similarity_matrix / row_sums
        
        # PageRank iterations
        for _ in range(max_iter):
            pr_prev = pr.copy()
            pr = (1 - damping) / n + damping * (norm_similarity.T @ pr)
            
            # Check convergence
            if np.abs(pr - pr_prev).sum() < tol:
                break
                
        return pr
    
    def summarize(self, text, language='en', num_sentences=3):
        """Generate extractive summary using BERT embeddings and PageRank"""
        # Use the appropriate spaCy model based on language
        nlp = self.nlp_en if language == 'en' else self.nlp_es
        
        # Process text with spaCy
        doc = nlp(text)
        
        # Extract sentences
        sentences = [sent.text.strip() for sent in doc.sents]
        
        # Skip summarization if there are too few sentences
        if len(sentences) <= num_sentences:
            return text
        
        # Generate sentence embeddings
        embeddings = self._get_sentence_embeddings(sentences)
        
        # Calculate similarity matrix
        similarity_matrix = self._cosine_similarity(embeddings)
        
        # Apply PageRank algorithm
        scores = self._pagerank(similarity_matrix)
        
        # Get top sentences
        ranked_indices = np.argsort(scores)[::-1][:num_sentences]
        ranked_indices = sorted(ranked_indices)  # Sort by position in document
        
        # Construct summary by joining top sentences
        summary = ' '.join([sentences[i] for i in ranked_indices])
        
        return summary

class DomainTunedSummarizer(BERTSummarizer):
    """
    An enhanced summarizer with domain-specific tuning for TED Talks
    """
    def __init__(self, model_name='paraphrase-multilingual-mpnet-base-v2'):
        super().__init__(model_name)
        
        # TED talk specific parameters
        self.ted_specific_params = {
            'title_boost': 1.5,        # Give more weight to sentences similar to title
            'intro_boost': 1.2,        # Boost sentences from the introduction
            'conclusion_boost': 1.3,   # Boost sentences from the conclusion
            'key_phrase_boost': 1.4    # Boost sentences with TED-specific keywords
        }
        
        # TED talk specific keywords/phrases
        self.ted_keywords = [
            'idea', 'innovation', 'discovery', 'research', 'experiment', 
            'global', 'change', 'future', 'technology', 'science',
            'society', 'challenge', 'opportunity', 'solution', 'problem'
        ]
    
    def summarize(self, text, language='en', num_sentences=5, title=None):
        """Generate domain-tuned extractive summary for TED talks"""
        # Use appropriate language model
        nlp = self.nlp_en if language == 'en' else self.nlp_es
        
        # Process text with spaCy
        doc = nlp(text)
        
        # Extract sentences
        sentences = [sent.text.strip() for sent in doc.sents]
        
        # Skip summarization if too few sentences
        if len(sentences) <= num_sentences:
            return text
        
        # Generate sentence embeddings
        embeddings = self._get_sentence_embeddings(sentences)
        
        # Calculate similarity matrix
        similarity_matrix = self._cosine_similarity(embeddings)
        
        # Apply TED-specific boosts
        boost_factors = np.ones(len(sentences))
        
        # Boost introduction (first 10% of sentences)
        intro_end = max(1, int(len(sentences) * 0.1))
        boost_factors[:intro_end] *= self.ted_specific_params['intro_boost']
        
        # Boost conclusion (last 10% of sentences)
        conclusion_start = min(len(sentences) - 1, int(len(sentences) * 0.9))
        boost_factors[conclusion_start:] *= self.ted_specific_params['conclusion_boost']
        
        # Boost sentences with keywords
        for i, sentence in enumerate(sentences):
            for keyword in self.ted_keywords:
                if keyword.lower() in sentence.lower():
                    boost_factors[i] *= self.ted_specific_params['key_phrase_boost']
                    break
        
        # Boost sentences similar to title if provided
        if title:
            title_embedding = self._get_sentence_embeddings([title])[0]
            title_similarities = []
            
            for embedding in embeddings:
                # Normalize embeddings
                norm_title = title_embedding / np.linalg.norm(title_embedding)
                norm_sent = embedding / np.linalg.norm(embedding)
                
                # Calculate similarity
                similarity = np.dot(norm_title, norm_sent)
                title_similarities.append(similarity)
            
            # Apply title boost
            for i, similarity in enumerate(title_similarities):
                if similarity > 0.5:  # Only boost if reasonably similar
                    boost_factors[i] *= self.ted_specific_params['title_boost']
        
        # Apply boost factors to similarity matrix
        for i in range(similarity_matrix.shape[0]):
            similarity_matrix[i, :] *= boost_factors
            similarity_matrix[:, i] *= boost_factors[i]
        
        # Apply PageRank with tuned similarity
        scores = self._pagerank(similarity_matrix)
        
        # Get top sentences
        ranked_indices = np.argsort(scores)[::-1][:num_sentences]
        ranked_indices = sorted(ranked_indices)  # Sort by position in document
        
        # Construct summary by joining top sentences
        summary = ' '.join([sentences[i] for i in ranked_indices])
        
        return summary
