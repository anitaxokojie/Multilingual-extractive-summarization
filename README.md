# Multilingual-extractive-summarization
BERT-based extractive summarization system for TED Talk transcripts in English and Spanish

# TED Talk Multilingual Extractive Summarization

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A BERT-based extractive summarization system for TED Talk transcripts in both English and Spanish. The system leverages multilingual transformer models to identify and extract the most important sentences while maintaining cross-lingual performance.

## Project Overview

This project implements an extractive summarization system specifically optimized for TED Talk content. Key features include:

- **Multilingual Support**: Works with both English and Spanish transcripts
- **BERT-based Sentence Embeddings**: Uses transformer models to understand semantic relationships
- **Custom PageRank Algorithm**: Identifies key sentences based on their centrality
- **Domain-Specific Tuning**: Enhanced for TED Talk structure and content
- **Explainable Results**: Provides transparency into sentence selection decisions

## Key Results

- **ROUGE-1 F1**: 0.3078
- **BLEU-1**: 0.2281
- **Cross-Lingual Analysis**: Spanish models show +1.6-1.7% better ROUGE scores
- **Optimal Summary Length**: 5 sentences balances quality and conciseness

## Repository Structure

- `notebooks/`: Jupyter notebooks with the full data exploration and model development process
- `src/`: Python modules containing the core implementation
- `data/`: Information about the TED Talk datasets used
- `images/`: Visualizations of evaluation results and model performance
- `examples/`: Sample TED Talk summaries in both languages

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/multilingual-extractive-summarization.git
cd multilingual-extractive-summarization

# Install dependencies
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Download spaCy models
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm

