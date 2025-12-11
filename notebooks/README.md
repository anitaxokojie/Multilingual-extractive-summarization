# Research Notebook

## Semantic_Summarization_Pipeline.ipynb

This notebook contains the full experimental pipeline, including:
- Data exploration and preprocessing
- Baseline model (PyTextRank) implementation
- BERT-based model development
- Domain-specific tuning experiments
- Evaluation with ROUGE, BLEU, and content coverage
- Cross-lingual analysis (English vs Spanish)
- Fairness assessment across topics
- Explainability with LIME
- Hyperparameter optimization

**Note:** The notebook shows the research process, including failed experiments and iterative improvements. For production usage, see the refactored code in `src/models.py`.

### Running the Notebook
```bash
# Install Jupyter
pip install jupyter

# Start notebook server
jupyter notebook

# Open Semantic_Summarization_Pipeline.ipynb
```

### Data Requirements

The notebook expects the TED Talks dataset from Kaggle:
- [TED Ultimate Dataset](https://www.kaggle.com/datasets/miguelcorraljr/ted-ultimate-dataset)
- Download and extract to a `data/` folder (not included in repo due to size)
