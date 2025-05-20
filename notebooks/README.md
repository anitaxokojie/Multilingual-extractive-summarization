# Project Notebooks

This directory contains the Jupyter notebooks for the TED Talk Multilingual Extractive Summarization project.

## Main Notebook: TED Talk Summarization

The [TED_Talk_Summarization.ipynb](TED_Talk_Summarization.ipynb) notebook contains all the code, analysis, and visualizations for this project, including:

- Data loading and preprocessing of TED Talk transcripts in English and Spanish
- Exploratory data analysis with word frequency and sentence distribution analysis
- Implementation of baseline summarization using TextRank
- Development of BERT-based extractive summarization with sentence embeddings
- Domain-specific tuning for TED Talk content
- Comprehensive evaluation using ROUGE, BLEU, and content coverage metrics
- Cross-lingual performance analysis comparing English and Spanish models
- Fairness assessment across different topic categories
- Explainability implementation with LIME
- Optimization experiments to determine ideal parameters

The notebook includes detailed markdown explanations of each step and the reasoning behind implementation decisions.

## Running the Notebook

To run this notebook:

1. Install all dependencies from the root `requirements.txt` file
2. Download the TED Talks dataset from Kaggle: [TED Ultimate Dataset](https://www.kaggle.com/datasets/miguelcorraljr/ted-ultimate-dataset)
3. Update the file paths in the notebook to point to your local copy of the dataset
4. Run all cells sequentially

Note: Running the full notebook requires significant computational resources, especially for the transformer-based models.

## From Notebook to Production Code

The clean, refactored implementation of the models developed in this notebook can be found in the `src/` directory of this repository, demonstrating the transition from exploratory notebook to production-ready code.
