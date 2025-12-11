# Experimental Results

This folder contains visualizations from the notebook experiments.

## Files

- **optimization_curve.png**: Comparison of summary quality (ROUGE-1, BLEU-1) across different sentence counts (k=2,3,4,5)
- **cross_lingual_comparison.png**: English vs Spanish performance metrics
- **fairness_heatmap.png**: Topic bias analysis showing keyword retention and ROUGE scores by topic
- **lime_explanation.png**: Example LIME visualization showing word importance for summary stability

### Example LIME Explanation (Talk 79)
   
   Words with positive influence (green): 'a', 'but', 'of', 'that', 'the', 'so', 'this', 'I', 'see'
   Words with negative influence (red): 'why'
   
   **Interpretation:** High-frequency function words stabilize summaries by appearing 
   in multiple sentences, increasing inter-sentence connectivity in the PageRank graph.

All results are generated from `notebooks/Semantic_Summarization_Pipeline.ipynb`.

