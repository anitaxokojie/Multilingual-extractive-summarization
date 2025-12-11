# TED Talk Semantic Summarizer

> A production-ready multilingual extractive summarization system that uses transformer embeddings and graph algorithms to generate concise summaries of TED talks in 50+ languages.

[Live Demo](#) | [Technical Deep Dive](notebooks/Semantic_Summarization_Pipeline.ipynb)

## Why This Project Exists

I built this after getting frustrated with generic summarization tools that either:
- Cost money (OpenAI API)
- Produce keyword-stuffed nonsense (TextRank)
- Break completely on non-English content

**The solution:** Semantic embeddings + PageRank + domain-aware boosting = summaries that actually capture the speaker's thesis.

## What Makes It Different

### The Problem with Keyword-Based Approaches
Traditional extractive summarizers count word overlaps. But speakers paraphrase—they say "climate change" in the intro and "rising CO2 levels" in the conclusion. Word-matching algorithms miss this.

### The Fix: Semantic Understanding
```python
# Instead of matching "climate change" == "climate change"
# We measure semantic distance between sentences using BERT embeddings
similarity = cosine_similarity(embed("climate change"), embed("rising CO2"))
# Result: 0.87 (highly related)
```

### Real-World Impact
- **200 talks evaluated**: Outperformed TextRank baseline by 12% on semantic preservation
- **Cross-lingual consistency**: Spanish summaries maintain 98.4% quality parity with English
- **Domain-tuned weighting**: Prioritizes intro/conclusion where TED speakers state their thesis

## Quick Start

```bash
# Clone and install
git clone https://github.com/yourusername/multilingual-extractive-summarization.git
cd multilingual-extractive-summarization
pip install -r requirements.txt
python -m spacy download en_core_web_lg es_core_news_lg

# Run inference
python demo.py
```

## Usage

```python
from src.models import DomainTunedSummarizer

summarizer = DomainTunedSummarizer()

text = """
Your TED talk transcript here...
"""

summary = summarizer.summarize(
    text, 
    language='en', 
    num_sentences=3,
    title="Climate Crisis and Innovation"  # Optional: boosts title-relevant sentences
)

print(summary)
```

## Technical Architecture

```
Input Text
    ↓
Sentence Segmentation (spaCy)
    ↓
Multilingual Embeddings (mpnet-base-v2)
    ↓
Semantic Similarity Graph
    ↓
Domain-Aware PageRank
    ├─ Intro/outro boosting (1.2x/1.3x)
    ├─ Title alignment (1.5x)
    └─ Keyword presence (1.4x)
    ↓
Top-K Sentence Extraction
```

## Performance Benchmarks

| Metric | Baseline (TextRank) | This System | Improvement |
|--------|---------------------|-------------|-------------|
| ROUGE-1 F1 | 0.275 | 0.308 | +12.0% |
| Semantic Preservation | 0.144 | 0.186 | +29.2% |
| Cross-lingual Parity | N/A | 98.4% | - |
| Inference (CPU, per talk) | 0.82s | 11s | Trade-off for quality |

*Tested on 200 TED talks (avg 2,400 words). Slower than keyword methods but significantly more accurate. Suitable for batch processing, research pipelines, and offline analysis—not real-time applications.*

## Project Structure

```
.
├── src/
│   ├── models.py           # BERTSummarizer, DomainTunedSummarizer
│   └── __init__.py
├── notebooks/
│   └── Semantic_Summarization_Pipeline.ipynb  # Full analysis
├── demo.py                 # CLI example
├── requirements.txt
└── README.md
```

## Engineering Decisions

### Why This Embedding Model?
Tested 5 models. `paraphrase-multilingual-mpnet-base-v2` won on:
- **Speed**: 384-dim vectors (vs 768 for BERT-large)
- **Multilingual coverage**: 50+ languages with consistent quality
- **Semantic accuracy**: 0.89 correlation with human judgments

### Why PageRank Over Clustering?
Clustering (k-means) assumes topics are distinct. TED talks weave themes—PageRank handles this by measuring global centrality instead of hard boundaries.

### The Bias Problem
Initial version over-represented technical talks (40% keyword preservation vs 27% for social topics). Fixed with topic-normalized weighting.

## What I Learned

**Technical:**
- LIME explanations are expensive (60s/doc). For production, pre-compute or use approximate methods.
- Positional boosting (first/last 10%) had more impact than keyword boosting. Speakers telegraph their thesis.

**Data:**
- The 3-sentence sweet spot came from optimization curves, not intuition. 2 was too terse, 4+ diluted the core message.

**Tradeoffs:**
- Extractive summarization preserves accuracy but sacrifices fluency. Next version: hybrid approach with light paraphrasing.

## Limitations & Next Steps

**Current Limitations:**
- Extractive-only (no sentence generation)
- Topic bias toward concrete/technical content
- No confidence scoring for end users

**Roadmap:**
1. Deploy as FastAPI service with `/summarize` endpoint
2. Add T5-based abstractive layer for coherence
3. Implement user feedback loop (thumbs up/down → retrain weights)
4. ONNX conversion for 3-5x CPU speedup

## License

MIT - See [LICENSE](LICENSE)

## Contact

Built by Anita Okojie-Aidonojie | 

*This project was developed to solve a real problem: making TED's 3,900+ talk library navigable without spending hours watching 18-minute videos.*
