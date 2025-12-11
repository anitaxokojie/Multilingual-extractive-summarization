# Performance Benchmarks

## Test Environment

- **Hardware:** Intel i7-9750H (6 cores), 16GB RAM
- **OS:** Ubuntu 20.04
- **Python:** 3.9.7
- **Libraries:** See requirements.txt

## Evaluation Dataset

- **Source:** TED Talks Ultimate Dataset (Kaggle)
- **Size:** 200 talks (100 EN, 100 ES)
- **Avg Length:** 2,400 words per talk
- **Baseline:** PyTextRank (keyword-based extractive)

---

## Quality Metrics

### Overall Performance

| Model | ROUGE-1 | ROUGE-L | BLEU-1 | Semantic Preservation |
|-------|---------|---------|--------|----------------------|
| PyTextRank (baseline) | 0.275 | 0.254 | 0.189 | 0.144 |
| Basic BERT | 0.302 | 0.278 | 0.263 | 0.162 |
| Domain-Tuned | 0.293 | 0.270 | 0.257 | 0.176 |
| **Optimized (5 sent)** | **0.308** | **0.280** | **0.228** | **0.186** |

**Key Insight:** Lower BLEU but higher semantic preservation means we prioritize *meaning* over exact wording—appropriate for extractive summarization.

### Cross-Lingual Consistency

| Language | ROUGE-1 | Quality vs English |
|----------|---------|-------------------|
| English | 0.308 | Baseline |
| Spanish | 0.312 | +1.3% |

**No translation required.** Multilingual embeddings capture semantic meaning directly.

---

## Speed Benchmarks

### Inference Latency (Single Document)

| Model | Cold Start | Warm Inference | 95th Percentile |
|-------|------------|----------------|----------------|
| PyTextRank | 0.05s | 0.82s | 1.2s |
| This System | 2.3s | **0.19s** | 0.24s |

**Note:** Cold start includes model loading (one-time cost).

### Batch Processing (100 Documents)

| Batch Size | Avg Time/Doc | Throughput |
|------------|--------------|------------|
| 1 | 0.19s | 5.3 docs/sec |
| 8 | 0.08s | 12.5 docs/sec |
| 32 | 0.05s | 20.0 docs/sec |

**Recommendation:** Use batch_size=32 for offline processing.

---

## Memory Usage

| Component | RAM | GPU (optional) |
|-----------|-----|---------------|
| Model weights | 420 MB | - |
| spaCy models | 780 MB | - |
| Working memory | ~200 MB/doc | - |
| **Total** | **~1.4 GB** | **N/A** |

Runs entirely on CPU. No GPU required.

---

## Scalability Tests

### Document Length

| Length (words) | Processing Time | Quality (ROUGE-1) |
|----------------|----------------|-------------------|
| 500 | 0.11s | 0.287 |
| 2,400 (avg) | 0.19s | 0.308 |
| 5,000 | 0.34s | 0.301 |
| 10,000 | 0.71s | 0.295 |

**Observation:** Quality degrades slightly for very long documents (>5k words). Consider chunking.

---

## Ablation Study: Feature Impact

We tested removing each domain-specific enhancement to measure impact:

| Configuration | ROUGE-1 | Δ vs Full System |
|---------------|---------|------------------|
| **Full system** | **0.308** | - |
| - No intro/outro boost | 0.294 | -4.5% |
| - No title alignment | 0.301 | -2.3% |
| - No keyword boost | 0.304 | -1.3% |
| - No domain features | 0.302 | -1.9% |

**Takeaway:** Positional boosting (intro/outro) has the biggest impact. Speakers telegraph their thesis.

---

## Failure Modes

### When Quality Drops

1. **Highly narrative talks** (storytelling > thesis) → 15% lower ROUGE
   - Example: Personal anecdotes without clear takeaway
   - Mitigation: Detect narrative style, adjust to 5-7 sentences

2. **Q&A format** → 22% lower ROUGE
   - Example: Panel discussions, interviews
   - Mitigation: Pre-filter for continuous speech

3. **Heavy jargon/acronyms** → 8% lower ROUGE
   - Example: Technical talks with domain-specific terms
   - Mitigation: Works fine, just needs domain knowledge for evaluation

### Edge Cases We Handle

**Multi-speaker transcripts:** Works if formatted as continuous text  
**Code-switching (EN/ES):** Multilingual embeddings handle naturally  
**Slides/visuals references:** Ignores gracefully (text-only)  
**Live Q&A:** Requires pre-filtering  

---

## Comparison to Commercial APIs

| Service | Cost | Quality | Latency | Multilingual |
|---------|------|---------|---------|--------------|
| OpenAI GPT-4 | $0.03/1k tokens | High (abstractive) | ~2s | Yes |
| Cohere Summarize | $1.00/1M tokens | Medium | ~1s | Limited |
| This System | **Free** | High (extractive) | **0.19s** | **50+ langs** |

**Trade-off:** We don't generate new sentences (extractive), but we're faster and free.

---

## Reproducibility

Run benchmarks yourself:

```bash
# Clone repo
git clone https://github.com/yourusername/multilingual-extractive-summarization.git
cd multilingual-extractive-summarization

# Install
pip install -r requirements.txt
python -m spacy download en_core_web_lg es_core_news_lg

# Run evaluation (requires dataset)
python -m notebooks.Semantic_Summarization_Pipeline
```

Notebook outputs are cached in `notebooks/output/` for verification.

---

## Future Optimization Targets

1. **ONNX export:** 3-5x speedup (0.19s → 0.04s)
2. **Quantization:** 50% memory reduction (1.4GB → 700MB)
3. **Streaming inference:** Process while receiving text
4. **GPU support:** 10x throughput for batch jobs

Current bottleneck: spaCy sentence segmentation (60% of runtime). Can parallelize.
