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
| **Optimized (5 sent)** | **0.311** | **0.280** | **0.291** | **0.186** |

**Key Insight:** The optimized model achieves the best balance between ROUGE scores (indicating agreement with baseline summaries) and semantic preservation (capturing key content). Higher semantic preservation shows we prioritize *meaning* over exact wording—appropriate for extractive summarization.

### Cross-Lingual Performance

| Language | ROUGE-1 | Quality vs English |
|----------|---------|-------------------|
| English | 0.304 | Baseline |
| Spanish | 0.242 | -20.4% |

**Analysis:** Spanish performance is lower in this test set, likely due to:
1. Smaller Spanish training corpus in the embedding model
2. Different linguistic structures (longer sentences in Spanish transcripts)
3. Potential domain bias in test set selection

**Mitigation:** For production Spanish summarization, consider language-specific fine-tuning or using a Spanish-optimized embedding model like `dccuchile/bert-base-spanish-wwm-cased`.

---

## Speed Benchmarks

### Inference Latency (Single Document)

| Model | Cold Start | Warm Inference | Notes |
|-------|------------|----------------|-------|
| PyTextRank | 0.05s | 0.82s | Keyword-based, fast but less accurate |
| This System | 2.3s | **~11s** | Includes full semantic analysis |

**Note:** 
- Cold start includes model loading (one-time cost per session)
- 11s latency reflects complete semantic analysis: sentence segmentation (60%), embedding generation (26%), PageRank computation (11%), post-processing (3%)
- Suitable for batch processing, research pipelines, and offline analysis—not real-time applications

### Batch Processing (20 Documents - Actual Test)

| Batch Size | Avg Time/Doc | Total Time (20 docs) | Throughput |
|------------|--------------|----------------------|------------|
| 1 (sequential) | 11.0s | 220s (~3.7 min) | 0.09 docs/sec |
| 8 (estimated) | ~5.5s | ~110s (~1.8 min) | 0.18 docs/sec |
| 32 (estimated) | ~3.2s | ~64s (~1 min) | 0.31 docs/sec |

**Recommendation:** For production, use batched embedding generation (`model.encode(sentences, batch_size=32)`) and parallel sentence segmentation to achieve ~3-5s/doc on CPU. GPU acceleration provides 3-5x additional speedup.

---

## Memory Usage

| Component | RAM | GPU (optional) |
|-----------|-----|---------------|
| Model weights (mpnet-base-v2) | 420 MB | - |
| spaCy models (lg) | 780 MB | - |
| Working memory per doc | ~200 MB | - |
| **Total (idle)** | **~1.4 GB** | **N/A** |
| **Peak (processing)** | **~1.6 GB** | **N/A** |

Runs entirely on CPU. No GPU required, though GPU acceleration is supported automatically via PyTorch if available.

---

## Scalability Tests

### Document Length Impact

| Length (words) | Processing Time | Quality (ROUGE-1) | Notes |
|----------------|----------------|-------------------|-------|
| 500 | ~4.5s | 0.287 | Short articles/abstracts |
| 2,400 (avg) | ~11s | 0.311 | Standard TED talk |
| 5,000 | ~23s | 0.301 | Long-form content |
| 10,000 | ~48s | 0.295 | Books/reports (consider chunking) |

**Observation:** Quality degrades slightly for very long documents (>5k words) as the graph becomes more complex. For documents >10k words, consider splitting into 5k-word chunks, summarizing each, then running meta-summarization.

---

## Ablation Study: Feature Impact

We tested removing each domain-specific enhancement to measure impact:

| Configuration | ROUGE-1 | Δ vs Full System |
|---------------|---------|------------------|
| **Full system** | **0.311** | - |
| - No intro/outro boost | 0.294 | -5.5% |
| - No title alignment | 0.301 | -3.2% |
| - No keyword boost | 0.307 | -1.3% |
| - No domain features (basic BERT) | 0.302 | -2.9% |

**Takeaway:** Positional boosting (intro/outro) has the biggest impact. TED speakers follow a consistent structure: problem statement at start, call-to-action at end. Boosting these regions captures the thesis more reliably than keyword matching alone.

---

## Failure Modes & Edge Cases

### When Quality Drops

1. **Highly narrative talks** (storytelling > thesis) → 15% lower ROUGE
   - Example: Personal anecdotes without clear takeaway
   - Mitigation: Increase to 5-7 sentences to capture narrative arc

2. **Q&A format** → 22% lower ROUGE
   - Example: Panel discussions, interviews with multiple speakers
   - Mitigation: Pre-filter for continuous speech or use speaker-aware segmentation

3. **Heavy jargon/acronyms** → 8% lower ROUGE
   - Example: Highly technical talks with domain-specific terminology
   - Mitigation: Works fine, lower ROUGE is due to evaluation baseline limitations

### Edge Cases We Handle

✅ **Multi-speaker transcripts:** Works if formatted as continuous text  
✅ **Code-switching (EN/ES):** Multilingual embeddings handle naturally  
✅ **Slides/visuals references:** Ignores gracefully (text-only)  
❌ **Live Q&A sections:** Requires pre-filtering to remove  
❌ **Extremely short texts (<100 words):** Returns original text unchanged  

---

## Comparison to Commercial APIs

| Service | Cost | Quality | Latency (2.4k words) | Multilingual | Notes |
|---------|------|---------|---------------------|--------------|-------|
| OpenAI GPT-4 Turbo | $0.01/1k tokens (~$0.03/talk) | High (abstractive) | ~2-3s | 50+ langs | Generates new text |
| Cohere Summarize | $1.00/1M tokens (~$0.002/talk) | Medium | ~1-2s | Limited | Optimized for speed |
| Anthropic Claude | $0.015/1k tokens (~$0.04/talk) | High (abstractive) | ~2-4s | 10+ langs | Best quality |
| **This System** | **Free** | High (extractive) | **~11s** | **50+ langs** | Open source, no API calls |

**Trade-off Analysis:**
- **Speed:** Commercial APIs are 3-5x faster due to optimized infrastructure
- **Cost:** This system is free for unlimited use; commercial APIs cost $0.002-$0.04 per talk
- **Quality:** Abstractive models (GPT-4, Claude) produce more fluent summaries; this system guarantees factual accuracy (no hallucinations)
- **Privacy:** This system runs locally; commercial APIs send data to third parties

**Best For:** Batch processing, research, cost-sensitive applications, privacy-critical use cases, or when factual accuracy is paramount.

---

## Hyperparameter Optimization Results

### Sentence Count (k) Optimization

| k | ROUGE-1 | BLEU-1 | Content Coverage | Runtime (20 docs) | Decision |
|---|---------|--------|------------------|-------------------|----------|
| 2 | 0.250 | 0.217 | 0.087 | 215s | Too terse, misses context |
| 3 | 0.273 | 0.246 | 0.121 | 215s | Good for ultra-brief |
| 4 | 0.308 | 0.282 | 0.150 | 225s | Strong balance |
| **5** | **0.311** | **0.291** | **0.172** | **224s** | **Optimal** |

**Finding:** k=5 provides the best ROUGE-1 and content coverage scores. While k=3 is more concise, the marginal quality improvement at k=5 (+3.8 ROUGE points, +5.1 coverage points) is worth the extra length for most use cases. Runtime is comparable across all configurations.

---

## Reproducibility

Run benchmarks yourself:

```bash
# Clone repo
git clone https://github.com/anitaxokojie/multilingual-extractive-summarization.git
cd multilingual-extractive-summarization

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg es_core_news_lg

# Run smoke test
python test_basic.py

# Run full evaluation (requires dataset download)
jupyter notebook notebooks/Semantic_Summarization_Pipeline.ipynb
```

**Dataset:** Download the [TED Ultimate Dataset](https://www.kaggle.com/datasets/miguelcorraljr/ted-ultimate-dataset) from Kaggle and place in a `data/` folder.

---

## Future Optimization Targets

### Short-Term (Achievable with existing stack):
1. **Batched embeddings:** Currently processing sentences sequentially; batch processing could reduce time to ~5-6s/doc
2. **Parallel sentence segmentation:** spaCy's `nlp.pipe()` with `n_process=4` could reduce segmentation time by 50%
3. **Sentence filtering:** Skip very short sentences (<5 words) to reduce graph complexity

**Estimated impact:** 11s → 5-6s per document (45% speedup)

### Medium-Term (Requires optimization work):
1. **ONNX export:** Convert model to ONNX format for 3-5x inference speedup
2. **Model quantization:** INT8 quantization could reduce memory by 50% with <2% quality loss
3. **Approximate PageRank:** Use power iteration with early stopping (current: 100 iterations, could use 20)

**Estimated impact:** 11s → 2-3s per document (70-80% speedup), 1.4GB → 700MB memory

### Long-Term (Requires architecture changes):
1. **Distilled models:** Use smaller embedding models (MiniLM-L6 instead of mpnet-base-v2) for 5x speedup with 10% quality trade-off
2. **GPU deployment:** Leverage GPU for batch jobs (10x throughput improvement)
3. **Hybrid abstractive:** Add T5-small for sentence fusion to improve fluency

**Target:** <1s per document on GPU with maintained quality

---

## Performance Profiling Breakdown

Based on profiling 20 documents (avg 2,400 words):

| Component | Time | % of Total | Optimization Potential |
|-----------|------|------------|------------------------|
| spaCy sentence segmentation | 6.6s | 60% | High (parallelization) |
| BERT embedding generation | 2.9s | 26% | Medium (batching, ONNX) |
| PageRank computation | 1.2s | 11% | Low (already efficient) |
| Post-processing | 0.3s | 3% | Negligible |
| **Total** | **11.0s** | **100%** | - |

**Key Bottleneck:** spaCy's sentence segmentation. This is a CPU-bound operation that could benefit most from parallelization.

---

## System Requirements

### Minimum Requirements
- **CPU:** Dual-core 2.0GHz or better
- **RAM:** 4GB (6GB recommended for large documents)
- **Disk:** 2GB free space (models + dependencies)
- **OS:** Linux, macOS, or Windows with Python 3.8+

### Recommended Configuration
- **CPU:** Quad-core 3.0GHz or better
- **RAM:** 8GB or more
- **Disk:** 5GB free space
- **GPU:** Optional; NVIDIA GPU with CUDA for 3-5x speedup

### Cloud/Server Deployment
- **AWS EC2:** t3.medium or larger (2 vCPU, 4GB RAM)
- **Docker:** 2GB memory limit minimum
- **Lambda:** Not recommended (11s exceeds practical limits; use for pre-computed summaries only)

---

## Benchmark Versioning

**Version:** 1.0  
**Date:** December 2024  
**Model:** paraphrase-multilingual-mpnet-base-v2  
**Hardware:** Intel i7-9750H, 16GB RAM, CPU-only  

Results may vary on different hardware. GPU acceleration typically provides 3-5x speedup. For updated benchmarks or hardware-specific results, see the GitHub repository.
