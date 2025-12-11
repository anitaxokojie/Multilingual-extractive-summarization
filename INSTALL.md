# Installation Guide

## Prerequisites

- Python 3.8 or higher
- 4GB RAM minimum
- ~2GB disk space for models

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/multilingual-extractive-summarization.git
cd multilingual-extractive-summarization
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n summarizer python=3.9
conda activate summarizer
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** This will download ~600MB of packages including PyTorch and transformers.


### 4. Download spaCy Language Models
```bash
# Download separately (works on all systems)
python -m spacy download en_core_web_lg
python -m spacy download es_core_news_lg
```

**Note:** The models download ~968MB total. This is normal.

### 5. Verify Installation

```bash
python test_basic.py
```

You should see:
```
✅ ALL TESTS PASSED
Your installation is working correctly!
```

## Quick Start

```bash
# Run the demo
python demo.py

# Or use directly in code
python
>>> from src.models import DomainTunedSummarizer
>>> summarizer = DomainTunedSummarizer()
>>> summary = summarizer.summarize("Your text here...", num_sentences=3)
>>> print(summary)
```

## Troubleshooting

### Issue: "No module named 'sentence_transformers'"

**Solution:**
```bash
pip install sentence-transformers
```

### Issue: "Can't find model 'en_core_web_lg'"

**Solution:**
```bash
python -m spacy download en_core_web_lg
```

### Issue: Out of memory errors

**Solution:** Your system may have insufficient RAM. Try:
1. Process smaller texts (split documents into chunks)
2. Use a smaller embedding model in `src/models.py`:
   ```python
   # Change this line:
   self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
   # To:
   self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # 80MB vs 420MB
   ```

### Issue: Slow performance (>30s per document)

**Possible causes:**
1. Running on very old CPU
2. Not using batch processing for multiple documents
3. Processing very long texts (>10k words)

**Solutions:**
- For batch jobs: Use `model.encode(sentences, batch_size=32)`
- For long texts: Split into 5k-word chunks first
- Consider using GPU if available (automatic with PyTorch)

## Optional: GPU Acceleration

If you have an NVIDIA GPU with CUDA:

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify GPU is detected
python -c "import torch; print(torch.cuda.is_available())"
```

Expected speedup: 3-5x faster on GPU vs CPU.

## Minimal Installation (No spaCy models)

If you only need English and want to save disk space:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm  # Only 12MB

# Note: Accuracy will be slightly lower with 'sm' model
```

## Docker Installation (Alternative)

```bash
docker build -t summarizer .
docker run -it summarizer python demo.py
```

## Development Setup

For contributing or modifying the code:

```bash
# Install additional dev dependencies
pip install pytest black flake8 jupyter

# Run tests
pytest tests/

# Format code
black src/

# Start Jupyter for experiments
jupyter notebook notebooks/
```

## Uninstallation

```bash
# Remove virtual environment
rm -rf venv/

# Or with conda
conda remove -n summarizer --all
```

## System Requirements Summary

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4GB | 8GB+ |
| CPU | Dual-core | Quad-core+ |
| Disk Space | 2GB | 5GB |
| Internet | Required (first run) | Offline after setup |

## Need Help?

- Check existing [GitHub Issues](https://github.com/yourusername/repo/issues)
- Review [TECHNICAL.md](TECHNICAL.md) for architecture details
- See [EXAMPLES.md](EXAMPLES.md) for usage patterns
