from setuptools import setup, find_packages

setup(
    name="ted-summarizer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "sentence-transformers>=2.2.2",
        "torch>=2.0.1",
        "spacy>=3.5.3",
        "numpy>=1.24.0",
    ],
    python_requires=">=3.8",
    author="Your Name",
    description="BERT-based extractive summarization for TED talks",
    url="https://github.com/yourusername/multilingual-extractive-summarization",
)
