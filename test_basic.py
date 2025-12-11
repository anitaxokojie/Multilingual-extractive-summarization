"""
Basic smoke tests to verify the summarizer works.

Run with: python test_basic.py
"""

from src.models import BERTSummarizer, DomainTunedSummarizer

def test_basic_summarization():
    """Test that basic summarizer produces output"""
    print("Testing BERTSummarizer...")
    
    summarizer = BERTSummarizer()
    
    text = """
    Climate change is one of the most pressing issues of our time. 
    Scientists agree that human activity is the primary cause. 
    We must reduce carbon emissions immediately. 
    Renewable energy offers a sustainable solution. 
    However, political will is lacking in many countries.
    """
    
    summary = summarizer.summarize(text, language='en', num_sentences=2)
    
    assert len(summary) > 0, "Summary should not be empty"
    assert len(summary) < len(text), "Summary should be shorter than original"
    assert '.' in summary, "Summary should contain sentences"
    
    print(f"✅ Basic summarization works")
    print(f"   Input: {len(text.split())} words")
    print(f"   Output: {len(summary.split())} words")
    print(f"   Summary: {summary[:100]}...\n")


def test_domain_tuned():
    """Test that domain-tuned summarizer works with title"""
    print("Testing DomainTunedSummarizer...")
    
    summarizer = DomainTunedSummarizer()
    
    text = """
    Thank you for having me here today. I want to talk about innovation.
    Innovation drives progress in every field.
    From technology to medicine, innovation changes lives.
    But innovation requires failure. We must embrace failure.
    Only through failure do we learn. Thank you.
    """
    
    summary = summarizer.summarize(
        text, 
        language='en', 
        num_sentences=2,
        title="The Power of Innovation"
    )
    
    assert len(summary) > 0
    assert 'innovation' in summary.lower() or 'failure' in summary.lower()
    
    print(f"✅ Domain-tuned summarization works")
    print(f"   Summary: {summary}\n")


def test_spanish():
    """Test Spanish summarization"""
    print("Testing Spanish summarization...")
    
    summarizer = DomainTunedSummarizer()
    
    text = """
    Hoy quiero hablar sobre el cambio climático.
    El cambio climático es real y está sucediendo ahora.
    Debemos actuar inmediatamente para reducir emisiones.
    Las energías renovables son la solución.
    Pero necesitamos voluntad política.
    """
    
    summary = summarizer.summarize(text, language='es', num_sentences=2)
    
    assert len(summary) > 0
    assert len(summary) < len(text)
    
    print(f"✅ Spanish summarization works")
    print(f"   Resumen: {summary}\n")


def test_edge_cases():
    """Test edge cases"""
    print("Testing edge cases...")
    
    summarizer = BERTSummarizer()
    
    # Very short text
    short_text = "This is short."
    summary = summarizer.summarize(short_text, num_sentences=3)
    assert summary == short_text
    print("✅ Handles short text correctly")
    
    # Text with exactly the requested number of sentences
    exact_text = "First sentence. Second sentence. Third sentence."
    summary = summarizer.summarize(exact_text, num_sentences=3)
    assert len(summary) > 0
    print("✅ Handles exact match correctly")
    
    print()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("RUNNING SMOKE TESTS")
    print("="*60 + "\n")
    
    try:
        test_basic_summarization()
        test_domain_tuned()
        test_spanish()
        test_edge_cases()
        
        print("="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        print("\nYour installation is working correctly!")
        print("Try the full demo with: python demo.py")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("\nCommon issues:")
        print("1. Missing spaCy models: python -m spacy download en_core_web_lg")
        print("2. Missing dependencies: pip install -r requirements.txt")
        raise
