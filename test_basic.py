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


def test_quality_regression():
    """Test that quality doesn't degrade below acceptable threshold"""
    print("Testing quality regression...")
    
    summarizer = DomainTunedSummarizer()
    
    # This is a known-good example from development
    # In a real project, you'd load this from a fixtures file
    gold_text = """
    Climate change represents one of the most significant challenges facing humanity today.
    Rising global temperatures are causing widespread environmental disruption.
    Scientists have conclusively demonstrated that human activities, particularly fossil fuel combustion, are the primary drivers.
    The consequences include more frequent extreme weather events, rising sea levels, and ecosystem collapse.
    Immediate action is required to reduce greenhouse gas emissions and transition to renewable energy sources.
    Political and economic barriers remain substantial obstacles to meaningful climate action.
    However, technological solutions like solar and wind power are becoming increasingly cost-effective.
    International cooperation through agreements like the Paris Accord provides a framework for collective response.
    """
    
    summary = summarizer.summarize(gold_text, language='en', num_sentences=3, title="Climate Change Crisis")
    
    # Check that key concepts are preserved
    key_terms = ['climate', 'emissions', 'renewable', 'action']
    terms_found = sum(1 for term in key_terms if term in summary.lower())
    
    assert terms_found >= 2, f"Summary should contain at least 2 key terms, found {terms_found}"
    assert len(summary.split()) > 30, "Summary should be substantive (>30 words)"
    assert len(summary.split()) < 200, "Summary should be concise (<200 words)"
    
    print(f"✅ Quality regression check passed")
    print(f"   Key terms preserved: {terms_found}/{len(key_terms)}")
    print(f"   Summary length: {len(summary.split())} words\n")


def test_deterministic_output():
    """Test that the same input produces the same output"""
    print("Testing deterministic output...")
    
    summarizer = BERTSummarizer()
    
    text = """
    Artificial intelligence is transforming every industry.
    Machine learning algorithms can now perform tasks that once required human intelligence.
    From medical diagnosis to autonomous vehicles, AI applications are proliferating.
    However, concerns about bias, privacy, and job displacement remain.
    Responsible AI development requires careful consideration of ethical implications.
    """
    
    # Generate summary twice
    summary1 = summarizer.summarize(text, language='en', num_sentences=2)
    summary2 = summarizer.summarize(text, language='en', num_sentences=2)
    
    assert summary1 == summary2, "Same input should produce identical output"
    
    print(f"✅ Deterministic output verified")
    print(f"   Both runs produced: {summary1[:80]}...\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("RUNNING SMOKE TESTS")
    print("="*60 + "\n")
    
    try:
        test_basic_summarization()
        test_domain_tuned()
        test_spanish()
        test_edge_cases()
        test_quality_regression()
        test_deterministic_output()
        
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
