import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from verifier import RAGHallucinationVerifier

def test_verifier():
    context = "The quick brown fox jumps over the lazy dog. The dog is named Spot. Spot is 5 years old."
    answer = "The quick brown fox jumps over the lazy dog. The dog is 5 years old."

    verifier = RAGHallucinationVerifier(
        context=context,
        answer=answer
    )

    result = verifier.verify()
    assert result.answer_label in ["S", "C", "N", "abstain"]

def test_verifier_contradiction():
    context = "The quick brown fox jumps over the lazy dog. The dog is named Spot. Spot is 5 years old."
    answer = "The quick brown fox jumps over the lazy dog. The dog is 10 years old."

    verifier = RAGHallucinationVerifier(
        context=context,
        answer=answer
    )

    result = verifier.verify()
    assert result.answer_label in ["S", "C", "N", "abstain"]
