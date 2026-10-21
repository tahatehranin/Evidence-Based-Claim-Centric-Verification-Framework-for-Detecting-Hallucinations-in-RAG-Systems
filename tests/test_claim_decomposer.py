import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from claim_decomposer import ClaimDecomposer

def test_claim_decomposer():
    decomposer = ClaimDecomposer()
    text = "The quick brown fox jumps over the lazy dog. The dog is named Spot."
    claims = decomposer.decompose(text, "")

    assert len(claims) >= 2
    assert any("fox" in c.text for c in claims)
    assert any("Spot" in c.text for c in claims)
