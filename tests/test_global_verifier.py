import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from global_verifier import GlobalVerifier

def test_global_verifier():
    verifier = GlobalVerifier()
    claim = "The dog is 5 years old"
    context = "Spot is a dog. The dog is 5 years old."

    result = verifier.verify(claim, context)
    assert result.final_label in ["S", "N", "C"]
