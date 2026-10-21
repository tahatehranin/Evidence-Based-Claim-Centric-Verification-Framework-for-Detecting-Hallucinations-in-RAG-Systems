import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from local_verifier import LocalVerifier

def test_local_verifier():
    verifier = LocalVerifier()
    claim = "The dog is 5 years old"
    context = "Spot is a dog. The dog is 5 years old. Spot is a good boy."

    result = verifier.verify(claim, context)
    assert result.overall_support == True
    assert result.confidence > 0.5
