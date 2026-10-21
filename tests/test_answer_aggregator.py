import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from answer_aggregator import AnswerAggregator
from global_verifier import GlobalVerificationResult, ClaimLabel

def test_answer_aggregator():
    aggregator = AnswerAggregator()

    claims = [
        GlobalVerificationResult(
            claim_text="The dog is 5 years old",
            labels=[],
            final_label="S",
            final_confidence=0.9,
            evidence_text="The dog is 5 years old",
            reliability_score=0.9
        ),
        GlobalVerificationResult(
            claim_text="The dog is named Spot",
            labels=[],
            final_label="S",
            final_confidence=0.8,
            evidence_text="The dog is named Spot",
            reliability_score=0.8
        )
    ]

    result = aggregator.aggregate(claims)
    assert result.answer_label == "S"

def test_answer_aggregator_contradiction():
    aggregator = AnswerAggregator()

    claims = [
        GlobalVerificationResult(
            claim_text="The dog is 5 years old",
            labels=[],
            final_label="S",
            final_confidence=0.9,
            evidence_text="The dog is 5 years old",
            reliability_score=0.9
        ),
        GlobalVerificationResult(
            claim_text="The dog is named Spot",
            labels=[],
            final_label="C",
            final_confidence=0.8,
            evidence_text="The dog is named Max",
            reliability_score=0.8
        )
    ]

    result = aggregator.aggregate(claims)
    # 2 claims, 1 is C -> u(y) = 1/2 = 0.5. With default rho=0.3, this should "abstain"
    assert result.answer_label == "abstain"
    assert "The dog is named Spot" in result.unfaithful_claims
