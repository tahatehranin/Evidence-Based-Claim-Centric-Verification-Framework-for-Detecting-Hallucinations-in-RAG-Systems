"""Main verifier module - orchestrates the entire hallucination detection pipeline."""
from typing import Optional
from dataclasses import dataclass
from claim_decomposer import ClaimDecomposer, Claim
from local_verifier import LocalVerifier, LocalVerificationResult
from global_verifier import GlobalVerifier, GlobalVerificationResult
from answer_aggregator import AnswerAggregator, AnswerAggregationResult
from constants import DEFAULT_RHO, DEFAULT_TAU


@dataclass
class HallucinationResult:
    """Complete hallucination detection result."""
    answer_label: str
    uncertainty_score: float
    claim_results: list
    unfaithful_claims: list
    evidence_summary: str


class RAGHallucinationVerifier:
    """Evidence-based claim-centric hallucination detection for RAG systems."""
    
    def __init__(
        self,
        context: str,
        answer: str,
        rho: float = DEFAULT_RHO,
        tau: float = DEFAULT_TAU
    ):
        self.context = context
        self.answer = answer
        self.rho = rho
        self.tau = tau
        
        self.decomposer = ClaimDecomposer()
        self.global_verifier = GlobalVerifier(tau=self.tau)
        self.aggregator = AnswerAggregator(rho=self.rho)
    
    def verify(self) -> HallucinationResult:
        """Run complete hallucination detection pipeline."""
        claims = self.decomposer.decompose(self.answer, self.context)
        
        claim_results = []
        for claim in claims:
            global_result = self.global_verifier.verify(claim.text, self.context)
            claim_results.append(global_result)
        
        aggregation_result = self.aggregator.aggregate(claim_results)
        
        final_claim_results = [
            {
                'claim_text': r.claim_text,
                'final_label': r.final_label,
                'final_confidence': r.final_confidence,
                'evidence_text': r.evidence_text,
                'reliability_score': r.reliability_score,
                'status': r.status
            }
            for r in aggregation_result.claims
        ]
        
        return HallucinationResult(
            answer_label=aggregation_result.answer_label,
            uncertainty_score=aggregation_result.uncertainty_score,
            claim_results=final_claim_results,
            unfaithful_claims=aggregation_result.unfaithful_claims,
            evidence_summary=aggregation_result.evidence_summary
        )
