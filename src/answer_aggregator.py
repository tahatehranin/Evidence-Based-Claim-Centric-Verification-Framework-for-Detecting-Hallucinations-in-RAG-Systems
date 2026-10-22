"""Answer-level aggregation - combine claim results into answer label based on methodology."""
from configs.config import default_config
from typing import List, Dict
from dataclasses import dataclass
from global_verifier import GlobalVerificationResult
from constants import LABEL_S, LABEL_C, LABEL_N, DEFAULT_RHO


@dataclass
class ClaimResult:
    """Detailed result for a single claim."""
    claim_text: str
    final_label: str
    final_confidence: float
    evidence_text: str
    reliability_score: float
    status: str  # 'kept', 'deleted', 'unverifiable'


@dataclass
class AnswerAggregationResult:
    """Final answer-level verification result."""
    answer_label: str  # 'S', 'C', 'N', 'abstain'
    uncertainty_score: float
    claims: List[ClaimResult]
    unfaithful_claims: List[str]
    evidence_summary: str


class AnswerAggregator:
    """Aggregates claim-level results into answer-level verification.
    
    Implements formula (5):
    u(y) = |{ a ∈ A : status(a) ∈ {unverifiable, deleted} }| / n
    If u(y) >= rho, the system abstains.
    """

    def __init__(self, rho: float = None):
        self.rho = rho
        if self.rho is None:
            self.rho = default_config.rho
    
    def aggregate(
        self,
        claims_results: List[GlobalVerificationResult]
    ) -> AnswerAggregationResult:
        """Aggregate multiple claim results into answer label."""
        n = len(claims_results)
        if n == 0:
            return AnswerAggregationResult("abstain", 1.0, [], [], "No claims found")

        unfaithful_claims = []
        claim_results = []

        unverifiable_deleted_count = 0

        # In reality, deleted and unverifiable status comes from the execution policy
        # If label is C, policy might delete it or it's unfaithful.
        # If label is N and T_max retries fail, it becomes unverifiable.
        
        for result in claims_results:
            status = 'kept'
            if result.final_label == LABEL_N:
                status = 'unverifiable'
                unverifiable_deleted_count += 1
                unfaithful_claims.append(result.claim_text)
            elif result.final_label == LABEL_C:
                status = 'deleted'
                unverifiable_deleted_count += 1
                unfaithful_claims.append(result.claim_text)
            
            claim_results.append(ClaimResult(
                claim_text=result.claim_text,
                final_label=result.final_label,
                final_confidence=result.final_confidence,
                evidence_text=result.evidence_text,
                reliability_score=result.reliability_score,
                status=status
            ))

        u_y = unverifiable_deleted_count / n
        
        if u_y >= self.rho:
            answer_label = "abstain"
        else:
            if any(r.final_label == LABEL_C for r in claims_results):
                answer_label = LABEL_C
            elif any(r.final_label == LABEL_N for r in claims_results):
                answer_label = LABEL_N
            else:
                answer_label = LABEL_S

        evidence_summary = self._generate_evidence_summary(claim_results)
        
        return AnswerAggregationResult(
            answer_label=answer_label,
            uncertainty_score=u_y,
            claims=claim_results,
            unfaithful_claims=unfaithful_claims,
            evidence_summary=evidence_summary
        )
    
    def _generate_evidence_summary(self, claim_results: List[ClaimResult]) -> str:
        """Generate evidence summary string."""
        if not claim_results:
            return "No evidence available."
        
        supported = [r for r in claim_results if r.final_label == LABEL_S]
        contradicted = [r for r in claim_results if r.final_label == LABEL_C]
        baseless = [r for r in claim_results if r.final_label == LABEL_N]
        
        summary_parts = []
        if supported:
            summary_parts.append(f"Supported: {len(supported)} claim(s)")
        if contradicted:
            summary_parts.append(f"C: {len(contradicted)} claim(s)")
        if baseless:
            summary_parts.append(f"Unverifiable: {len(baseless)} claim(s)")

        return "; ".join(summary_parts)
