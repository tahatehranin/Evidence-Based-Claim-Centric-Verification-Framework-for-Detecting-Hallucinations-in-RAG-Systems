"""Answer-level aggregation - combine claim results into answer label."""
from typing import List
from dataclasses import dataclass
from global_verifier import GlobalVerificationResult


@dataclass
class ClaimResult:
    """Detailed result for a single claim."""
    claim_text: str
    final_label: str
    final_confidence: float
    evidence_text: str
    reliability_score: float


@dataclass
class AnswerAggregationResult:
    """Final answer-level verification result."""
    answer_label: str  # 'entailed', 'contradicted', 'baseless'
    answer_confidence: float
    claims: List[ClaimResult]
    unfaithful_claims: List[str]
    evidence_summary: str


class AnswerAggregator:
    """Aggregates claim-level results into answer-level verification."""
    
    def __init__(self):
        pass
    
    def aggregate(
        self,
        claims_results: List[GlobalVerificationResult],
        allow_partial_faithfulness: bool = False
    ) -> AnswerAggregationResult:
        """Aggregate multiple claim results into answer label."""
        # Count label frequencies
        label_counts = {
            'entailed': 0,
            'contradicted': 0,
            'baseless': 0
        }
        
        unfaithful_claims = []
        
        for result in claims_results:
            label_counts[result.final_label] += 1
            
            # Track unfaithful claims
            if result.final_label != 'entailed':
                unfaithful_claims.append(result.claim_text)
        
        # Determine answer label
        if allow_partial_faithfulness:
            # Soft aggregation: weighted by confidence
            weighted_score = sum(
                result.final_confidence * (1 if result.final_label == 'entailed' else 0)
                for result in claims_results
            ) / len(claims_results)
            
            if weighted_score >= 0.5:
                answer_label = 'entailed'
            else:
                answer_label = 'contradicted' if label_counts['contradicted'] > 0 else 'baseless'
        else:
            # Hard AND-join: single unfaithful claim flags entire answer
            if label_counts['contradicted'] > 0:
                answer_label = 'contradicted'
            elif label_counts['baseless'] > 0:
                answer_label = 'baseless'
            else:
                answer_label = 'entailed'
        
        # Calculate answer confidence
        answer_confidence = self._calculate_answer_confidence(
            label_counts,
            claims_results,
            answer_label
        )
        
        # Create claim result objects
        claim_results = [
            ClaimResult(
                claim_text=result.claim_text,
                final_label=result.final_label,
                final_confidence=result.final_confidence,
                evidence_text=result.evidence_text,
                reliability_score=result.reliability_score
            )
            for result in claims_results
        ]
        
        # Generate evidence summary
        evidence_summary = self._generate_evidence_summary(claim_results)
        
        return AnswerAggregationResult(
            answer_label=answer_label,
            answer_confidence=answer_confidence,
            claims=claim_results,
            unfaithful_claims=unfaithful_claims,
            evidence_summary=evidence_summary
        )
    
    def _calculate_answer_confidence(
        self,
        label_counts: dict,
        claims_results: List[GlobalVerificationResult],
        answer_label: str
    ) -> float:
        """Calculate answer-level confidence score."""
        # Base on majority confidence
        entailed_confidences = [
            r.final_confidence for r in claims_results
            if r.final_label == 'entailed'
        ]
        
        if answer_label == 'entailed' and entailed_confidences:
            return max(entailed_confidences)
        elif answer_label == 'contradicted':
            max_contradicted = max(
                r.final_confidence for r in claims_results
                if r.final_label == 'contradicted'
            ) or 0.1
            return max_contradicted * 0.8
        else:  # baseless
            max_baseless = max(
                r.final_confidence for r in claims_results
                if r.final_label == 'baseless'
            ) or 0.1
            return max_baseless * 0.6
    
    def _generate_evidence_summary(self, claim_results: List[ClaimResult]) -> str:
        """Generate evidence summary string."""
        if not claim_results:
            return "No evidence available."
        
        entailed = [r for r in claim_results if r.final_label == 'entailed']
        baseless = [r for r in claim_results if r.final_label == 'baseless']
        
        summary_parts = []
        
        if entailed:
            summary_parts.append(f"Supported: {len(entailed)} claim(s)")
        
        if baseless:
            summary_parts.append(f"Unsupported: {len(baseless)} claim(s)")
        
        return "; ".join(summary_parts)
