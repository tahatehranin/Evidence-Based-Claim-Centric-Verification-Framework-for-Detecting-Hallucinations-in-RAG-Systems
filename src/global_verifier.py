"""Global verification module - final hierarchical verification implementing Eq 3 & 4."""
from typing import List, Dict, Any
from dataclasses import dataclass
from local_verifier import LocalVerificationResult, LocalVerifier
from constants import LABEL_S, LABEL_C, LABEL_N, DEFAULT_TAU


@dataclass
class ClaimLabel:
    """Claim verification label and metadata."""
    label: str  # 'S', 'C', 'N'
    confidence: float
    evidence: str
    nli_result: Dict[str, float]


@dataclass
class GlobalVerificationResult:
    """Final verification result for a claim."""
    claim_text: str
    labels: List[ClaimLabel]
    final_label: str
    final_confidence: float
    evidence_text: str
    reliability_score: float


class GlobalVerifier:
    """Performs full-context verification and implements threshold logic.
    
    Implements formula (4):
    c(a) = (1/s) Σ_{j=1..s} 1[ ℓ_j(a) = ℓ_maj(a) ]
    If c(a) < tau, label becomes N.
    """
    
    def __init__(
        self,
        tau: float = DEFAULT_TAU,
        nli_threshold: float = 0.45,
        counterfactual_stability: bool = True
    ):
        self.tau = tau
        self.nli_threshold = nli_threshold
        self.counterfactual_stability = counterfactual_stability
        self._local_verifier = LocalVerifier()
    
    def verify(self, claim_text: str, context: str) -> GlobalVerificationResult:
        """Verify claim using NLI models, voting, and thresholds."""
        local_result = self._local_verifier.verify(claim_text, context)
        
        # Simulate s=5 votes for confidence c(a)
        nli_labels_votes = [self._nli_verification(claim_text, context) for _ in range(5)]
        
        # Determine majority label ℓ_maj(a)
        vote_counts = {LABEL_S: 0, LABEL_C: 0, LABEL_N: 0}
        for vote in nli_labels_votes:
            if vote:
                vote_counts[vote[0].label] += 1

        # Calculate c(a) as ratio of majority
        majority_label = max(vote_counts, key=vote_counts.get)
        c_a = vote_counts[majority_label] / 5.0

        # Apply threshold tau
        if c_a < self.tau:
            final_label = LABEL_N
        else:
            final_label = majority_label

        # Get the representative nli_label
        rep_nli_label = next((v[0] for v in nli_labels_votes if v and v[0].label == final_label), None)

        if not rep_nli_label:
            rep_nli_label = ClaimLabel(label=LABEL_N, confidence=0.0, evidence="", nli_result={})

        reliability = self._calculate_reliability(local_result, rep_nli_label)
        
        return GlobalVerificationResult(
            claim_text=claim_text,
            labels=[rep_nli_label],
            final_label=final_label,
            final_confidence=c_a,
            evidence_text=rep_nli_label.evidence,
            reliability_score=reliability
        )
    
    def _nli_verification(self, claim_text: str, context: str) -> List[ClaimLabel]:
        """Verify claim using simulated NLI model with contradiction detection."""
        relevant_context = self._extract_relevant_context(claim_text, context)
        similarity = self._local_verifier._compute_similarity(claim_text, relevant_context)
        contradiction_score = self._detect_contradiction(claim_text, relevant_context)
        
        if contradiction_score > 0.3:
            label = LABEL_C
            confidence = contradiction_score
        elif similarity >= self.nli_threshold:
            label = LABEL_S
            confidence = similarity
        elif similarity >= 0.25:
            # We map 'neutral' NLI to N (Not Enough Evidence)
            label = LABEL_N
            confidence = similarity * 0.7
        else:
            label = LABEL_C
            confidence = 1.0 - similarity
        
        nli_result = {
            LABEL_S: similarity if label == LABEL_S else 0.1,
            LABEL_N: similarity if label == LABEL_N else 0.1,
            LABEL_C: confidence if label == LABEL_C else 0.1
        }
        
        return [
            ClaimLabel(
                label=label,
                confidence=confidence,
                evidence=relevant_context,
                nli_result=nli_result
            )
        ]
    
    def _detect_contradiction(self, claim: str, context: str) -> float:
        """Detect contradictions by comparing numeric and entity values."""
        import re
        claim_numbers = set(re.findall(r'\b\d+\b', claim))
        context_numbers = set(re.findall(r'\b\d+\b', context))
        
        if claim_numbers and context_numbers:
            claim_only = claim_numbers - context_numbers
            if claim_only:
                return min(len(claim_only) / max(len(claim_numbers), 1), 0.9)
        
        negation_patterns = [
            (r'\bnot\s+', r'\bis\s+'),
            (r'\bnever\s+', r'\balways\s+'),
            (r'\bless\s+than\s+', r'\bmore\s+than\s+'),
            (r'\bdoes not\s+', r'\bdoes\s+'),
        ]
        
        for neg_pat, pos_pat in negation_patterns:
            if re.search(neg_pat, claim, re.IGNORECASE) and re.search(pos_pat, context, re.IGNORECASE):
                return 0.8
        
        return 0.0
    
    def _extract_relevant_context(self, claim: str, context: str, max_length: int = 512) -> str:
        """Extract most relevant context spans for NLI."""
        sentences = context.split('. ')
        if not sentences:
            return context[:max_length]
        
        similarities = []
        for sent in sentences:
            if len(sent) < 10:
                continue
            sim = self._local_verifier._compute_similarity(claim, sent)
            similarities.append((sent, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s for s, sim in similarities[:3]]
        
        result = ". ".join(top_sentences)
        return result[:max_length] if result else context[:max_length]

    def _calculate_reliability(self, local_result, nli_label: ClaimLabel) -> float:
        """Calculate reliability score (0-1)."""
        local_score = local_result.avg_similarity if local_result.overall_support else 0.1
        nli_score = nli_label.confidence if nli_label else 0.1
        reliability = 0.6 * local_score + 0.4 * nli_score
        return min(max(reliability, 0.1), 0.95)
