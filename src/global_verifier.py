"""Global verification module - final hierarchical verification with NLI."""
from typing import List, Dict, Any
from dataclasses import dataclass
from local_verifier import LocalVerificationResult, LocalVerifier


@dataclass
class ClaimLabel:
    """Claim verification label and metadata."""
    label: str  # 'entailed', 'contradicted', 'baseless'
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
    """Hierarchical verification with NLI and counterfactual stability.
    
    Uses lightweight lexical/similarity scoring instead of heavy ML models.
    """
    
    def __init__(
        self,
        nli_threshold: float = 0.45,
        counterfactual_stability: bool = True
    ):
        self.nli_threshold = nli_threshold
        self.counterfactual_stability = counterfactual_stability
        self._local_verifier = LocalVerifier()
    
    def verify(self, claim_text: str, context: str) -> GlobalVerificationResult:
        """Verify claim using hierarchical verification."""
        local_result = self._local_verifier.verify(claim_text, context)
        nli_labels = self._nli_verification(claim_text, context)
        
        if self.counterfactual_stability:
            nli_labels = self._counterfactual_check(claim_text, context, nli_labels)
        
        final_label, final_confidence = self._determine_label(local_result, nli_labels)
        reliability_score = self._calculate_reliability(local_result, nli_labels)
        
        return GlobalVerificationResult(
            claim_text=claim_text,
            labels=nli_labels,
            final_label=final_label,
            final_confidence=final_confidence,
            evidence_text=local_result.evidence_text,
            reliability_score=reliability_score
        )
    
    def _nli_verification(self, claim_text: str, context: str) -> List[ClaimLabel]:
        """Verify claim using NLI model with contradiction detection."""
        relevant_context = self._extract_relevant_context(claim_text, context)
        similarity = self._local_verifier._compute_similarity(claim_text, relevant_context)
        contradiction_score = self._detect_contradiction(claim_text, relevant_context)
        
        if contradiction_score > 0.3:
            label = "contradicted"
            confidence = contradiction_score
        elif similarity >= self.nli_threshold:
            label = "entailed"
            confidence = similarity
        elif similarity >= 0.25:
            label = "neutral"
            confidence = similarity * 0.7
        else:
            label = "contradicted"
            confidence = 1.0 - similarity
        
        nli_result = {
            "entailed": similarity if label == "entailed" else 0.1,
            "neutral": similarity if label == "neutral" else 0.1,
            "contradicted": confidence if label == "contradicted" else 0.1
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
        
        # Extract numbers from both
        claim_numbers = set(re.findall(r'\b\d+\b', claim))
        context_numbers = set(re.findall(r'\b\d+\b', context))
        
        # If claim has numbers not in context, potential contradiction
        if claim_numbers and context_numbers:
            claim_only = claim_numbers - context_numbers
            if claim_only:
                return min(len(claim_only) / max(len(claim_numbers), 1), 0.9)
        
        # Check for negation patterns
        negation_patterns = [
            (r'\bnot\s+', r'\bis\s+'),
            (r'\bnever\s+', r'\balways\s+'),
            (r'\bless\s+than\s+', r'\bmore\s+than\s+'),
            (r'\bdoes not\s+', r'\bdoes\s+'),
            (r'\bdoes not support\b', r'\bsupports?\b'),
            (r'\bdoes not have\b', r'\bhas\b'),
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
    
    def _counterfactual_check(
        self,
        claim: str,
        context: str,
        nli_labels: List[ClaimLabel]
    ) -> List[ClaimLabel]:
        """Check counterfactual stability."""
        if not nli_labels:
            return nli_labels
        
        original = nli_labels[0]
        
        if original.label in ["contradicted", "baseless"]:
            counterfactual_claims = [
                f"{claim} is false",
                f"Is it true that {claim}?"
            ]
            
            for cf in counterfactual_claims:
                if cf.lower() == claim.lower():
                    continue
                cf_result = self._nli_verification(cf, context)
                if cf_result and cf_result[0].label == original.label:
                    original.confidence *= 0.9
        
        return nli_labels
    
    def _determine_label(
        self,
        local_result,
        nli_labels: List[ClaimLabel]
    ) -> tuple:
        """Determine final claim label."""
        if not local_result.overall_support:
            return "baseless", local_result.confidence * 0.7
        
        if not nli_labels:
            return "baseless", 0.4
        
        nli_label = nli_labels[0]
        
        if nli_label.label == "contradicted" and nli_label.confidence > self.nli_threshold:
            return "contradicted", nli_label.confidence * 0.8
        
        if nli_label.label == "entailed" and nli_label.confidence > self.nli_threshold:
            return "entailed", nli_label.confidence
        
        return "baseless", 0.4
    
    def _calculate_reliability(self, local_result, nli_labels: List[ClaimLabel]) -> float:
        """Calculate reliability score (0-1)."""
        local_score = local_result.avg_similarity if local_result.overall_support else 0.1
        nli_score = max(l.confidence for l in nli_labels) if nli_labels else 0.1
        
        reliability = 0.6 * local_score + 0.4 * nli_score
        return min(max(reliability, 0.1), 0.95)
