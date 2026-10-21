"""Main verifier module - orchestrates the entire hallucination detection pipeline."""
from typing import Optional
from dataclasses import dataclass
from claim_decomposer import ClaimDecomposer, Claim
from local_verifier import LocalVerifier, LocalVerificationResult
from global_verifier import GlobalVerifier, GlobalVerificationResult
from answer_aggregator import AnswerAggregator, AnswerAggregationResult


@dataclass
class HallucinationResult:
    """Complete hallucination detection result."""
    answer_label: str
    answer_confidence: float
    claim_results: list
    unfaithful_claims: list
    evidence_summary: str


class RAGHallucinationVerifier:
    """Evidence-based claim-centric hallucination detection for RAG systems.
    
    Uses lightweight lexical/similarity scoring instead of heavy ML models.
    """
    
    def __init__(
        self,
        context: str,
        answer: str,
        enable_web_verification: bool = False,
        enable_counterfactual_stability: bool = True,
        allow_partial_faithfulness: bool = False
    ):
        """
        Initialize verifier.
        
        Args:
            context: Retrieved context from RAG system
            answer: Generated answer text
            enable_web_verification: Whether to verify claims against web
            enable_counterfactual_stability: Whether to perform counterfactual checks
            allow_partial_faithfulness: If True, allow partial faithfulness
        """
        self.context = context
        self.answer = answer
        self.allow_partial_faithfulness = allow_partial_faithfulness
        
        self.decomposer = ClaimDecomposer()
        self.local_verifier = LocalVerifier()
        self.global_verifier = GlobalVerifier(
            counterfactual_stability=enable_counterfactual_stability
        )
        self.aggregator = AnswerAggregator()
        
        self.enable_web_verification = enable_web_verification
    
    def verify(self) -> HallucinationResult:
        """
        Run complete hallucination detection pipeline.
        
        Returns:
            HallucinationResult with answer-level label and detailed claim results
        """
        claims = self.decomposer.decompose(self.answer, self.context)
        
        claim_results = []
        for claim in claims:
            global_result = self.global_verifier.verify(claim.text, self.context)
            claim_results.append(global_result)
        
        aggregation_result = self.aggregator.aggregate(
            claim_results,
            allow_partial_faithfulness=self.allow_partial_faithfulness
        )
        
        final_claim_results = [
            {
                'claim_text': r.claim_text,
                'final_label': r.final_label,
                'final_confidence': r.final_confidence,
                'evidence_text': r.evidence_text,
                'reliability_score': r.reliability_score
            }
            for r in claim_results
        ]
        
        return HallucinationResult(
            answer_label=aggregation_result.answer_label,
            answer_confidence=aggregation_result.answer_confidence,
            claim_results=final_claim_results,
            unfaithful_claims=aggregation_result.unfaithful_claims,
            evidence_summary=aggregation_result.evidence_summary
        )
    
    def print_report(self):
        """Print readable verification report."""
        result = self.verify()
        
        print("=" * 70)
        print("EVIDENCE-BASED HALLUCINATION DETECTION REPORT")
        print("=" * 70)
        print(f"\nAnswer Label: {result.answer_label.upper()}")
        print(f"Confidence: {result.answer_confidence:.3f}")
        print(f"Evidence Summary: {result.evidence_summary}")
        
        print(f"\n{'=' * 70}")
        print("CLAIM DETAILS")
        print(f"{'=' * 70}")
        
        for i, claim in enumerate(result.claim_results, 1):
            print(f"\n[{i}] {claim['claim_text']}")
            print(f"    Label: {claim['final_label'].upper()}")
            print(f"    Confidence: {claim['final_confidence']:.3f}")
            print(f"    Reliability: {claim['reliability_score']:.3f}")
            print(f"    Evidence: {claim['evidence_text'][:200]}...")
        
        print(f"\n{'=' * 70}")
        print("UNFAITHFUL CLAIMS")
        print(f"{'=' * 70}")
        
        if result.unfaithful_claims:
            for i, claim in enumerate(result.unfaithful_claims, 1):
                print(f"{i}. {claim}")
        else:
            print("No unfaithful claims found.")
        
        print(f"\n{'=' * 70}")
        print("END OF REPORT")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    context = """
    Google Cloud Platform offers various services including Cloud Storage, 
    Compute Engine, and Kubernetes. The global computing power is about 
    18 zettabytes. Google processes over 3.5 billion searches per day.
    """
    
    answer = """
    Google's cloud platform includes Kubernetes and manages 18 zettabytes of 
    global computing power. They handle over 3.5 billion daily searches 
    which shows their massive scale.
    """
    
    verifier = RAGHallucinationVerifier(
        context=context,
        answer=answer,
        enable_web_verification=False
    )
    verifier.print_report()
