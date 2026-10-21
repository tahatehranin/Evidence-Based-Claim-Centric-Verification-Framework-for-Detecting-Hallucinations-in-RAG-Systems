"""Local verification module - checks claims against overlapping context chunks."""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re


@dataclass
class ChunkAssessment:
    """Assessment of a claim against a single context chunk."""
    chunk_index: int
    chunk_text: str
    supported: bool
    similarity: float
    evidence_spans: List[tuple]
    confidence: float


@dataclass
class LocalVerificationResult:
    """Result of local verification for a claim."""
    claim_text: str
    supported_chunks: List[ChunkAssessment]
    overall_support: bool
    avg_similarity: float
    evidence_text: str
    confidence: float


class LocalVerifier:
    """Verifies claims against context using local chunk-based verification.
    
    Uses lightweight lexical/similarity scoring instead of heavy ML models.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.30,
        overlap: int = 50
    ):
        self.similarity_threshold = similarity_threshold
        self.overlap = overlap
    
    def verify(self, claim_text: str, context: str) -> LocalVerificationResult:
        """Verify a single claim against context chunks."""
        chunks = self._create_chunks(context, self.overlap)
        
        chunk_assessments = []
        for idx, chunk in enumerate(chunks):
            assessment = self._assess_against_chunk(claim_text, chunk, idx)
            chunk_assessments.append(assessment)
        
        supported = any(a.supported for a in chunk_assessments)
        avg_similarity = sum(a.similarity for a in chunk_assessments) / max(len(chunk_assessments), 1)
        
        evidence_spans = []
        evidence_text = ""
        for a in chunk_assessments:
            if a.supported and a.evidence_spans:
                evidence_spans.extend(a.evidence_spans)
                evidence_text += " ".join(a.chunk_text[i:j] for i, j in a.evidence_spans)
        
        confidence = self._calculate_confidence(supported, avg_similarity, chunk_assessments)
        
        return LocalVerificationResult(
            claim_text=claim_text,
            supported_chunks=chunk_assessments,
            overall_support=supported,
            avg_similarity=avg_similarity,
            evidence_text=evidence_text,
            confidence=confidence
        )
    
    def _create_chunks(self, text: str, overlap: int) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        step = max(512 - overlap, 100)
        for i in range(0, len(text), step):
            chunk = text[i:i+512].strip()
            if chunk:
                chunks.append(chunk)
        return chunks
    
    def _assess_against_chunk(
        self,
        claim: str,
        chunk: str,
        chunk_index: int
    ) -> ChunkAssessment:
        """Assess a claim against a single chunk using lexical similarity."""
        similarity = self._compute_similarity(claim, chunk)
        
        supported = similarity >= self.similarity_threshold
        
        evidence_spans = []
        if supported:
            evidence_spans = self._extract_spans(claim, chunk, chunk_index)
        
        confidence = self._calculate_chunk_confidence(similarity)
        
        return ChunkAssessment(
            chunk_index=chunk_index,
            chunk_text=chunk,
            supported=supported,
            similarity=similarity,
            evidence_spans=evidence_spans,
            confidence=confidence
        )
    
    def _compute_similarity(self, claim: str, chunk: str) -> float:
        """Compute lexical similarity between claim and chunk.
        
        Uses token overlap ratio (coverage of claim words in chunk).
        """
        claim_words = set(self._normalize(claim).split())
        chunk_words = set(self._normalize(chunk).split())
        
        if not claim_words or not chunk_words:
            return 0.0
        
        # Token overlap: what fraction of claim words appear in chunk
        overlap = claim_words & chunk_words
        token_ratio = len(overlap) / len(claim_words)
        
        # Also check n-gram similarity for phrase matching
        claim_ngrams = self._get_ngrams(claim, n=3)
        chunk_ngrams = self._get_ngrams(chunk, n=3)
        
        if not claim_ngrams:
            return token_ratio
        
        ngram_overlap = claim_ngrams & chunk_ngrams
        ngram_ratio = len(ngram_overlap) / len(claim_ngrams)
        
        # Weighted combination
        return 0.4 * token_ratio + 0.6 * ngram_ratio
    
    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _get_ngrams(self, text: str, n: int = 3) -> set:
        """Get character n-grams from text."""
        text = self._normalize(text)
        if len(text) < n:
            return {text}
        return {text[i:i+n] for i in range(len(text) - n + 1)}
    
    def _extract_spans(self, claim: str, chunk: str, chunk_index: int) -> List[tuple]:
        """Extract evidence spans from chunk."""
        spans = []
        claim_words = self._normalize(claim).split()
        chunk_words = self._normalize(chunk).split()
        
        if not claim_words:
            return spans
        
        # Find longest matching substring
        best_match = ""
        best_start = 0
        for i in range(len(chunk_words)):
            for j in range(i + min(5, len(chunk_words) - i), len(chunk_words) + 1):
                candidate = " ".join(chunk_words[i:j])
                if len(candidate) > len(best_match):
                    if any(w in candidate for w in claim_words[:3]):
                        best_match = candidate
                        best_start = i
        
        if best_match:
            start_idx = chunk_index * 400 + best_start * 4
            end_idx = start_idx + len(best_match) * 4
            spans.append((start_idx, end_idx))
        
        return spans
    
    def _calculate_confidence(
        self,
        supported: bool,
        avg_similarity: float,
        assessments: List[ChunkAssessment]
    ) -> float:
        """Calculate overall confidence score."""
        if not supported:
            return 0.1
        
        confidence = 0.5 * avg_similarity + 0.3 * min(a.confidence for a in assessments) + 0.2
        return min(confidence, 0.95)
    
    def _calculate_chunk_confidence(self, similarity: float) -> float:
        """Calculate confidence for a single chunk assessment."""
        if similarity >= 0.6:
            return 0.9
        elif similarity >= 0.5:
            return 0.7
        elif similarity >= 0.35:
            return 0.5
        else:
            return 0.3
