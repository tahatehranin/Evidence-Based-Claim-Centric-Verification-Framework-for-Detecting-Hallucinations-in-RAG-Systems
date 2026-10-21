"""Claim decomposition module - splits RAG answers into atomic verifiable claims."""
from typing import List, Dict, Any
from dataclasses import dataclass
import re


@dataclass
class Claim:
    """Atomic claim extracted from an answer."""
    text: str
    qualifiers: Dict[str, str]  # {'quantity': '5', 'temporal': '2024', 'negated': 'false'}
    span_indices: tuple  # (start, end) in original answer
    category: str = None


class ClaimDecomposer:
    """Decomposes RAG answers into evidence-grounded claims."""
    
    def __init__(self, llm_client=None, threshold=0.3):
        self.llm_client = llm_client
        self.threshold = threshold
    
    def decompose(self, answer: str, context: str) -> List[Claim]:
        """Split answer into verifiable claims."""
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        
        claims = []
        for idx, sent in enumerate(sentences):
            if not sent.strip():
                continue
            
            atomic_claims = self._decompose_sentence(sent, idx * 100)
            
            for atomic in atomic_claims:
                text = atomic['text']
                qualifiers = self._extract_qualifiers(text)
                claims.append(Claim(
                    text=text,
                    qualifiers=qualifiers,
                    span_indices=(atomic['start'], atomic['end']),
                    category=atomic.get('category', 'general')
                ))
        
        return claims
    
    def _decompose_sentence(self, sentence: str, offset: int) -> List[Dict]:
        """Decompose a sentence into atomic claims with spans."""
        conjunctions = [' and ', ' or ', ' but ', ' also ']
        parts = [sentence]
        
        for conj in conjunctions:
            new_parts = []
            for part in parts:
                new_parts.extend(part.split(conj))
            parts = new_parts
        
        atomic_claims = []
        for part in parts:
            if len(part.strip()) < 5:
                continue
            
            atomic_claims.append({
                'text': part.strip(),
                'start': offset,
                'end': offset + len(part.strip()),
                'category': self._classify_category(part.strip())
            })
        
        return atomic_claims
    
    def _extract_qualifiers(self, claim_text: str) -> Dict[str, str]:
        """Extract temporal, quantity, negation qualifiers."""
        qualifiers = {
            'negated': 'false',
            'quantity': '1',
            'temporal': None
        }
        
        if re.search(r'\bnot\s+(?:a\s+)?[\w-]+', claim_text, re.IGNORECASE):
            qualifiers['negated'] = 'true'
        
        qty_match = re.search(r'(\d+(?:\.\d+)?)\s+times?', claim_text)
        if qty_match:
            qualifiers['quantity'] = qty_match.group(1)
        
        temporal_patterns = [
            r'\b(in|on|at|during|before|after|from|to)\s+(202\d|20\d\d)',
            r'\b(last|this|next)\s+(year|month|day|week)',
            r'\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})'
        ]
        
        for pattern in temporal_patterns:
            match = re.search(pattern, claim_text, re.IGNORECASE)
            if match:
                qualifiers['temporal'] = match.group(0)
                break
        
        return qualifiers
    
    def _classify_category(self, text: str) -> str:
        """Classify claim category (entity, quantity, temporal, relation)."""
        if re.search(r'\b(\w+):(\w+)\s+is\s+', text):
            return 'relation'
        elif re.search(r'\b\d+\s*(?:million|billion|thousand)', text):
            return 'quantity'
        elif re.search(r'\b(in|on|at|during|before|after)\s+', text):
            return 'temporal'
        elif re.search(r'\b(\w+)\s+(is|are|was|were)\s+', text):
            return 'entity'
        return 'general'
