# Evidence-Based Claim-Centric Verification Framework for Hallucination Detection in RAG Systems

## Overview

This framework implements evidence-based hallucination detection for RAG systems using a claim-centric approach with hierarchical verification.

## Key Features

- **Claim Decomposition**: Splits generated answers into atomic, verifiable claims
- **Hierarchical Verification**: Local + global verification pipeline
- **Evidence Grounding**: Returns explicit context-side evidence with each claim
- **Multi-Signal Scoring**: Combines lexical similarity + contradiction detection + counterfactual stability
- **Entity-Centric Alignment**: Identity, semantic, and consistency alignment dimensions

## Architecture

### Pipeline Stages

1. **Claim Decomposition** (`claim_decomposer.py`)
   - Segment answer into sentences
   - Decompose each sentence into atomic claims
   - Preserve qualifiers (negation, quantities, temporal markers)

2. **Local Verification** (`local_verifier.py`)
   - Split context into overlapping chunks
   - Screen each claim against chunks using lexical similarity
   - OR-join chunk-specific assessments

3. **Global Verification** (`global_verifier.py`)
   - Re-verify against full context
   - Detect contradictions (numeric mismatches, negation patterns)
   - Extract final evidence spans
   - Assign final claim labels

4. **Answer-Level Aggregation** (`answer_aggregator.py`)
   - AND-join: single unfaithful claim flags entire answer
   - Labels: Entailed, Contradicted, Baseless

## Verification Labels

| Label | Meaning | Confidence Formula |
|-------|---------|---------------------|
| Entailed | Claim supported by evidence | `0.5 × similarity + 0.3 × chunk + 0.2` |
| Contradicted | Claim conflicts with evidence | Contradiction detection + negation patterns |
| Baseless | No relevant evidence found | Low similarity + neutral NLI |

## Evidence Sources

### Internal Knowledge Base
- FAISS-indexed document chunks (configurable)
- Semantic similarity threshold: ~0.30
- NLI threshold for support: ~0.45

### Reliability-Tiered Web Verification
- **Tier 1**: Government, official orgs, reference sources (weight 1.0)
- **Tier 2**: Established news organizations (weight 0.6)
- **Tier 3**: Blogs, forums (weight 0.3)

## Installation

```bash
# Clone repository
git clone https://github.com/tahatehran/Evidence-Based-Claim-Centric-Verification-Framework-for-Detecting-Hallucinations-in-RAG-Systems.git
cd Evidence-Based-Claim-Centric-Verification-Framework-for-Detecting-Hallucinations-in-RAG-Systems

# Setup
bash setup.sh
```

## Usage

```python
from verifier import RAGHallucinationVerifier

# Initialize verifier
verifier = RAGHallucinationVerifier(
    context="Retrieved context passages...",
    answer="Generated response text...",
    enable_web_verification=False
)

# Verify
result = verifier.verify()

# Output
print(f"Answer label: {result.answer_label}")
print(f"Confidence: {result.answer_confidence:.3f}")
for claim in result.claim_results:
    print(f"  [{claim['label']}] {claim['text']}")
    print(f"    Evidence: {claim['evidence']}")
    print(f"    Confidence: {claim['confidence']:.3f}")
```

## Evaluation

Test cases cover:
- Clean entailment (all claims supported)
- Hallucination (unsupported claims)
- Contradiction (numeric mismatches)
- Negation contradictions
- Baseless claims (no evidence)

## Dependencies

- Python 3.13+
- No heavy ML models required (uses lightweight lexical/similarity scoring)
- Optional: sentence-transformers for enhanced similarity

## Configuration

```python
verifier = RAGHallucinationVerifier(
    context=context,
    answer=answer,
    enable_counterfactual_stability=True,  # Counterfactual checks
    allow_partial_faithfulness=False       # AND-join (default)
)
```
