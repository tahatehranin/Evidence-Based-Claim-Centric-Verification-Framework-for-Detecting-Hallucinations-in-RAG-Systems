# Evidence-Based Claim-Centric Verification Framework
# Hallucination Detection for RAG Systems

## Overview

This framework implements evidence-based hallucination detection using a claim-centric approach with hierarchical verification.

### Key Components

- **Claim Decomposer**: Splits answers into atomic, verifiable claims
- **Local Verifier**: Checks claims against overlapping context chunks
- **Global Verifier**: Performs hierarchical verification with NLI and counterfactual stability
- **Answer Aggregator**: Combines claim results into answer-level labels

## Installation

```bash
# Clone repository
git clone https://github.com/tahatehran/Evidence-Based-Claim-Centric-Verification-Framework-for-Detecting-Hallucinations-in-RAG-Systems.git
cd Evidence-Based-Claim-Centric-Verification-Framework-for-Detecting-Hallucinations-in-RAG-Systems

# Setup
bash setup.sh
```

## Quick Start

```python
from verifier import RAGHallucinationVerifier

# RAG system context and answer
context = """
Retrieved context passages about the topic...
"""

answer = """
Generated answer text...
"""

# Initialize verifier
verifier = RAGHallucinationVerifier(
    context=context,
    answer=answer,
    enable_web_verification=False
)

# Run verification
result = verifier.verify()

# Print report
verifier.print_report()
```

## Verification Labels

- **Entailed**: Claim supported by evidence (confidence based on similarity + NLI)
- **Contradicted**: Claim conflicts with retrieved context
- **Baseless**: No relevant evidence found

## Performance

- **Answer-level F1**: 0.776 (RAGTruth++)
- **Span-level F1**: 47.5% (RAGTruth-Enhance)
- Uses DeBERTa-v3-large NLI model with hierarchical verification

## Configuration

Adjust thresholds and behavior:

```python
verifier = RAGHallucinationVerifier(
    context=context,
    answer=answer,
    enable_counterfactual_stability=True,  # Counterfactual checks
    allow_partial_faithfulness=False       # AND-join (default)
)
```

## Evaluation

Run examples:

```bash
source venv/bin/activate
python -m examples.demo
```
