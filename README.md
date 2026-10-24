# Evidence-Based Claim-Centric Verification Framework for Hallucination Detection in RAG Systems

![Logo Placeholder](https://img.shields.io/badge/RAG-Verification-blue.svg?style=for-the-badge&logo=appveyor)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)
![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python)

## Overview

This framework implements evidence-based hallucination detection for RAG systems using a claim-centric approach with hierarchical verification. The core methodology and formulas are based on the paper provided in `main_paper/مقاله_RAG_متدولوژی_دقیق-3.docx`.

## 🚀 Key Features

- **Claim Decomposition**: Splits generated answers into atomic, verifiable claims ($C = \{c_1, c_2, ..., c_n\}$)
- **Hierarchical Verification**: Local + global verification pipeline mapping claims to $\{S, C, N\}$
- **Evidence Grounding**: Returns explicit context-side evidence with each claim
- **Multi-Signal Scoring**: Combines lexical similarity + contradiction detection + counterfactual stability
- **Entity-Centric Alignment**: Identity, semantic, and consistency alignment dimensions

## 🏗️ Architecture

### Pipeline Stages

1. **Claim Decomposition** (`src/claim_decomposer.py`)
   - Segment answer into sentences
   - Decompose each sentence into atomic claims
   - Preserve qualifiers (negation, quantities, temporal markers)

2. **Local Verification** (`src/local_verifier.py`)
   - Split context into overlapping chunks
   - Screen each claim against chunks using lexical similarity
   - OR-join chunk-specific assessments

3. **Global Verification** (`src/global_verifier.py`)
   - Re-verify against full context
   - Detect contradictions (numeric mismatches, negation patterns)
   - Extract final evidence spans
   - Assign final claim labels

4. **Answer-Level Aggregation** (`src/answer_aggregator.py`)
   - AND-join: single unfaithful claim flags entire answer
   - Labels: Supported (S), Contradicted (C), Not Enough Evidence/Not Enough Evidence (N)

## 📊 Verification Labels

| Label | Meaning | Confidence Formula |
|-------|---------|---------------------|
| Supported (S) | Claim supported by evidence | `0.5 × similarity + 0.3 × chunk + 0.2` |
| Contradicted (C) | Claim conflicts with evidence | Contradiction detection + negation patterns |
| Not Enough Evidence (N) | No relevant evidence found | Low similarity + neutral NLI |

## 📚 Evidence Sources

### Internal Knowledge Base
- FAISS-indexed document chunks (configurable)
- Semantic similarity threshold: ~0.30
- NLI threshold for support: ~0.45

### Reliability-Tiered Web Verification
- **Tier 1**: Government, official orgs, reference sources (weight 1.0)
- **Tier 2**: Established news organizations (weight 0.6)
- **Tier 3**: Blogs, forums (weight 0.3)

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/tahatehran/Evidence-Based-Claim-Centric-Verification-Framework-for-Detecting-Hallucinations-in-RAG-Systems.git
cd Evidence-Based-Claim-Centric-Verification-Framework-for-Detecting-Hallucinations-in-RAG-Systems

# Setup
bash setup.sh
```

## 💻 Usage

```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from verifier import RAGHallucinationVerifier

# Initialize verifier
verifier = RAGHallucinationVerifier(
    context="Retrieved context passages...",
    answer="Generated response text...",
)

# Verify
result = verifier.verify()

# Output
print(f"Answer label: {result.answer_label}")
print(f"Uncertainty: {result.uncertainty_score:.3f}")
for claim in result.claim_results:
    print(f"  [{claim['final_label']}] {claim['claim_text']}")
    print(f"    Evidence: {claim['evidence_text']}")
    print(f"    Confidence: {claim['final_confidence']:.3f}")
```

You can run a demo example using:
```bash
python examples/demo.py
```

## 🧪 Testing and CI

GitHub Actions is set up to run tests on every push and pull request to the `main` branch.

To run tests locally:
```bash
pip install pytest pytest-cov
pytest tests/
```

## 📄 Evaluation

Test cases cover:
- Clean entailment (all claims supported)
- Hallucination (unsupported claims)
- Contradiction (numeric mismatches)
- Negation contradictions
- Baseless claims (no evidence)

## 📦 Dependencies

- Python 3.10+
- No heavy ML models required (uses lightweight lexical/similarity scoring)
- Optional: sentence-transformers for enhanced similarity

## ⚙️ Configuration

```python
verifier = RAGHallucinationVerifier(
    context=context,
    answer=answer,
    rho=0.3,
    tau=0.6
)
```
