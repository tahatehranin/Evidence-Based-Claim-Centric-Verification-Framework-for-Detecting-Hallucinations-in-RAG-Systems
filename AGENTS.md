# Agent Instructions for RAG Hallucination Detection Framework

This repository implements the methodology for an evidence-based, claim-centric hallucination detection framework in Retrieval-Augmented Generation (RAG) systems.

## Coding Conventions
- **Language**: Python 3.10+
- **Style**: PEP 8 compliance.
- **Typing**: Use standard Python type hinting (`typing` module) for all function arguments and return values.
- **Docstrings**: Use clear, concise docstrings for all modules, classes, and methods explaining their purpose and parameters.

## Core Methodology & Equations
The Python code MUST rigorously follow the mathematical symbols and methodology from the core paper (`main_paper/مقاله_RAG_متدولوژی_دقیق_2__1_.docx`).
- **Claim Extraction**: The answer is divided into atomic claims $C = \{c_1, c_2, ..., c_n\}$.
- **Verification Labels**: Each claim $c_i$ is mapped to $y_i \in \{S, C, N\}$ (Supported, Contradicted, Not Enough Evidence/Baseless).
- **Equations to Implement/Keep Track of**:
  - Precision ($P$), Recall ($R$), F1 ($F1$), Macro-F1.
  - $P = TP / (TP + FP)$
  - $R = TP / (TP + FN)$
  - $F1 = 2PR / (P + R)$
  - Macro-F1 = $(1/3) \sum_{c \in \{S, C, N\}} F1_c$
  - Uncertainty threshold $\rho$ (e.g., if uncertainty $u(y) \geq \rho$, the system abstains).
  - Search spaces for hyperparameters: $k \in \{10, 20, 50\}$, $m \in \{3, 5, 8\}$, $T_{max} \in \{0, 1, 2\}$, $\rho \in \{0.1, 0.2, 0.3, 0.4, 0.5\}$, $\tau \in \{0.4, 0.6, 0.8\}$.
  - Cost metrics: $C_{rel} = (T_{in} + T_{out})_{verify} / (T_{in} + T_{out})_{generation}$

## Execution Workflow
1. When editing verification logic (`src/local_verifier.py`, `src/global_verifier.py`, `src/answer_aggregator.py`), make sure threshold checks map exactly to $\tau$ (tau), $\rho$ (rho), and other parameters mentioned in the paper.
2. The code must produce JSON outputs that record unique identifiers, origins, and intermediate decisions for traceability.
3. Tests must be added or updated whenever verification rules or mathematical formulas are altered.

## Pre-Commit and Tests
Always run `pytest tests/` before submitting. Do not push failing tests.
Ensure the `.gitignore` correctly prevents tracking of `__pycache__` and other runtime artifacts.
