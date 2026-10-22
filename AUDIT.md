# AUDIT of RAG Hallucination Detection Framework

## 1. Gaps between Codebase and Methodology Paper
- **Retrieval**: The code currently uses basic lexical similarity (`LocalVerifier` and `GlobalVerifier`). The paper specifies hybrid BM25 + dense retrieval (BGE-M3) with Reciprocal Rank Fusion (RRF) $1/(60+rank)$ over top $k=20$.
- **Judge / LLM**: The current code heavily relies on heuristic thresholds (e.g., similarity `0.5 × sim + 0.3 × chunk + 0.2`) for evaluation. The paper mandates an independent three-way LLM judge evaluating claim+evidence+metadata.
- **Labels**: Current labels are 'Entailed', 'Contradicted', and 'Baseless'. The paper explicitly uses S, C, N. This mapping exists in variables but needs enforcement across strings and outputs.
- **Verification Policy**: The paper requires a correction/re-retrieval/abstention policy ($T_{max}$, $u(y) \geq \rho$, etc.). This is entirely absent in the current implementation.
- **Provenance Records**: The system doesn't generate rigorous JSON provenance traces from final claims back to initial answers.
- **Web Verification**: Current implementation in `AGENTS.md` and `QUICKSTART.md` mentions "live web verification" via tiers, whereas the paper strictly outlines fixed dated corpus snapshots (closed/open modes).

## 2. PR Review
- **Open PRs**: PR #1 (setup tests) and PR #2 (organize references) exist on remote. PR #2 was already merged into main (`14a49e9`), which included PDF renaming, README updates, and `main_paper` docx renaming.
- **Status**: The repository currently reflects the merged state of PR #2.

## 3. Bugs and Outdated Dependencies
- **Dependencies**: `requirements.txt` is missing explicit pinning for production reproducibility (e.g., exact versions of `sentence-transformers`, `torch`, etc., to avoid future breakages) and is missing standard LLM libraries (e.g., `openai` or specific SDKs for the LLM judge).
- **Hard-coded Parameters**: Many threshold values (e.g., overlap=50, similarity_threshold=0.30) are hard-coded in class initializers inside `src/`. They should be extracted into centralized config objects.
- **README/Docs URLs**: The clone URL in `README.md` (`https://github.com/tahatehran/...`) needs to point to the current fork.
