"""Configuration for verification framework parameters."""

from dataclasses import dataclass, field
from typing import List

@dataclass
class VerificationConfig:
    # Model configs
    dense_retriever_model: str = "BAAI/bge-m3"
    llm_judge_model: str = "zai/glm-4.5-flash"
    llm_judge_base_url: str = "https://gateway.movtigroup.ir/v1"

    # Search spaces for evaluation (from Phase 6)
    k_values: List[int] = field(default_factory=lambda: [10, 20, 50])
    m_values: List[int] = field(default_factory=lambda: [3, 5, 8])
    t_max_values: List[int] = field(default_factory=lambda: [0, 1, 2])
    rho_values: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5])
    tau_values: List[float] = field(default_factory=lambda: [0.4, 0.6, 0.8])

    # Default execution parameters
    k: int = 20
    m: int = 5
    t_max: int = 2
    rho: float = 0.3
    tau: float = 0.6

    # Decomposer parameters
    max_claims: int = 40
    decomposer_temperature: float = 0.0

    # Retrieval parameters
    chunk_size: int = 200
    chunk_overlap: int = 50
    rrf_k: int = 60 # RRF = 1 / (k + rank)

    # Judge parameters
    judge_temperature: float = 0.7
    judge_samples: int = 5

default_config = VerificationConfig()
