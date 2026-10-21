"""Constants and thresholds used in the verification methodology."""

# Search space for hyperparameters
K_VALUES = [10, 20, 50]         # Number of retrieved top snippets
M_VALUES = [3, 5, 8]            # Number of kept snippets as evidence E_a
T_MAX_VALUES = [0, 1, 2]        # Maximum retry loops for unverified claims
RHO_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5] # Uncertainty threshold u(y) >= rho
TAU_VALUES = [0.4, 0.6, 0.8]    # Claim confidence threshold c(a) < tau -> N

# Default configuration based on paper
DEFAULT_K = 20
DEFAULT_M = 5
DEFAULT_T_MAX = 2
DEFAULT_RHO = 0.3
DEFAULT_TAU = 0.6

# Labels
LABEL_S = "S" # Supported / Entailed
LABEL_C = "C" # Contradicted
LABEL_N = "N" # Not Enough Evidence / Baseless
LABEL_MIXED = "MIXED" # Both S and C evidence exist
