# =====================================================================
# Chain configuration for prompt_3
# Defines the extraction stages and their order
# =====================================================================

CHAIN_NAME = "prompt_3"

# Extraction stages in order
EXTRACT_STAGES = [
    "EXTRACT_STAGE_1",  # Plot type verification (time-kill)
    "EXTRACT_STAGE_2",  # X-axis verification (time in hours)
    "EXTRACT_STAGE_3",  # Y-axis verification (bacterial burden, log10)
    "EXTRACT_STAGE_4",  # Data point extraction (CSV + JSON)
    "EXTRACT_STAGE_5",  # Validation / sanity checks
]
