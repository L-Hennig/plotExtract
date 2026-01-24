# =====================================================================
# Chain configuration for prompt 12
# Defines the extraction stages and their order
# =====================================================================

CHAIN_NAME = "prompt_12"

# Which complete schema to use for accumulated facts (constraint)
from complete_schema_reevaluation import ACCUMULATED_FACTS_SCHEMA, SCHEMA_CONSTRAINTS

COMPLETE_SCHEMA = ACCUMULATED_FACTS_SCHEMA
COMPLETE_SCHEMA_CONSTRAINTS = SCHEMA_CONSTRAINTS

# Stages that must NOT receive the plot image as input
NO_IMAGE_STAGES = [
    "EXTRACT_STAGE_6",  # Stage 6: deterministic evaluation from CSV + diagnostics only
    "EXTRACT_STAGE_8",  # Stage 8: merge CSV fixes only
]

# Extraction stages in order
EXTRACT_STAGES = [
    "EXTRACT_STAGE_1",  # Plot type verification (time-kill)
    "EXTRACT_STAGE_2",  # X-axis verification (time in hours)
    "EXTRACT_STAGE_3",  # Y-axis verification (bacterial burden, log10)
    "EXTRACT_STAGE_4",  # Curve definitions (style metadata)
    "EXTRACT_STAGE_5",  # Data point extraction (CSV + JSON)
    "EXTRACT_STAGE_6",  # Extraction evaluation (uses CSV + diagnostics)
    "EXTRACT_STAGE_7",  # Targeted re-extraction (uses image)
    "EXTRACT_STAGE_8",  # Compile new CSV (merge fixes)
]




