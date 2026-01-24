# =====================================================================
# Chain configuration for prompt9
# Defines the extraction stages and their order
# =====================================================================

CHAIN_NAME = "prompt_9"

# Which complete schema to use for accumulated facts (constraint)
from complete_extraction_schema import ACCUMULATED_FACTS_SCHEMA

COMPLETE_SCHEMA = ACCUMULATED_FACTS_SCHEMA

# Extraction stages in order
EXTRACT_STAGES = [
    "EXTRACT_STAGE_1",  # Plot type verification (time-kill)
    "EXTRACT_STAGE_2",  # X-axis verification (time in hours)
    "EXTRACT_STAGE_3",  # Y-axis verification (bacterial burden, log10)
    "EXTRACT_STAGE_4",  # Curve definitions (style metadata)
    "EXTRACT_STAGE_5"  # Data point extraction (CSV + JSON)
]


