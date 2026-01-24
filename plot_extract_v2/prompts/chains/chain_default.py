CHAIN_NAME = "default"

# Which complete schema to use for accumulated facts (constraint)
from complete_extraction_schema import ACCUMULATED_FACTS_SCHEMA

COMPLETE_SCHEMA = ACCUMULATED_FACTS_SCHEMA

# Ordered list of stage modules (importable paths)
STAGE_MODULES = [
    "plot_extract_v2.prompts.stages.stage_01_extract",
    "plot_extract_v2.prompts.stages.stage_02_code_plot",
]
