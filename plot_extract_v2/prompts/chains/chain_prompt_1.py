# Chain definition for Prompt Set 1
# Three-stage extraction with accumulated facts
# Specifies which stages to execute and in what order

CHAIN_NAME = "prompt_1"

# Ordered list of stage names to execute (must match variable names in prompts.py)
EXTRACT_STAGES = [
    "EXTRACT_STAGE_1",  # X-axis verification
    "EXTRACT_STAGE_2",  # Y-axis verification (uses Stage 1 facts)
    "EXTRACT_STAGE_3",  # Marker extraction + CSV (uses all facts)
]
