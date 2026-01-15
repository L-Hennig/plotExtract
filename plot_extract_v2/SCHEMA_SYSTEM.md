# Schema-Based Accumulated Facts System

## Overview

Implemented a two-schema approach for extraction tracking:

1. **Complete Schema** (`extraction_schema.json`) - Defines all possible fields across all stages
2. **Accumulated Facts** - Gradually built JSON object that grows with each stage

## Files Created/Modified

### New Files
- **`extraction_schema.json`** - Clean JSON schema with all possible fields and no comments
- Currently loading schema in `runner.py` and passing to prompts

### Modified Files

#### `prompts.py` (prompt_1)
Updated all extraction stages to:
- Receive `{complete_schema}` - Shows all possible fields
- Receive `{accumulated_facts}` - Shows what's been determined so far
- Output only populated fields (no empty fields)

**Stage progression:**
1. Stage 1: Receives empty accumulated facts, fills only `axis_facts.x_axis`
2. Stage 2: Receives facts from Stage 1, adds `axis_facts.y_axis`
3. Stage 3: Receives facts from Stages 1+2, adds `marker_facts`

#### `runner.py`
Added functionality:
- `load_extraction_schema()` - Loads the complete schema JSON
- Schema and accumulated facts passed to each prompt
- JSON parsing of stage outputs with automatic merging into accumulated facts
- Backwards compatible with old `{data_context}` style prompts

## How It Works

```
Initialize:
  accumulated_facts = {article_info if provided}
  complete_schema = loaded from JSON

For each stage:
  1. Format prompt with:
     - complete_schema (JSON template)
     - accumulated_facts (what's been found so far)
  2. Run LLM
  3. Parse output as JSON
  4. Merge output into accumulated_facts
  5. Pass updated accumulated_facts to next stage

Result:
  Single accumulated JSON object with only populated fields
```

## Schema Access

Each prompt now receives:
```
COMPLETE SCHEMA:
{
  "article_info": {...},
  "axis_facts": {...},
  "marker_facts": {...}
}

ACCUMULATED FACTS SO FAR:
{
  "article_info": {...populated if provided...},
  "axis_facts": {
    "x_axis": {...what stage 1 found...}
  }
}
```

Stages know what fields are possible and what's been determined so far.

## Benefits

✓ Each stage knows the complete structure (from schema)
✓ Each stage knows what's already been determined (from accumulated)
✓ Stages only output new/updated fields
✓ Clean JSON merging throughout
✓ Easy to track what changed at each stage
✓ Backwards compatible with old prompt format

## Backwards Compatibility

Old prompts using `{data_context}` still work - they receive accumulated context as before.
