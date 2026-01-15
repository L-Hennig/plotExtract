# Extraction Progress Tracking System

The PlotExtract v2 now includes a comprehensive progress tracking system that monitors extraction quality, confidence, extracted facts, and contradictions throughout the entire extraction pipeline.

## Overview

The tracking system automatically captures:

- **Stage Progress**: Status, confidence, execution time, and output size for each extraction stage
- **Confidence Scores**: 0.0-1.0 confidence values for each stage and an overall average
- **Extracted Facts**: Structured data extracted from each stage (e.g., CSV dimensions, column names)
- **Contradictions**: When different stages produce conflicting information
- **Validation Results**: Pass/fail status for X-axis, Y-axis, point count, and trend matching
- **Execution Metrics**: Timing data and resource usage

## How It Works

### Automatic Tracking

When you run the extraction pipeline:

```bash
python plot_extract_v2/runner.py <image_path> <prompt_name>
```

The system automatically:

1. **Initializes** tracking for all stages in the pipeline
2. **Records** each stage's start and completion
3. **Extracts** facts from CSV data (if applicable)
4. **Tracks** validation results
5. **Generates** a comprehensive JSON report

### Output Files

For each extraction run, the following files are created:

```
<output_dir>/
├── example.mistral.out_data                    # Extracted CSV data
├── example.mistral.out_code                    # Generated plotting code
├── example.mistral.out_conversation            # Full conversation log
├── example.mistral.out_validate                # Validation result (yes/no)
├── example.mistral.out_validate_why            # Reasons for validation failure
└── example.mistral.out_tracking                # 🆕 Progress tracking report (JSON)
```

## Tracking Report Structure

The `_tracking` file is a JSON report containing:

```json
{
  "image_path": "/path/to/image.png",
  "prompt_name": "prompt_1",
  "overall_status": "completed",
  "validation_status": "passed",
  "total_confidence": 0.75,
  "execution_time_seconds": 45.3,
  
  "stages": {
    "STAGE_01_EXTRACT": {
      "stage_name": "STAGE_01_EXTRACT",
      "status": "completed",
      "confidence": 0.8,
      "facts_extracted": {
        "num_columns": 3,
        "num_rows": 50,
        "columns": ["x", "y1", "y2"]
      },
      "output_length": 1500,
      "execution_time_ms": 15000,
      "error": null,
      "notes": []
    }
  },
  
  "facts_by_stage": {
    "STAGE_01_EXTRACT": {
      "num_columns": 3,
      "num_rows": 50,
      "columns": ["x", "y1", "y2"]
    }
  },
  
  "contradictions": [
    {
      "stage_1": "STAGE_01_EXTRACT",
      "stage_2": "STAGE_02_CODE_PLOT",
      "field_name": "num_rows",
      "value_1": "50",
      "value_2": "48",
      "severity": "warning",
      "resolved": false
    }
  ],
  
  "output_files": {
    "validation_details": {
      "x_axis": "yes",
      "y_axis": "yes",
      "num_points": "yes",
      "trend": "yes"
    }
  }
}
```

## Analyzing Tracking Reports

### Quick View (Terminal)

View a report directly in the terminal:

```bash
python plot_extract_v2/track_analyzer.py <path_to_tracking_report.json>
```

Example output:

```
====================================================================================================
EXTRACTION TRACKING REPORT - OVERVIEW
====================================================================================================

📊 Image: /path/to/plot.png
🎯 Prompt: prompt_1
📈 Status: COMPLETED
✅ Validation: PASSED
💯 Confidence: 75.0%
⏱️  Total Time: 45.30s

----------------------------------------------------------------------------------------------------
STAGE BREAKDOWN
----------------------------------------------------------------------------------------------------

Stage                         Status       Confidence  Time (ms)    Output (chars)
----------------------------------------------------------------------------------------------------
STAGE_01_EXTRACT              ✅ completed 80.0%       15000        1500
STAGE_02_CODE_PLOT            ✅ completed 70.0%       8000         800
----------------------------------------------------------------------------------------------------
TOTAL                                                   23000 ms

----------------------------------------------------------------------------------------------------
EXTRACTED FACTS BY STAGE
----------------------------------------------------------------------------------------------------

STAGE_01_EXTRACT:
  • num_columns: 3
  • num_rows: 50
  • columns: ['x', 'y1', 'y2']

----------------------------------------------------------------------------------------------------
VALIDATION RESULTS
----------------------------------------------------------------------------------------------------

Overall: PASSED

Detailed Results:
  ✅ X Axis: PASS
  ✅ Y Axis: PASS
  ✅ Num Points: PASS
  ✅ Trend: PASS

----------------------------------------------------------------------------------------------------
CONFIDENCE ANALYSIS
----------------------------------------------------------------------------------------------------

Overall Confidence: 75.0%

Per-Stage Breakdown:
  STAGE_01_EXTRACT           [████████████████░░] 80.0%
  STAGE_02_CODE_PLOT         [██████████████░░░░] 70.0%

Interpretation:
  ✓ MODERATE CONFIDENCE - Extraction probably successful

====================================================================================================
```

### CSV Export for Batch Analysis

Export summary metrics as CSV for analysis across multiple extractions:

```bash
python plot_extract_v2/track_analyzer.py report.json --csv summary.csv
```

This creates a CSV file with:
- Image path, prompt name, overall status, validation status
- Total confidence score, execution time
- Per-stage confidence scores
- Number of contradictions

## Confidence Scoring

### How Confidence Is Calculated

Each stage receives a confidence score (0.0-1.0) based on:

- **Stage Status**: Completed stages get higher confidence
- **Output Quality**: Presence of expected data patterns
- **Validation Results**: If stage output matches expected format

**Overall Confidence** = Average of all completed stage confidences

### Confidence Interpretation

| Score | Interpretation |
|-------|---|
| 80-100% | ✨ HIGH - Extraction likely successful |
| 60-80% | ✓ MODERATE - Extraction probably successful |
| 40-60% | ⚠️ LOW - Results should be verified |
| 0-40% | ❌ VERY LOW - Extraction likely failed |

## Tracking Data: What It Captures

### For Each Stage

```python
StageProgress(
    stage_name: str              # e.g., "STAGE_01_EXTRACT"
    status: str                  # pending, running, completed, failed, skipped
    confidence: float            # 0.0 to 1.0
    facts_extracted: Dict        # Extracted structured data
    output_length: int           # Number of characters in output
    execution_time_ms: float     # Time to execute (milliseconds)
    error: Optional[str]         # Error message if failed
    validation_result: Optional[str]  # yes/no/skipped
    notes: List[str]            # Additional notes/warnings
)
```

### Extracted Facts from CSV

When CSV data is extracted, the system automatically captures:

- `num_columns`: Number of columns
- `num_rows`: Number of data rows
- `columns`: List of column headers
- `axis_labels`: Parsed axis labels

### Contradictions Tracked

The system can detect contradictions between stages:

```python
Contradiction(
    stage_1: str                 # First stage name
    stage_2: str                 # Second stage name
    field_name: str              # The field that conflicts
    value_1: Any                 # Value from stage 1
    value_2: Any                 # Value from stage 2
    severity: str                # warning, error, critical
    resolved: bool               # Whether contradiction was resolved
    resolution: Optional[str]    # How it was resolved
)
```

## Integration with Your Code

### Using the Tracker in Custom Code

```python
from extraction_tracker import ExtractionTracker

# Initialize
tracker = ExtractionTracker(image_path="plot.png", prompt_name="prompt_1")
tracker.initialize_stages(["STAGE_01", "STAGE_02"])

# During processing
tracker.start_stage("STAGE_01")
# ... do extraction work ...
tracker.complete_stage("STAGE_01", output_text, confidence=0.8, facts={...})

# Record contradictions
tracker.add_contradiction(
    stage_1="STAGE_01",
    stage_2="STAGE_02",
    field_name="num_rows",
    value_1=50,
    value_2=48,
    severity="warning"
)

# Finalize
tracker.mark_complete()
tracker.save_tracking_report("output_path.json")
tracker.print_summary()
```

## Examples

### Example 1: Check if Extraction Passed

```bash
# View the report
python plot_extract_v2/track_analyzer.py plots/synthetic/AA/AA.pv2_prompt_1.v1/AA_original.png.pv2_prompt_1.v1.mistral.out_tracking

# Check if validation passed
grep -A2 "VALIDATION RESULTS" <report_file>
```

### Example 2: Find Low-Confidence Extractions

The tracking reports can be analyzed to find problematic extractions:

```bash
# Check confidence scores across multiple runs
for f in plots/synthetic/*/report_*_tracking; do
    python plot_extract_v2/track_analyzer.py "$f" --csv temp.csv
    grep "total_confidence" temp.csv | cut -d, -f2
done
```

### Example 3: Analyze Contradictions

```bash
# Extract and display any contradictions
python -c "
import json
import sys
with open(sys.argv[1]) as f:
    data = json.load(f)
    for c in data['contradictions']:
        print(f\"❌ {c['field_name']}: {c['value_1']} vs {c['value_2']}\")
" <report_file>
```

## Best Practices

1. **Always Check the Tracking Report**: The JSON report is your source of truth for extraction quality
2. **Monitor Confidence Scores**: Confidence < 60% should trigger review
3. **Investigate Contradictions**: They often indicate data extraction issues
4. **Compare Validation Results**: A "yes" on all four validation checks is ideal
5. **Use CSV Exports**: For batch analysis across many extractions

## Files Reference

| File | Purpose |
|------|---------|
| `extraction_tracker.py` | Core tracking module |
| `track_analyzer.py` | CLI tool for analyzing reports |
| `runner.py` | Main extraction script (now with tracking integrated) |

## Customizing Confidence Scores

To customize how confidence is calculated for your stages, modify the confidence values in `runner.py`:

```python
# Current: default 0.7 confidence for all completed stages
tracker.complete_stage(stage_name, output_text, confidence=0.7, ...)

# Customize based on output quality
if len(output_text) > 1000:
    confidence = 0.8
elif len(output_text) > 500:
    confidence = 0.6
else:
    confidence = 0.4

tracker.complete_stage(stage_name, output_text, confidence=confidence, ...)
```

## Troubleshooting

### No tracking report generated

- Ensure the extraction completed without crashing
- Check that `extraction_tracker.py` is in the same directory as `runner.py`

### Low confidence scores

- Check the stage notes for specific issues
- Review the facts extracted to ensure they match expectations
- Look at contradictions between stages

### Analyzer script errors

- Ensure you're passing the correct path to the `_tracking` JSON file
- The file should be valid JSON (not a different output file)
