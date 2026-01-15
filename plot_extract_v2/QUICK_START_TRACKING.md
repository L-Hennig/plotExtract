# Quick Start: Progress Tracking

## TL;DR

The extraction system now **automatically tracks progress** for every run. Each extraction creates a `*_tracking` JSON file with confidence scores, extracted facts, contradictions, and validation results.

## Getting Started (30 seconds)

### 1. Run Extraction (same as before)
```bash
python plot_extract_v2/runner.py plots/synthetic/AA/AA-original.csv prompt_1
```

### 2. View the Summary (new!)
At the end of extraction, you'll see:
```
EXTRACTION PROGRESS SUMMARY
===========================
Image: plots/synthetic/AA/AA-original.csv
Status: COMPLETED
Overall Confidence: 75.0%
...
```

### 3. View Full Report
```bash
python plot_extract_v2/track_analyzer.py plots/synthetic/AA/AA_original.png.pv2_prompt_1.v1.mistral.out_tracking
```

## What You Get

### Confidence Scores
- **Per-stage**: How confident each extraction step was (0-100%)
- **Overall**: Average confidence across all stages
- **Interpretation**: HIGH/MODERATE/LOW/VERY LOW with recommendations

### Extracted Facts
- Number of data points extracted
- Column names and axis labels
- Any patterns detected

### Contradictions
- When different stages report conflicting data
- Severity: warning/error/critical
- Helps identify data extraction issues

### Validation Results
- ✓/✗ for X-axis, Y-axis, point count, trend
- Shows which aspects of the extraction passed validation

### Timing & Performance
- Execution time per stage
- Total extraction time
- Performance bottleneck identification

## Files Generated

For each extraction:

```
<output_dir>/
├── example.mistral.out_data          ← CSV data
├── example.mistral.out_code          ← Plotting code
├── example.mistral.out_validate      ← yes/no result
├── example.mistral.out_validate_why  ← failure reasons (if failed)
└── example.mistral.out_tracking      ← 🆕 TRACKING REPORT (JSON)
```

The `*_tracking` file is your main source of truth for extraction quality.

## Commands

### View any tracking report
```bash
python plot_extract_v2/track_analyzer.py <path_to_tracking_file>
```

### Export results as CSV (for batch analysis)
```bash
python plot_extract_v2/track_analyzer.py report.json --csv summary.csv
```

### Programmatic access
```python
from extraction_tracker import ExtractionTracker
import json

# Load report
with open("example.mistral.out_tracking") as f:
    data = json.load(f)

# Access data
confidence = data['total_confidence']
stages = data['stages']
facts = data['facts_by_stage']
validation = data['output_files']['validation_details']
```

## Understanding Confidence

```
80-100% ✨  HIGH       → Ready to use
60-80%  ✓   MODERATE   → Probably good
40-60%  ⚠️  LOW        → Review recommended
0-40%   ❌  VERY LOW   → Likely failed
```

## Checking Quality

Quick checklist:
- [ ] Overall confidence > 60%?
- [ ] Validation status = "yes"?
- [ ] No errors in any stage?
- [ ] No contradictions, or all resolved?
- [ ] Execution time reasonable (< 2min)?

If all ✓, extraction is good!

## Examples

### "Is this extraction good?"
```bash
# View report
python plot_extract_v2/track_analyzer.py report.json

# Look for:
# - Confidence: 60%+
# - Status: COMPLETED
# - Validation: PASSED
# - Contradictions: None
```

### "Why did validation fail?"
```bash
# Check validation details
python plot_extract_v2/track_analyzer.py report.json

# Look at VALIDATION RESULTS section
# Shows which tests failed (X axis, Y axis, points, trend)
```

### "Which extraction was fastest?"
```bash
# Compare execution times
python -c "
import json
for f in ['report1.json', 'report2.json']:
    with open(f) as file:
        data = json.load(file)
        print(f\"{f}: {data['execution_time_seconds']:.1f}s\")
"
```

## Troubleshooting

### No tracking report generated?
- Extraction must complete (or fail cleanly)
- Check `runner.py` has `extraction_tracker.py` imported
- Look for errors in terminal output

### Low confidence?
- Check the stage breakdown to see which stage is weak
- Look at extracted facts - are they what you expect?
- Review contradictions between stages
- Increase prompt quality

### Validation failed?
- See which test failed (X, Y, points, trend)
- This tells you what's wrong with the extraction
- May need to adjust extraction prompt

## Full Documentation

For complete details, see:
- [`TRACKING.md`](./plot_extract_v2/TRACKING.md) - Full guide
- [`example_tracking_usage.py`](./plot_extract_v2/example_tracking_usage.py) - Code examples
- [`PROGRESS_TRACKING_SUMMARY.md`](./PROGRESS_TRACKING_SUMMARY.md) - Implementation details

## Key Takeaway

**Every extraction now produces tracking data automatically.** Use the confidence scores and contradictions to understand extraction quality and debug issues.

---

*Progress tracking is active and ready to use. No configuration needed!*
