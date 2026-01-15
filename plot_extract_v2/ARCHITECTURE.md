# Progress Tracking Architecture

## System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    runner.py (Main Script)                      │
│                                                                   │
│  1. Parse arguments                                              │
│  2. Load prompts & chains                                        │
│  3. Encode image                                                 │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Initialize ExtractionTracker                             │   │
│  │ └─ tracker.initialize_stages(EXTRACT_STAGES)             │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          │                                        │
└──────────────────────────┼────────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────────┐
        │   For each EXTRACT_STAGE in pipeline:        │
        │                                               │
        │  1. tracker.start_stage(stage_name)           │
        │  2. Call LLM with prompt + image              │
        │  3. tracker.complete_stage() or .fail_stage() │
        │  4. Extract facts if CSV-like output          │
        │  5. Check for contradictions                  │
        │                                               │
        └──────────────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────────┐
        │   Generate visualization code (CODE_PLOT)    │
        │   Execute code or fix errors                 │
        │   Run validation tests                        │
        │   tracker.set_validation_result()            │
        │                                               │
        └──────────────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────────┐
        │   Final Steps:                               │
        │                                               │
        │  • tracker.mark_complete()                   │
        │  • tracker.save_tracking_report(path)        │
        │  • tracker.print_summary()                   │
        │                                               │
        └──────────────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────────┐
        │   Output Files Generated:                    │
        │                                               │
        │  ✓ *.mistral.out_data                        │
        │  ✓ *.mistral.out_code                        │
        │  ✓ *.mistral.out_conversation                │
        │  ✓ *.mistral.out_validate                    │
        │  ✓ *.mistral.out_tracking ← 🆕              │
        │                                               │
        └──────────────────────────────────────────────┘
```

## Data Structure Hierarchy

```
ExtractionTracker
├── stages: Dict[str, StageProgress]
│   ├── stage_name: str
│   ├── status: "pending|running|completed|failed|skipped"
│   ├── confidence: float (0.0-1.0)
│   ├── facts_extracted: Dict[str, Any]
│   ├── output_length: int
│   ├── execution_time_ms: float
│   ├── error: Optional[str]
│   ├── validation_result: Optional[str]
│   ├── timestamp: str
│   └── notes: List[str]
│
├── facts: Dict[str, Dict[str, Any]]
│   ├── stage_name: {extracted facts}
│   └── ...
│
├── contradictions: List[Contradiction]
│   ├── stage_1: str
│   ├── stage_2: str
│   ├── field_name: str
│   ├── value_1: Any
│   ├── value_2: Any
│   ├── severity: "warning|error|critical"
│   ├── resolved: bool
│   └── resolution: Optional[str]
│
├── total_confidence: float
├── validation_status: str
├── overall_status: str
└── output_files: Dict[str, Any]
    └── validation_details: Dict[test_name, result]
```

## Integration Points in runner.py

```python
# Line 1-20: Imports
from extraction_tracker import ExtractionTracker
import time

# Line 375-380: Tracker Initialization
tracker = ExtractionTracker(input_plot, prompt_name)
tracker.initialize_stages(EXTRACT_STAGES)

# Line 390-435: Main extraction loop (FOR EACH STAGE)
tracker.start_stage(stage_name)
stage_start_time = time.time()

# ... LLM call ...

stage_time = (time.time() - stage_start_time) * 1000
facts = tracker.extract_facts_from_csv(result_text)
tracker.complete_stage(stage_name, result_text, 
                      confidence=0.7,
                      execution_time_ms=stage_time,
                      facts=facts)

# Line 425: Early exit tracking
if result_text == "None":
    tracker.fail_stage(stage_name, "Returned None")
    # ... cleanup ...

# Line 440: Mark extraction complete
tracker.mark_complete()

# Line 530-535: Validation tracking
validation_details = {
    'x_axis': validate_x,
    'y_axis': validate_y,
    'num_points': validate_n,
    'trend': validate_t
}
tracker.set_validation_result(result_flag, validation_details)

# Line 595-600: Save and display
tracker.save_tracking_report(output_out + "_tracking")
tracker.print_summary()
```

## Information Flow

```
INPUT
│
├─→ Image + Prompts
│
▼
ExtractionTracker
├─→ Tracks each stage execution
├─→ Collects confidence scores
├─→ Extracts facts from outputs
├─→ Detects contradictions
├─→ Records validation results
├─→ Measures performance
│
▼
JSON Report (*_tracking)
├─→ Stages data
├─→ Facts by stage
├─→ Contradictions
├─→ Validation details
├─→ Execution metrics
│
▼
track_analyzer.py
├─→ Parse JSON
├─→ Format for display
├─→ Generate summaries
├─→ Export to CSV
│
▼
OUTPUT
├─→ Terminal display (pretty tables)
├─→ CSV export (for batch analysis)
├─→ Programmatic access (JSON)
└─→ Summary statistics
```

## Confidence Calculation

```
┌─────────────────────────────────────────┐
│ Stage Completes Successfully            │
│ └─ confidence = 0.7 (default)           │
│    (customizable based on output)       │
└─────────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────────┐
        │ Collect All Stage         │
        │ Confidences               │
        │ [0.8, 0.7, 0.9]           │
        └───────────────────────────┘
                    │
                    ▼
        ┌───────────────────────────┐
        │ Average = sum / count     │
        │ (0.8 + 0.7 + 0.9) / 3     │
        │ = 0.8 = 80%               │
        └───────────────────────────┘
                    │
                    ▼
        ┌───────────────────────────┐
        │ Interpret Confidence      │
        │ 80% = ✨ HIGH             │
        │ Ready to use              │
        └───────────────────────────┘
```

## Validation Tracking

```
Validation Stage
│
├─→ Compare X Axes → ✓ YES or ✗ NO
├─→ Compare Y Axes → ✓ YES or ✗ NO
├─→ Compare Points → ✓ YES or ✗ NO
├─→ Compare Trends → ✓ YES or ✗ NO
│
▼
All tests pass? → Validation = "yes"  ✓ PASSED
Any test fails? → Validation = "no"   ✗ FAILED
No comparison?  → Validation = "skipped"
│
▼
tracker.set_validation_result(status, details)
│
▼
Report includes:
├─ validation_status (string)
└─ validation_details (dict with each test result)
```

## Contradiction Detection

```
Stage 1 Output          Stage 2 Output
│                        │
├─ num_rows = 50        ├─ num_rows = 48
├─ num_cols = 3         ├─ num_cols = 3
└─ axis_x = "Time"      └─ axis_x = "Time"
│                        │
└────────┬───────────────┘
         │
         ▼
    Compare Facts
    │
    ├─ num_rows: 50 ≠ 48 → CONTRADICTION!
    ├─ num_cols: 3 = 3 ✓
    └─ axis_x: same ✓
    │
    ▼
tracker.add_contradiction(
    stage_1="STAGE_01",
    stage_2="STAGE_02",
    field_name="num_rows",
    value_1=50,
    value_2=48,
    severity="warning"
)
    │
    ▼
Report includes contradiction with:
├─ Both stages
├─ Which field
├─ Both values
└─ Severity level
```

## Quick Reference: What Each Component Does

| Component | Responsibility |
|-----------|-----------------|
| `ExtractionTracker` | Main tracker class, orchestrates all tracking |
| `StageProgress` | Dataclass storing single stage metrics |
| `Contradiction` | Dataclass for conflict records |
| `runner.py` | Calls tracker methods during extraction |
| `track_analyzer.py` | Reads JSON reports and displays them |
| `*_tracking` | JSON output file with all tracking data |

---

**Total System Size**: ~600 lines of code across 2 main modules + 1 analysis tool
