# PlotExtract v2 - Progress Tracking System

## 🎯 What's New

A complete progress tracking system that automatically monitors extraction quality, confidence, facts, and contradictions throughout the entire pipeline.

## 📚 Documentation Index

Start here based on your needs:

### 🚀 **Just Want to Use It?**
→ [`QUICK_START_TRACKING.md`](./QUICK_START_TRACKING.md)
- 30-second setup
- Basic commands
- View your first report

### 📖 **Want Full Details?**
→ [`TRACKING.md`](./TRACKING.md)
- Complete feature documentation
- All API details
- Advanced usage examples
- Troubleshooting guide

### 💻 **Want to Understand the Architecture?**
→ [`ARCHITECTURE.md`](./ARCHITECTURE.md)
- System flow diagrams
- Data structures
- Integration points
- Information flow visualization

### 💡 **Want Code Examples?**
→ [`example_tracking_usage.py`](./example_tracking_usage.py)
- Practical examples
- Quality check functions
- Report inspection
- Data access patterns

### 📋 **Want Implementation Details?**
→ [`../PROGRESS_TRACKING_SUMMARY.md`](../PROGRESS_TRACKING_SUMMARY.md)
- What was added
- Files changed
- Feature breakdown
- Integration summary

## 🎯 Core Features

### 1️⃣ **Confidence Scoring**
Every extraction stage receives a confidence score (0-100%):
- **80-100%**: ✨ HIGH - Ready to use
- **60-80%**: ✓ MODERATE - Probably good  
- **40-60%**: ⚠️ LOW - Review recommended
- **0-40%**: ❌ VERY LOW - Likely failed

### 2️⃣ **Fact Extraction & Tracking**
Automatically extract and track key facts:
- Data dimensions (rows × columns)
- Column/axis names
- Value ranges
- Data patterns detected

### 3️⃣ **Contradiction Detection**
Identifies when different stages produce conflicting data:
- Tracks which stages contradict
- Records what differs
- Severity levels (warning/error/critical)
- Helps debug extraction issues

### 4️⃣ **Validation Integration**
Tracks visual validation results:
- X-axis validation: ✓/✗
- Y-axis validation: ✓/✗
- Point count validation: ✓/✗
- Trend validation: ✓/✗

### 5️⃣ **Performance Metrics**
Monitor extraction efficiency:
- Per-stage execution time
- Total extraction time
- Output size tracking
- Bottleneck identification

## 📊 Quick Example

### Run extraction (automatic tracking)
```bash
python plot_extract_v2/runner.py plots/synthetic/AA/AA-original.csv prompt_1
```

### View results
```bash
python plot_extract_v2/track_analyzer.py plots/synthetic/AA/AA_original.png.pv2_prompt_1.v1.mistral.out_tracking
```

### Output
```
====================================================================================================
EXTRACTION TRACKING REPORT - OVERVIEW
====================================================================================================

📊 Image: plots/synthetic/AA/AA-original.csv
🎯 Prompt: prompt_1
📈 Status: COMPLETED
✅ Validation: PASSED
💯 Confidence: 75.0%
⏱️  Total Time: 45.30s

----------------------------------------------------------------------------------------------------
STAGE BREAKDOWN
----------------------------------------------------------------------------------------------------

Stage                         Status       Confidence  Time (ms)
----------------------------------------------------------------------------------------------------
STAGE_01_EXTRACT              ✅ completed 80.0%       15000
STAGE_02_CODE_PLOT            ✅ completed 70.0%       8000
----------------------------------------------------------------------------------------------------

CONFIDENCE ANALYSIS
Overall Confidence: 75.0%
✓ MODERATE CONFIDENCE - Extraction probably successful
```

## 📁 New Files

| File | Purpose | Lines |
|------|---------|-------|
| `extraction_tracker.py` | Core tracking module | 220 |
| `track_analyzer.py` | Report analysis tool | 380 |
| `QUICK_START_TRACKING.md` | Quick reference | - |
| `TRACKING.md` | Full documentation | - |
| `ARCHITECTURE.md` | System design | - |
| `example_tracking_usage.py` | Code examples | - |

## 🔧 How It Works

1. **Initialization**: Runner creates `ExtractionTracker` with stage list
2. **Extraction**: For each stage:
   - Start tracking: `tracker.start_stage()`
   - Run extraction (LLM call)
   - Record result: `tracker.complete_stage()` or `.fail_stage()`
   - Extract facts from CSV
3. **Validation**: Record validation results with details
4. **Finalization**: 
   - Generate JSON report
   - Print summary
   - Save to `*_tracking` file

## 💾 Output Files

After extraction, you get:
```
<output_dir>/
├── *.mistral.out_data           ← CSV data
├── *.mistral.out_code           ← Plot code
├── *.mistral.out_conversation   ← Message history
├── *.mistral.out_validate       ← yes/no
├── *.mistral.out_validate_why   ← failure reasons
└── *.mistral.out_tracking       ← 🆕 JSON report with all tracking data
```

## 🔍 Analyzing Results

### Simple: View in terminal
```bash
python plot_extract_v2/track_analyzer.py <tracking_file>
```

### Advanced: Export to CSV
```bash
python plot_extract_v2/track_analyzer.py <tracking_file> --csv summary.csv
```

### Code: Access programmatically
```python
from extraction_tracker import ExtractionTracker
import json

with open("report.json") as f:
    data = json.load(f)
    
confidence = data['total_confidence']
stages = data['stages']
facts = data['facts_by_stage']
contradictions = data['contradictions']
```

## 🎓 Learning Path

1. **Start here**: [`QUICK_START_TRACKING.md`](./QUICK_START_TRACKING.md)
2. **Try it**: Run an extraction and view the report
3. **Explore**: Read [`TRACKING.md`](./TRACKING.md) for details
4. **Understand**: Check [`ARCHITECTURE.md`](./ARCHITECTURE.md)
5. **Code**: Look at [`example_tracking_usage.py`](./example_tracking_usage.py)

## ✨ Key Highlights

✓ **Automatic**: No configuration needed, works on all extractions  
✓ **Non-invasive**: Doesn't interfere with extraction process  
✓ **Comprehensive**: Tracks everything from facts to contradictions  
✓ **Actionable**: Clear confidence scores and recommendations  
✓ **Analytical**: CSV export for batch analysis  
✓ **Visual**: Pretty terminal output with summaries  
✓ **Extensible**: Easy to customize and add new metrics  

## 🚀 Commands

```bash
# View tracking report
python plot_extract_v2/track_analyzer.py <path_to_tracking_file>

# Export to CSV
python plot_extract_v2/track_analyzer.py report.json --csv output.csv

# Run extraction (automatic tracking)
python plot_extract_v2/runner.py <image> <prompt_name>

# See examples
python plot_extract_v2/example_tracking_usage.py
```

## 🤔 FAQ

**Q: Do I need to do anything special?**  
A: No! Tracking is automatic. Just run extraction normally.

**Q: Where are the tracking reports?**  
A: They're saved as `*_tracking` files in the output directory.

**Q: How do I use the tracking data?**  
A: Use `track_analyzer.py` to view, or access the JSON directly.

**Q: Can I customize confidence scores?**  
A: Yes! See TRACKING.md for customization guide.

**Q: What if tracking fails?**  
A: Extraction will still work. Check console for errors.

## 🔗 Related Files

- Main implementation: [`extraction_tracker.py`](./extraction_tracker.py)
- Analysis tool: [`track_analyzer.py`](./track_analyzer.py)
- Integration: [`runner.py`](./runner.py) (modified)
- Implementation details: [`../PROGRESS_TRACKING_SUMMARY.md`](../PROGRESS_TRACKING_SUMMARY.md)

## 📞 Support

If something isn't working:

1. Check the console output during extraction
2. Look for error messages in the tracking report
3. See TRACKING.md troubleshooting section
4. Check example_tracking_usage.py for usage patterns

---

**Last Updated**: January 2026  
**Status**: ✅ Production Ready  
**Test Coverage**: All core features tested

The tracking system is active and running on all extractions! 🚀
