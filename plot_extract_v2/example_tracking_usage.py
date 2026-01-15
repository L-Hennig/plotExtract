"""
Example: Using the Extraction Tracker

This demonstrates how the tracking system works and what data it captures.
Run this after executing an extraction to see the tracking report.
"""

import json
import sys
from pathlib import Path

# Example: Load and inspect a tracking report
def example_inspect_tracking_report():
    """Show how to programmatically inspect a tracking report."""
    
    # Path to tracking report (adjust to your extraction output)
    report_path = "plots/synthetic/AA/AA_original.png.pv2_prompt_1.v1.mistral.out_tracking"
    
    if not Path(report_path).exists():
        print(f"Note: Example report not found at {report_path}")
        print("Run an extraction first to generate tracking data.\n")
        return
    
    # Load the report
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    print("=" * 80)
    print("TRACKING REPORT INSPECTION EXAMPLE")
    print("=" * 80)
    
    # 1. Overall statistics
    print("\n1. OVERALL STATISTICS")
    print(f"   Image: {report['image_path']}")
    print(f"   Status: {report['overall_status']}")
    print(f"   Overall Confidence: {report['total_confidence']:.1%}")
    print(f"   Validation: {report['validation_status']}")
    print(f"   Execution Time: {report['execution_time_seconds']:.2f}s")
    
    # 2. Stage-by-stage breakdown
    print("\n2. STAGE BREAKDOWN")
    for stage_name, stage_data in report['stages'].items():
        print(f"\n   {stage_name}:")
        print(f"     Status: {stage_data['status']}")
        print(f"     Confidence: {stage_data['confidence']:.1%}")
        print(f"     Time: {stage_data['execution_time_ms']:.0f}ms")
        print(f"     Output Length: {stage_data['output_length']} chars")
        
        if stage_data['error']:
            print(f"     ERROR: {stage_data['error'][:50]}...")
    
    # 3. Extracted facts
    print("\n3. EXTRACTED FACTS")
    for stage_name, facts in report['facts_by_stage'].items():
        print(f"\n   {stage_name}:")
        for key, value in facts.items():
            if isinstance(value, list) and len(value) > 3:
                value_str = f"[{len(value)} items]"
            else:
                value_str = str(value)
            print(f"     {key}: {value_str}")
    
    # 4. Contradictions
    print("\n4. CONTRADICTIONS")
    contradictions = report.get('contradictions', [])
    if contradictions:
        for i, c in enumerate(contradictions, 1):
            print(f"   {i}. {c['stage_1']} vs {c['stage_2']}")
            print(f"      Field: {c['field_name']}")
            print(f"      Values: '{c['value_1']}' vs '{c['value_2']}'")
            print(f"      Severity: {c['severity']}")
    else:
        print("   None detected ✓")
    
    # 5. Validation results
    print("\n5. VALIDATION DETAILS")
    validation_details = report.get('output_files', {}).get('validation_details', {})
    if validation_details:
        for test_name, result in validation_details.items():
            status = "✓" if "yes" in result.lower() else "✗"
            print(f"   {status} {test_name}: {result}")
    
    print("\n" + "=" * 80)


def example_quality_check():
    """Example: Check extraction quality using tracking data."""
    
    report_path = "plots/synthetic/AA/AA_original.png.pv2_prompt_1.v1.mistral.out_tracking"
    
    if not Path(report_path).exists():
        print("Report not found. Run extraction first.")
        return
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    print("\n" + "=" * 80)
    print("QUALITY CHECK")
    print("=" * 80)
    
    checks = []
    
    # Check 1: Overall confidence
    confidence = report['total_confidence']
    if confidence >= 0.8:
        checks.append(("Confidence", "✓ HIGH", True))
    elif confidence >= 0.6:
        checks.append(("Confidence", "⚠ MODERATE", True))
    else:
        checks.append(("Confidence", "✗ LOW", False))
    
    # Check 2: Validation status
    validation = report['validation_status']
    passed = validation == "yes"
    checks.append(("Validation", f"{'✓ PASSED' if passed else '✗ FAILED'}", passed))
    
    # Check 3: No errors
    has_errors = any(
        stage.get('error') for stage in report['stages'].values()
    )
    checks.append(("Errors", f"{'✓ NONE' if not has_errors else '✗ PRESENT'}", not has_errors))
    
    # Check 4: No contradictions
    contradictions = report.get('contradictions', [])
    has_contradictions = len(contradictions) > 0
    checks.append(("Contradictions", f"{'✓ NONE' if not has_contradictions else f'✗ {len(contradictions)}'}", not has_contradictions))
    
    # Check 5: Execution time reasonable
    exec_time = report['execution_time_seconds']
    time_ok = exec_time < 120
    checks.append(("Execution Time", f"{'✓' if time_ok else '⚠'} {exec_time:.1f}s", time_ok))
    
    # Print results
    for check_name, result, status in checks:
        status_icon = "✓" if status else "✗"
        print(f"  {status_icon} {check_name:25} {result}")
    
    # Overall verdict
    all_passed = all(status for _, _, status in checks)
    print("\n" + "-" * 80)
    if all_passed:
        print("✓ EXTRACTION QUALITY: GOOD - Ready for use")
    else:
        failed_checks = [name for name, _, status in checks if not status]
        print(f"✗ EXTRACTION QUALITY: ISSUES - Review: {', '.join(failed_checks)}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    print("\n" + "▶" * 40)
    print("EXTRACTION TRACKING SYSTEM - EXAMPLES")
    print("▶" * 40)
    
    print("\n📖 Example 1: Inspect a Tracking Report")
    print("-" * 80)
    example_inspect_tracking_report()
    
    print("\n📊 Example 2: Quality Check")
    print("-" * 80)
    example_quality_check()
    
    print("\n💡 To use these in your code:")
    print("   from extraction_tracker import ExtractionTracker")
    print("   tracker = ExtractionTracker('image.png', 'prompt_1')")
    print("   # ... run extraction ...")
    print("   summary = tracker.get_summary()  # Get all tracking data")
    print("   tracker.save_tracking_report('output.json')")
