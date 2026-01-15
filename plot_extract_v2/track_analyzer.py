"""
Progress analyzer and visualizer for extraction tracking reports.
Usage: python track_analyzer.py <path_to_tracking_report.json>
"""
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, List


class TrackingReportAnalyzer:
    """Analyze and visualize extraction tracking reports."""
    
    def __init__(self, report_path: str):
        self.report_path = report_path
        self.report = self._load_report()
    
    def _load_report(self) -> Dict[str, Any]:
        """Load the JSON tracking report."""
        with open(self.report_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def print_overview(self):
        """Print a high-level overview of the extraction."""
        print("\n" + "=" * 100)
        print("EXTRACTION TRACKING REPORT - OVERVIEW")
        print("=" * 100)
        
        print(f"\n📊 Image: {self.report.get('image_path', 'N/A')}")
        print(f"🎯 Prompt: {self.report.get('prompt_name', 'N/A')}")
        print(f"📈 Status: {self.report.get('overall_status', 'N/A').upper()}")
        print(f"✅ Validation: {self.report.get('validation_status', 'N/A').upper()}")
        print(f"💯 Confidence: {self.report.get('total_confidence', 0):.1%}")
        print(f"⏱️  Total Time: {self.report.get('execution_time_seconds', 0):.2f}s")
    
    def print_stages(self):
        """Print detailed stage information."""
        print("\n" + "-" * 100)
        print("STAGE BREAKDOWN")
        print("-" * 100)
        
        stages = self.report.get('stages', {})
        if not stages:
            print("No stage data found.")
            return
        
        # Create table header
        print(f"{'Stage':<30} {'Status':<12} {'Confidence':<12} {'Time (ms)':<12} {'Output (chars)':<15}")
        print("-" * 100)
        
        total_time = 0
        for stage_name, stage_data in stages.items():
            status = stage_data.get('status', 'unknown')
            confidence = stage_data.get('confidence', 0)
            execution_time = stage_data.get('execution_time_ms', 0)
            output_length = stage_data.get('output_length', 0)
            
            total_time += execution_time
            
            # Status emoji/color indicators
            status_emoji = {
                'completed': '✅',
                'running': '⏳',
                'pending': '⏹️',
                'failed': '❌',
                'skipped': '⊘'
            }.get(status, '?')
            
            confidence_str = f"{confidence:.1%}" if confidence > 0 else "N/A"
            time_str = f"{execution_time:.0f}"
            
            print(f"{stage_name:<30} {status_emoji} {status:<10} {confidence_str:<12} {time_str:<12} {output_length:<15}")
            
            # Print error if exists
            if stage_data.get('error'):
                print(f"  {'ERROR':<28} {stage_data['error'][:70]}")
            
            # Print notes
            for note in stage_data.get('notes', []):
                print(f"  {'NOTE':<28} {note[:70]}")
        
        print("-" * 100)
        print(f"{'TOTAL':<30} {'':<12} {'':<12} {total_time:.0f} ms")
    
    def print_facts(self):
        """Print extracted facts by stage."""
        facts_by_stage = self.report.get('facts_by_stage', {})
        
        if not facts_by_stage:
            print("\n📝 EXTRACTED FACTS: No facts found.")
            return
        
        print("\n" + "-" * 100)
        print("EXTRACTED FACTS BY STAGE")
        print("-" * 100)
        
        for stage_name, facts in facts_by_stage.items():
            print(f"\n{stage_name}:")
            for key, value in facts.items():
                value_str = str(value)
                if len(value_str) > 70:
                    value_str = value_str[:67] + "..."
                print(f"  • {key}: {value_str}")
    
    def print_contradictions(self):
        """Print any contradictions found."""
        contradictions = self.report.get('contradictions', [])
        
        print("\n" + "-" * 100)
        print("CONTRADICTIONS")
        print("-" * 100)
        
        if not contradictions:
            print("✨ No contradictions detected!")
            return
        
        for i, contradiction in enumerate(contradictions, 1):
            severity = contradiction.get('severity', 'warning').upper()
            stage_1 = contradiction.get('stage_1', 'unknown')
            stage_2 = contradiction.get('stage_2', 'unknown')
            field = contradiction.get('field_name', 'unknown')
            value_1 = contradiction.get('value_1', 'N/A')
            value_2 = contradiction.get('value_2', 'N/A')
            resolved = contradiction.get('resolved', False)
            
            severity_emoji = {
                'WARNING': '⚠️',
                'ERROR': '❌',
                'CRITICAL': '🚨'
            }.get(severity, '❓')
            
            print(f"\n{i}. {severity_emoji} [{severity}] {stage_1} vs {stage_2}")
            print(f"   Field: {field}")
            print(f"   {stage_1}: '{value_1}'")
            print(f"   {stage_2}: '{value_2}'")
            if resolved:
                print(f"   ✓ Resolved")
    
    def print_validation(self):
        """Print validation details."""
        validation_status = self.report.get('validation_status', 'unknown')
        output_files = self.report.get('output_files', {})
        validation_details = output_files.get('validation_details', {})
        
        print("\n" + "-" * 100)
        print("VALIDATION RESULTS")
        print("-" * 100)
        
        print(f"Overall: {validation_status.upper()}")
        
        if validation_details:
            print("\nDetailed Results:")
            for test_name, result in validation_details.items():
                result_lower = result.lower()
                result_emoji = '✅' if 'yes' in result_lower else '❌' if 'no' in result_lower else '❓'
                result_short = 'PASS' if 'yes' in result_lower else 'FAIL' if 'no' in result_lower else result
                print(f"  {result_emoji} {test_name.replace('_', ' ').title()}: {result_short}")
    
    def print_confidence_analysis(self):
        """Analyze and print confidence breakdown."""
        stages = self.report.get('stages', {})
        total_confidence = self.report.get('total_confidence', 0)
        
        print("\n" + "-" * 100)
        print("CONFIDENCE ANALYSIS")
        print("-" * 100)
        
        print(f"Overall Confidence: {total_confidence:.1%}")
        
        if stages:
            print("\nPer-Stage Breakdown:")
            # Sort by stage order if available
            for stage_name, stage_data in stages.items():
                if stage_data.get('status') == 'completed':
                    confidence = stage_data.get('confidence', 0)
                    bar_length = int(confidence * 20)
                    bar = '█' * bar_length + '░' * (20 - bar_length)
                    print(f"  {stage_name:<30} [{bar}] {confidence:.1%}")
        
        # Confidence interpretation
        print("\nInterpretation:")
        if total_confidence >= 0.8:
            print("  ✨ HIGH CONFIDENCE - Extraction likely successful")
        elif total_confidence >= 0.6:
            print("  ✓ MODERATE CONFIDENCE - Extraction probably successful")
        elif total_confidence >= 0.4:
            print("  ⚠️  LOW CONFIDENCE - Results should be verified")
        else:
            print("  ❌ VERY LOW CONFIDENCE - Extraction likely failed")
    
    def export_summary_csv(self, output_path: str):
        """Export a summary as CSV for batch analysis."""
        stages = self.report.get('stages', {})
        
        rows = []
        rows.append(['metric', 'value'])
        rows.append(['image_path', self.report.get('image_path', 'N/A')])
        rows.append(['prompt_name', self.report.get('prompt_name', 'N/A')])
        rows.append(['overall_status', self.report.get('overall_status', 'N/A')])
        rows.append(['validation_status', self.report.get('validation_status', 'N/A')])
        rows.append(['total_confidence', f"{self.report.get('total_confidence', 0):.3f}"])
        rows.append(['execution_time_seconds', f"{self.report.get('execution_time_seconds', 0):.2f}"])
        rows.append(['num_stages', len(stages)])
        rows.append(['num_contradictions', len(self.report.get('contradictions', []))])
        
        # Add stage confidence scores
        for stage_name, stage_data in stages.items():
            if stage_data.get('status') == 'completed':
                rows.append([f'stage_{stage_name}_confidence', f"{stage_data.get('confidence', 0):.3f}"])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(','.join(str(v) for v in row) + '\n')
        
        print(f"CSV summary saved to: {output_path}")
    
    def print_full_report(self):
        """Print the complete report."""
        self.print_overview()
        self.print_stages()
        self.print_facts()
        self.print_contradictions()
        self.print_validation()
        self.print_confidence_analysis()
        print("\n" + "=" * 100 + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python track_analyzer.py <path_to_tracking_report.json> [--csv <output.csv>]")
        print("\nExample:")
        print("  python track_analyzer.py plots/synthetic/AA/AA.pv2_prompt_1.v1/example.mistral.out_tracking")
        print("  python track_analyzer.py report.json --csv summary.csv")
        sys.exit(1)
    
    report_path = sys.argv[1]
    
    if not os.path.exists(report_path):
        print(f"Error: Report file not found: {report_path}")
        sys.exit(1)
    
    analyzer = TrackingReportAnalyzer(report_path)
    
    # Check for CSV export flag
    if len(sys.argv) > 3 and sys.argv[2] == '--csv':
        csv_output = sys.argv[3]
        analyzer.print_full_report()
        analyzer.export_summary_csv(csv_output)
    else:
        analyzer.print_full_report()


if __name__ == '__main__':
    main()
