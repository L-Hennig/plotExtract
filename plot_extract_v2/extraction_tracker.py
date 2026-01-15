"""
ExtractionTracker: Tracks progress, confidence, facts, and contradictions during plot extraction.
"""
import json
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class StageProgress:
    """Track progress for a single extraction stage."""
    stage_name: str
    status: str = "pending"  # pending, running, completed, failed, skipped
    confidence: float = 0.0  # 0.0 to 1.0
    facts_extracted: Dict[str, Any] = field(default_factory=dict)
    output_length: int = 0
    execution_time_ms: float = 0.0
    error: Optional[str] = None
    validation_result: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: List[str] = field(default_factory=list)


@dataclass
class Contradiction:
    """Track contradictions found during extraction."""
    stage_1: str
    stage_2: str
    field_name: str
    value_1: Any
    value_2: Any
    severity: str = "warning"  # warning, error, critical
    resolved: bool = False
    resolution: Optional[str] = None


class ExtractionTracker:
    """Main tracker for monitoring extraction progress."""
    
    def __init__(self, image_path: str, prompt_name: str):
        self.image_path = image_path
        self.prompt_name = prompt_name
        self.start_time = datetime.now()
        
        # Stage tracking
        self.stages: Dict[str, StageProgress] = {}
        self.stage_order: List[str] = []
        
        # Fact tracking
        self.facts: Dict[str, Dict[str, Any]] = {}  # stage_name -> extracted facts
        self.contradictions: List[Contradiction] = []
        
        # Overall metrics
        self.total_confidence: float = 0.0
        self.validation_status: str = "pending"  # pending, passed, failed, skipped
        self.overall_status: str = "running"  # running, completed, failed
        
        # Output files generated
        self.output_files: Dict[str, str] = {}
    
    def initialize_stages(self, stage_names: List[str]):
        """Initialize tracking for all stages."""
        self.stage_order = stage_names
        for stage_name in stage_names:
            self.stages[stage_name] = StageProgress(stage_name=stage_name)
    
    def start_stage(self, stage_name: str):
        """Mark a stage as started."""
        if stage_name not in self.stages:
            self.stages[stage_name] = StageProgress(stage_name=stage_name)
        self.stages[stage_name].status = "running"
        self.stages[stage_name].timestamp = datetime.now().isoformat()
        print(f"[TRACKER] Starting stage: {stage_name}")
    
    def complete_stage(self, stage_name: str, output_text: str, confidence: float = 0.5,
                       execution_time_ms: float = 0.0, facts: Optional[Dict[str, Any]] = None):
        """Mark a stage as completed and record its output."""
        if stage_name not in self.stages:
            self.stages[stage_name] = StageProgress(stage_name=stage_name)
        
        stage = self.stages[stage_name]
        stage.status = "completed"
        stage.output_length = len(output_text)
        stage.confidence = max(0.0, min(1.0, confidence))  # Clamp to 0-1
        stage.execution_time_ms = execution_time_ms
        
        if facts:
            stage.facts_extracted = facts
            self.facts[stage_name] = facts
            print(f"[TRACKER] Stage '{stage_name}' extracted facts: {list(facts.keys())}")
        
        self._analyze_output(stage_name, output_text)
        print(f"[TRACKER] Completed stage '{stage_name}' (confidence: {stage.confidence:.2f})")
    
    def fail_stage(self, stage_name: str, error: str, execution_time_ms: float = 0.0):
        """Mark a stage as failed."""
        if stage_name not in self.stages:
            self.stages[stage_name] = StageProgress(stage_name=stage_name)
        
        stage = self.stages[stage_name]
        stage.status = "failed"
        stage.error = error
        stage.execution_time_ms = execution_time_ms
        self.overall_status = "failed"
        print(f"[TRACKER] Stage '{stage_name}' FAILED: {error}")
    
    def skip_stage(self, stage_name: str, reason: str = ""):
        """Mark a stage as skipped."""
        if stage_name not in self.stages:
            self.stages[stage_name] = StageProgress(stage_name=stage_name)
        
        stage = self.stages[stage_name]
        stage.status = "skipped"
        stage.notes.append(f"Skipped: {reason}")
        print(f"[TRACKER] Stage '{stage_name}' skipped: {reason}")
    
    def _analyze_output(self, stage_name: str, output_text: str):
        """Analyze output for potential issues."""
        output_lower = output_text.lower().strip()
        
        # Check for "None" response
        if output_lower == "none":
            stage = self.stages[stage_name]
            stage.notes.append("Output is 'None' - no data extracted")
            stage.confidence = 0.0
    
    def add_contradiction(self, stage_1: str, stage_2: str, field_name: str,
                         value_1: Any, value_2: Any, severity: str = "warning"):
        """Record a contradiction between two stages."""
        contradiction = Contradiction(
            stage_1=stage_1,
            stage_2=stage_2,
            field_name=field_name,
            value_1=value_1,
            value_2=value_2,
            severity=severity
        )
        self.contradictions.append(contradiction)
        print(f"[TRACKER] CONTRADICTION: {stage_1}.{field_name} != {stage_2}.{field_name}")
        print(f"          Value 1: {value_1}, Value 2: {value_2} (severity: {severity})")
    
    def extract_facts_from_csv(self, csv_data: str, stage_name: str = "extraction") -> Dict[str, Any]:
        """Extract facts from CSV data."""
        facts = {}
        lines = csv_data.strip().split('\n')
        
        if len(lines) < 2:
            return facts
        
        try:
            header = lines[0].split(',')
            facts['num_columns'] = len(header)
            facts['num_rows'] = len(lines) - 1
            facts['columns'] = header
            facts['axis_labels'] = header
        except Exception as e:
            print(f"[TRACKER] Error extracting facts from CSV: {e}")
        
        return facts
    
    def check_axis_consistency(self, axis_name: str, expected_label: str, stage_name: str):
        """Check if axis labels are consistent across stages."""
        if stage_name not in self.facts:
            return
        
        facts = self.facts[stage_name]
        if 'axis_labels' in facts:
            labels = facts['axis_labels']
            # This can be extended with more sophisticated checks
            print(f"[TRACKER] Axis '{axis_name}' has {len(labels)} labels in stage '{stage_name}'")
    
    def set_validation_result(self, result: str, details: Optional[Dict[str, Any]] = None):
        """Set the validation result."""
        self.validation_status = result  # passed, failed, skipped
        if details:
            self.output_files['validation_details'] = details
        print(f"[TRACKER] Validation result: {result}")
        if details:
            print(f"          Details: {details}")
    
    def mark_complete(self):
        """Mark extraction as complete."""
        self.overall_status = "completed"
        print(f"[TRACKER] Extraction completed successfully")
    
    def get_confidence_summary(self) -> float:
        """Calculate overall confidence as average of stage confidences."""
        if not self.stages:
            return 0.0
        
        completed_stages = [s for s in self.stages.values() if s.status == "completed"]
        if not completed_stages:
            return 0.0
        
        avg_confidence = sum(s.confidence for s in completed_stages) / len(completed_stages)
        self.total_confidence = avg_confidence
        return avg_confidence
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all tracking data."""
        return {
            'image_path': self.image_path,
            'prompt_name': self.prompt_name,
            'overall_status': self.overall_status,
            'validation_status': self.validation_status,
            'total_confidence': self.get_confidence_summary(),
            'stages': {name: asdict(stage) for name, stage in self.stages.items()},
            'facts_by_stage': self.facts,
            'contradictions': [
                {
                    'stage_1': c.stage_1,
                    'stage_2': c.stage_2,
                    'field_name': c.field_name,
                    'value_1': str(c.value_1),
                    'value_2': str(c.value_2),
                    'severity': c.severity,
                    'resolved': c.resolved
                }
                for c in self.contradictions
            ],
            'execution_time_seconds': (datetime.now() - self.start_time).total_seconds(),
            'output_files': self.output_files,
        }
    
    def save_tracking_report(self, output_path: str):
        """Save tracking report to file."""
        summary = self.get_summary()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[TRACKER] Report saved to {output_path}")
        return output_path
    
    def print_summary(self):
        """Print a human-readable summary."""
        print("\n" + "=" * 80)
        print("EXTRACTION PROGRESS SUMMARY")
        print("=" * 80)
        
        print(f"\nImage: {self.image_path}")
        print(f"Prompt: {self.prompt_name}")
        print(f"Status: {self.overall_status}")
        print(f"Validation: {self.validation_status}")
        print(f"Overall Confidence: {self.get_confidence_summary():.2%}")
        
        print("\n--- STAGE RESULTS ---")
        for stage_name in self.stage_order:
            if stage_name in self.stages:
                stage = self.stages[stage_name]
                confidence_display = f"{stage.confidence:.1%}" if stage.confidence > 0 else "N/A"
                print(f"  {stage_name:30} [{stage.status:10}] confidence: {confidence_display:6}")
                if stage.error:
                    print(f"    └─ Error: {stage.error[:60]}")
                if stage.notes:
                    for note in stage.notes:
                        print(f"    └─ {note}")
        
        if self.contradictions:
            print("\n--- CONTRADICTIONS ---")
            for c in self.contradictions:
                severity_tag = f"[{c.severity.upper()}]"
                print(f"  {severity_tag} {c.stage_1} vs {c.stage_2}")
                print(f"    {c.field_name}: '{c.value_1}' != '{c.value_2}'")
                if c.resolved:
                    print(f"    Resolution: {c.resolution}")
        else:
            print("\n--- CONTRADICTIONS ---")
            print("  None detected")
        
        if self.facts:
            print("\n--- EXTRACTED FACTS ---")
            for stage_name, facts in self.facts.items():
                print(f"  {stage_name}:")
                for key, value in facts.items():
                    value_str = str(value)
                    if len(value_str) > 50:
                        value_str = value_str[:47] + "..."
                    print(f"    - {key}: {value_str}")
        
        exec_time = (datetime.now() - self.start_time).total_seconds()
        print(f"\nExecution time: {exec_time:.2f}s")
        print("=" * 80 + "\n")
