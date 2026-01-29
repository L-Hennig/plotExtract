# =====================================================================
# Complete Extraction Schema
# This schema defines the structure for all extraction stages:
# - article_info: Preprocessed article context
# - plot_type_facts: Stage 1 time-kill verification outputs
# - axis_facts: Stage 2 & 3 axis + grid verification outputs
# - geometry_facts: Stage 3A geometry pixel anchors for UI overlay
# - curve_style_facts: Stage 4 curve appearance metadata
# - marker_facts: Stage 5 extraction output (points + CSV)
# - re_extraction: Stage 6–8 correction loop (requests + results)
# =====================================================================

ACCUMULATED_FACTS_SCHEMA = """{
  "article_info": {
    "figure_id": "string",
    "figure_caption": "string",
    "experimental_context": {
      "assay_type": "string",
      "organism": "string",
      "model": "string"
    },
    "axis_definitions": {
      "x_axis": {
        "quantity": "string",
        "unit": "string"
      },
      "y_axis": {
        "quantity": "string",
        "unit": "string",
        "scale": "string"
      }
    },
    "curve_legend": [
      {
        "curve_label": "string",
        "description": "string",
        "expected_trend": "string"
      }
    ]
  },
  "plot_type_facts": {
    "declared_in_article_info": "boolean or unknown",
    "visually_verified": "boolean",
    "plot_type": "string",
    "confidence": "0-1",
    "reason": "string"
  },
  "axis_facts": {
    "x_axis": {
      "quantity": "string",
      "unit": "string",
      "discrete": "boolean",
      "ticks_verified": ["number array"],
      "tick_labels": ["string array"],
      "breaks": [
        {
          "start": "number",
          "end": "number",
          "reason": "string"
        }
      ],
      "grid": {
        "present": "boolean",
        "aligned_ticks": ["number array"]
      },
      "confidence": "0-1"
    },
    "y_axis": {
      "quantity": "string",
      "unit": "string",
      "scale": "string",
      "upper_plausible_limit": "number",
      "lower_plausible_limit": "number",
      "ticks_verified": ["number array"],
      "tick_labels": ["string array"],
      "breaks": [
        {
          "start": "number",
          "end": "number",
          "reason": "string"
        }
      ],
      "grid": {
        "present": "boolean",
        "aligned_ticks": ["number array"]
      },
      "confidence": "0-1"
    }
  },
  "geometry_facts": {
    "plot_box_px": {
      "left": "integer",
      "top": "integer",
      "right": "integer",
      "bottom": "integer"
    },
    "x_axis_geometry": {
      "scale": "linear",
      "anchors": [
        {
          "x_value_hours": "number",
          "x_px": "integer"
        }
      ],
      "x_break": {
        "present": "boolean",
        "break_px_start": "integer",
        "break_px_end": "integer",
        "left_end_value_hours": "number",
        "right_start_value_hours": "number",
        "left_segment_anchors": [
          {
            "x_value_hours": "number",
            "x_px": "integer"
          }
        ],
        "right_segment_anchors": [
          {
            "x_value_hours": "number",
            "x_px": "integer"
          }
        ]
      }
    },
    "y_axis_geometry": {
      "scale": "log10",
      "decade_anchors": [
        {
          "log10_value": "number",
          "y_px": "integer"
        }
      ],
      "y0_tick": {
        "present": "boolean",
        "y_px": "integer"
      }
    },
    "confidence": "0-1"
  },
  "curve_style_facts": {
    "curves": [
      {
        "curve_label": "string",
        "colour": "string",
        "line_type": "string",
        "marker_symbol": "string",
        "source": "article_text | figure_legend | direct_visual",
        "confidence": "0-1"
      }
    ]
  },
  "marker_facts": {
    "markers_detected": "boolean",
    "curves": [
      {
        "curve_label": "string",
        "points": [
          {
            "x": "number",
            "y": "number",
            "confidence": "0-1"
          }
        ]
      }
    ],
    "csv_output": "string",
    "confidence": "0-1"
  },
  "re_extraction": {
    "requests": [
      {
        "curve_label": "string",
        "issue_type": "truncated_curve | bad_intercept_time0 | direction_flip_full | direction_suspicious_unknown_expectation | midcurve_switch | midcurve_switch_suspicious_unknown_expectation",
        "target_times_hours": ["number array"],
        "expected_trend": "kill/decrease | growth/increase | unknown",
        "evidence": {
          "n_points": "number",
          "expected_n_points": "number",
          "times_missing": ["number array"],
          "pct_increasing": "0-1",
          "pct_decreasing": "0-1",
          "baseline_delta_from_median": "number or null",
          "largest_jump": {
            "abs_jump": "number or null",
            "from_time": "number or null",
            "to_time": "number or null",
            "delta": "number or null"
          }
        },
        "visual_guidance": "string"
      }
    ],
    "results": [
      {
        "curve_label": "string",
        "time_hours": "number",
        "new_log10_cfu_ml": "number",
        "confidence": "high | medium | low",
        "reason": "string",
        "checks": {
          "used_curve_definition": "boolean",
          "matches_expected_trend_if_known": "boolean",
          "avoids_previous_issue": "string"
        }
      }
    ]
  }
}"""

SCHEMA_CONSTRAINTS = """Additional constraints:
- `experimental_context` is optional.
- `axis_definitions` should only be filled if the text explicitly defines them.
- `expected_trend` is optional and must come from explicit textual statements.
- If no curve legend is present, omit `curve_legend` entirely.
- Every curve in `curve_style_facts.curves` must have `colour`, `line_type`, and `marker_symbol`.
- If `grid.present` is true, `aligned_ticks` must not be empty.
- If grid lines are present but do not align with valid axis ticks, extraction must abort.
- `geometry_facts` is optional. Omit it entirely unless Stage 3A runs successfully.
- In `geometry_facts`:
  - `plot_box_px` must be present with integers and satisfy left < right and top < bottom.
  - `x_axis_geometry.anchors` must contain exactly 4 entries when present (min, max, and two mids).
  - `y_axis_geometry.decade_anchors` must contain exactly 4 entries when present (min, max, and two mids).
  - Each x anchor value MUST come from `axis_facts.x_axis.ticks_verified`.
  - Each y anchor log10_value MUST come from `axis_facts.y_axis.ticks_verified`.
  - If `x_axis_geometry.x_break.present` is false, omit all other `x_break` fields.
  - If `x_axis_geometry.x_break.present` is true, then:
    - `left_end_value_hours` MUST equal `axis_facts.x_axis.breaks[0].start`
    - `right_start_value_hours` MUST equal `axis_facts.x_axis.breaks[0].end`
    - `break_px_start` and `break_px_end` must be present
    - `left_segment_anchors` must have exactly 2 entries with x_value_hours < left_end_value_hours
    - `right_segment_anchors` must have exactly 2 entries with x_value_hours > right_start_value_hours
  - If `y_axis_geometry.y0_tick.present` is false, omit `y0_tick.y_px`.
- `re_extraction` is optional. Omit it entirely unless Stage 6 produces at least one request.
- If `re_extraction.requests` is present, it must not be empty.
- If `re_extraction.results` is present, it must not be empty.
- `re_extraction.results` must only include (curve_label, time_hours) pairs that appear in `re_extraction.requests.target_times_hours`.
- Do not add empty objects or empty arrays."""
