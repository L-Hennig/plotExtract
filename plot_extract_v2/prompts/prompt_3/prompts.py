# =====================================================================
# PlotExtract V2 - Prompt Set 3
# Five-stage extraction with plot type verification
# =====================================================================

# Extract stage 1: Plot type verification (time-kill)
EXTRACT_STAGE_1 = r"""
You have access to the complete extraction schema (showing all possible fields).
You will output ONLY the fields you determine, building on previous facts.

COMPLETE SCHEMA:
{complete_schema}

ACCUMULATED FACTS SO FAR:
{accumulated_facts}

---

STAGE 1 — VERIFY PLOT TYPE (TIME-KILL)

Step 1 — Check article_info.experimental_context.assay_type:

If the key contains "time-kill" (case-insensitive), set "declared_in_article_info": true.

If it exists but does NOT contain "time-kill", set "declared_in_article_info": false.

If missing or empty, proceed to Step 2.

Step 2 — Check article_info.figure_caption (secondary source):

Only check if Step 1 returned missing or empty.

If it contains "time-kill" (case-insensitive), set "declared_in_article_info": true.

If exists but does NOT contain "time-kill", set "declared_in_article_info": false.

If missing, set "declared_in_article_info": "unknown".

Step 3 — Visually verify the plot:

Confirm x-axis = time, y-axis = bacterial counts/burden.

Confirm multiple curves with markers and consistent line/marker styles.

Do NOT read numeric values, ticks, or extract points.

Set "visually_verified": true if all indicators match, else false.

Step 4 — Reconcile article info and visual check:

Visual check is decisive.

Set confidence according to agreement:

High (0.9–1.0): article info and visual check agree.

Medium (0.6–0.8): visual true, article info missing.

Low (0.0–0.5): visual true, article info contradicts.

0: visual false → abort.

Abort output if "visually_verified": false:

{{
  "abort": true,
  "reason": "Plot does not appear to be a time-kill plot",
  "confidence": 0
}}

Output JSON (if not aborting):

Merge with accumulated_facts — preserve existing keys.

Add/update only fields you can confidently determine.

{{
  "plot_type_facts": {{
    "declared_in_article_info": true | false | "unknown",
    "visually_verified": true | false,
    "plot_type": "time-kill plot",
    "confidence": 0-1,
    "reason": "short explanation of decision"
  }}
}}
"""

# Extract stage 2: X-axis verification (time in hours)
EXTRACT_STAGE_2 = r"""
You have access to the complete extraction schema (showing all possible fields) and accumulated facts.

COMPLETE SCHEMA:
{complete_schema}

ACCUMULATED FACTS SO FAR:
{accumulated_facts}

---

STAGE 2 — VERIFY X-AXIS (TIME IN HOURS)

Step 1 — Verify axis quantity and unit:

Visually confirm x-axis represents time in hours.

Cross-check article_info.axis_definitions.x_axis.quantity and .unit if present.

Any contradiction or uncertainty → abort.

Step 2 — Verify x-axis range:

Confirm lower = 0, upper ∈ [12,36]. Prefer 24–26 for higher confidence.

Cross-check article_info.axis_definitions.x_axis.range.

Any major disagreement → abort.

Step 3 — Extract ticks, labels, breaks:

Extract ticks_verified and tick_labels.

Determine discrete vs continuous.

Record axis breaks if present (start, end, reason), else omit.

Step 4 — Assign confidence:

Based on clarity, consistency, and agreement with article info.

Abort output if any hard check fails:

{{
  "abort": true,
  "reason": "X-axis is not verifiable as time in hours with valid range",
  "confidence": 0.0-1.0
}}

Output JSON (merge with accumulated facts):

{{
  "axis_facts": {{
    "x_axis": {{
      "quantity": "time",
      "unit": "hours",
      "discrete": true | false,
      "lower_plausible_limit": 0,
      "upper_plausible_limit": 26,
      "ticks_verified": [],
      "tick_labels": [],
      "breaks": [],
      "confidence": 0-1
    }}
  }}
}}
"""

# Extract stage 3: Y-axis verification (bacterial burden, log10)
EXTRACT_STAGE_3 = r"""
You have access to the complete extraction schema (showing all possible fields) and accumulated facts.

COMPLETE SCHEMA:
{complete_schema}

ACCUMULATED FACTS SO FAR:
{accumulated_facts}

---

STAGE 3 — VERIFY Y-AXIS (BACTERIAL BURDEN, LOG10)

Step 1 — Verify quantity and unit:

Visually confirm y-axis = bacterial burden / CFU-based unit.

Cross-check article_info.axis_definitions.y_axis.quantity and .unit.

Contradictions → abort.

Step 2 — Verify log10 scaling:

Inspect tick spacing and labels for logarithmic spacing.

Cross-check article_info.axis_definitions.y_axis.scale.

Any failure → abort.

Step 3 — Verify plausible range:

Confirm lower ≥ 0, upper ≤ 9 (unless explicitly stated).

Cross-check with article info.

Step 4 — Extract ticks, labels, breaks:

ticks_verified, tick_labels, axis breaks (start, end, reason).

Step 5 — Assign confidence (0–1) based on clarity, consistency, and agreement.

Abort output if any hard requirement fails:

{{
  "abort": true,
  "reason": "Y-axis log10 verification failed (ticks inconsistent or implausible)",
  "confidence": 0.0-1.0
}}

Output JSON (merge with accumulated facts):

{{
  "axis_facts": {{
    "y_axis": {{
      "quantity": "bacterial burden",
      "unit": "log10 CFU/mL",
      "scale": "log10",
      "upper_plausible_limit": 9,
      "lower_plausible_limit": 0,
      "ticks_verified": [],
      "tick_labels": [],
      "breaks": [],
      "confidence": 0-1
    }}
  }}
}}
"""

# Extract stage 4: Data point extraction (CSV + JSON)
EXTRACT_STAGE_4 = r"""
You have access to the complete extraction schema and accumulated facts (including axes and plot type).

COMPLETE SCHEMA:
{complete_schema}

ACCUMULATED FACTS SO FAR:
{accumulated_facts}

---

STAGE 4 — DATA POINT EXTRACTION (CSV + JSON)

Step 1 — Preconditions:

X and Y axes labeled and monotonic.

Tickmarks are numeric and ordered.

Numeric values can be assigned to each tickmark.

If preconditions fail → abort.

Step 2 — Extraction rules:

Look at each point individually; no interpolation or estimation.

Use only existing x-axis ticks as candidate x-values.

For each curve at each tick:

If a plotted point exists → extract y.

If missing → omit.

Step 3 — Output construction:

Output CSV (WIDE FORMAT - critical):

IMPORTANT: Use WIDE format (multiple X,Y column pairs side-by-side), NOT long format.

For each curve, create TWO consecutive columns: [X_label] and [Y_label (Curve Name)].

Header row: x_col_1, y_col_1, x_col_2, y_col_2, x_col_3, y_col_3, etc.

Data rows: x1, y1, x2, y2, x3, y3, etc. (one row per x-value)

Missing y-values in any curve: omit that curve for that x-value by leaving blank.

Example: If 3 curves at x=0: "Time (hours),log10 CFU/mL (Condition 1),Time (hours),log10 CFU/mL (Condition 2),Time (hours),log10 CFU/mL (Condition 3)"

Each row after header: "0,0.1,0,0.2,0,0.3" (one x repeated for each curve with its corresponding y).

Output JSON:

Include accumulated facts, marker_facts with extracted points.

Abort output if extraction is impossible:

{{
  "abort": true,
  "reason": "Data extraction impossible due to missing labels or unreadable points",
  "confidence": 0-1
}}

Example JSON output:

{{
  "marker_facts": {{
    "markers_detected": true | false,
    "curves": [
      {{
        "curve_label": "Condition 1",
        "points": [{{"x": 0, "y": 0.1}}, {{"x": 2, "y": 0.5}}]
      }}
    ],
    "csv_output": "Time (hours),log10 CFU/mL (Condition 1),Time (hours),log10 CFU/mL (Condition 2)\\n0,0.1,0,0.2\\n2,0.5,2,0.6",
    "confidence": 0-1
  }}
}}
"""

# Extract stage 5: Validation / sanity checks
EXTRACT_STAGE_5 = r"""
Inputs:

CSV from Stage 4 (contained in accumulated_facts)

JSON with accumulated facts

COMPLETE SCHEMA:
{complete_schema}

ACCUMULATED FACTS SO FAR:
{accumulated_facts}

---

STAGE 5 — VALIDATION / SANITY CHECKS

Step 1 — Impossible jumps / sudden changes:

Flag consecutive points >1–2 log10 CFU/mL change.

Step 2 — Unexpected monotonicity:

Compare points to expected_trend in curve_legend.

Flag points strongly contradicting overall trend.

Step 3 — Out-of-bound values:

Flag y outside y_axis.upper_plausible_limit/lower_plausible_limit.

Flag x outside x_axis.lower_plausible_limit/upper_plausible_limit.

Step 4 — Duplicates / overlapping points:

Flag points with identical x and y values.

Step 5 — Trend divergence check:

Flag large deviations from expected trend without modifying data.

Optional abort if CSV or JSON invalid:

{{
  "abort": true,
  "reason": "CSV or JSON input invalid for validation",
  "confidence": 0-1
}}

Output JSON:

You MUST output the complete marker_facts object from Stage 4 (found in accumulated_facts) exactly as-is, plus validation_flags:

{{
  "marker_facts": {{
    "markers_detected": <boolean from Stage 4>,
    "curves": <array from Stage 4>,
    "csv_output": "<EXACT csv_output string from Stage 4 marker_facts - do not modify>",
    "confidence": <number from Stage 4>
  }},
  "validation_flags": [
    {{
      "curve_label": "Condition 1",
      "x": 4,
      "y": 7.5,
      "issue": "Impossible jump"
    }}
  ]
}}

CRITICAL: Copy the marker_facts.csv_output field from accumulated_facts EXACTLY. Do not regenerate it.
"""

# Code generation stage: Create matplotlib code to replot
CODE_PLOT = r"""
You are provided with:
- The same plot image.
- Extracted data and facts:
{accumulated_facts}

Generate Python (matplotlib) code that exactly replots the figure, matching styles, colors, markers, axis labels, limits, ticks, and legend.
- Save the plot only to: {replot_path}
- Do not display the plot.
- Respond with code only so it can be executed directly.

IMPORTANT: If the y-axis scale in axis_facts is "log10", the data values are ALREADY in log10 units.
- Do NOT use plt.yscale('log')
- Use a LINEAR scale (normal plotting)
- The axis label should indicate "log10 scale" or "(log10)" to show units are logarithmic
"""

# Code repair prompt
CODE_FIX = (
    "The text above is an error produced by your code. "
    "Rewrite the full corrected code only; no explanations; the response must be executable as-is."
)

# Validation prompts
COMPARE_X = "You are provided with two images of research plots extracted from academic papers. Do these two plots have the same x-axis (horizontal)? Do they have the same ranges, labels, etc.? Answer with a single word, \"yes\" or \"no\" only."
COMPARE_Y = "You are provided with two images of research plots extracted from academic papers. Do these two plots have the same y-axis (vertical)? Do they have the same ranges, labels, etc.? Answer with a single word, \"yes\" or \"no\" only."
COMPARE_NUMBER = "You are provided with two images of research plots extracted from academic papers. Do these two plots have the same number of points (for point plots)? Do the curves look like they connect the same amount of points (for line plots)? Answer with a single word, \"yes\" or \"no\" only."
COMPARE_TREND = "You are provided with two images of research plots extracted from academic papers. Do these sets of points or curves on these two plots represent the same trends? Do they follow the same patterns? Are points distributed in the same way? Answer with a single word, \"yes\" or \"no\" only."
