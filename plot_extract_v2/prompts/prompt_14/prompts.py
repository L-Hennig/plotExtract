# =====================================================================
# PlotExtract V2 - Prompt 13
# copy of prompt 12 witgh modifications
# Added reinforcments to not mistake dotted line for points in data extraction stage
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

Do not write anything other than the explicit facts that have a specific place in the JSON schema. Do NOT add any extra commentary or explanation. Do NOT use any triple backticks.

Merge with accumulated_facts — preserve existing keys.

Add/update only fields you can confidently determine.

Before moving on the the next stage ensure that any triple backticks have been removed from your response.

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

Do not write anything other than the explicit facts that have a specific place in the JSON schema. Do NOT add any extra commentary or explanation. Do NOT use any triple backticks.

Before moving on the the next stage ensure that any triple backticks have been removed from your response.

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

# Extract stage 3: Y-axis verification (bacterial burden, log10) + grid alignment
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

Step 4 — Verify grid presence and alignment:

Determine whether grid lines are present (horizontal and/or vertical).

If no grid is present:
- Set grid.present = false.
- Skip alignment checks.

If a grid is present:
- Set grid.present = true.
- Verify grid lines align exactly with x-axis and y-axis tick positions.
- Identify which axis ticks each grid line corresponds to.
- Record aligned tick values for both axes.

If any grid line does not align with a valid axis tick:
- Abort extraction (grid–tick inconsistency).

Step 5 — Extract ticks, labels, breaks:

Extract ticks_verified and tick_labels.

Record axis breaks if present (start, end, reason), else omit.

Step 6 — Assign confidence (0–1):

Based on clarity, consistency, and agreement with article info.

Abort output if any hard requirement fails:

{{
  "abort": true,
  "reason": "Y-axis log10 verification failed (ticks inconsistent or implausible)",
  "confidence": 0.0-1.0
}}

Output JSON (merge with accumulated facts):

Do not write anything other than the explicit facts that have a specific place in the JSON schema. Do NOT add any extra commentary or explanation. Do NOT use any triple backticks.

Before moving on the next stage ensure that any triple backticks have been removed from your response.

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
      "grid": {{
        "present": true | false,
        "aligned_y_ticks": [],
        "aligned_x_ticks": []
      }},
      "confidence": 0-1
    }}
  }}
}}
"""

# Extract stage 3A: Geometry extraction (pixel anchors for overlay)
EXTRACT_STAGE_3A = r"""
You have access to the complete extraction schema (showing all possible fields) and accumulated facts.

COMPLETE SCHEMA:
{complete_schema}

ACCUMULATED FACTS SO FAR:
{accumulated_facts}

---

STAGE 3A — EXTRACT GEOMETRY FACTS (PIXEL ANCHORS FOR OVERLAY)

PURPOSE:
Create a pixel↔data mapping so the UI can overlay extracted curves on the ORIGINAL image in the correct place.
You MUST use the VERIFIED ticks already stored in accumulated facts:
- axis_facts.x_axis.ticks_verified
- axis_facts.y_axis.ticks_verified
- axis_facts.x_axis.breaks (if any)

HARD RULES:
- Output JSON only. No commentary. No backticks.
- Do NOT read any curve point values.
- Do NOT invent ticks. Use ONLY ticks_verified arrays.
- For each anchor you output, you MUST provide:
  - the tick value (data units)
  - the pixel coordinate of the corresponding tick mark/gridline at the plot area boundary
- If any required anchor cannot be located confidently → abort.

IMAGE COORDINATE SYSTEM:
- origin (0,0) is the TOP-LEFT of the image
- x increases to the RIGHT
- y increases DOWN

---

Step 0 — Preconditions (abort if any fail):
- axis_facts.x_axis.ticks_verified exists and has >= 4 values.
- axis_facts.y_axis.ticks_verified exists and has >= 4 values.
- axis_facts.y_axis.scale indicates log10 behaviour (per Stage 3).
If any fail → abort.

---

Step 1 — Determine the plot area bounding box (plot_box_px):
Find the rectangular plotting region bounded by the axes spines and/or the outermost gridlines.
Return:
- left, top, right, bottom (all integers, left < right, top < bottom)

This box MUST tightly bound the region where the data markers appear.
Abort if the plot box cannot be confidently identified.

---

Step 2 — Select REQUIRED x-axis anchor tick values (from axis_facts.x_axis.ticks_verified):
Let X = axis_facts.x_axis.ticks_verified (sorted ascending).
Select exactly 4 tick values:
- x_min_value = X[0]
- x_max_value = X[-1]
- x_mid1_value = X[floor((len(X)-1) * 1/3)]
- x_mid2_value = X[floor((len(X)-1) * 2/3)]

If any selected mid duplicates min/max due to small length, choose the nearest distinct ticks instead.

For each selected x tick value:
- Locate the tick mark or vertical gridline corresponding to that tick.
- Record the pixel x-position at the bottom edge of the plot area (y = plot_box_px.bottom).
Output as anchors: { "x_value_hours": ..., "x_px": ... }

Do NOT use axis break positions as anchors.

---

Step 3 — Select REQUIRED y-axis anchor tick values (from axis_facts.y_axis.ticks_verified):
Let Y = axis_facts.y_axis.ticks_verified (sorted ascending).
Select exactly 4 tick values:
- y_min_value = Y[0]
- y_max_value = Y[-1]
- y_mid1_value = Y[floor((len(Y)-1) * 1/3)]
- y_mid2_value = Y[floor((len(Y)-1) * 2/3)]

If any selected mid duplicates min/max due to small length, choose the nearest distinct ticks instead.

For each selected y tick value:
- Locate the tick mark or horizontal gridline corresponding to that tick.
- Record the pixel y-position at the left edge of the plot area (x = plot_box_px.left).
Output as anchors in decade_anchors:
- If tick labels represent decades, store log10_value consistent with axis_facts.y_axis.ticks_verified meaning.
  (Use the numeric values in ticks_verified directly; do not re-interpret them.)

---

Step 4 — Handle explicit y=0 tick (if present visually):
Independently check whether the y-axis shows an explicit tick labeled "0".
If present:
- record y0_tick.present = true
- record y0_tick.y_px as the pixel y-position of that tick mark/gridline at x = plot_box_px.left
If not present:
- record y0_tick.present = false (omit y_px)

IMPORTANT:
Do NOT convert y=0 into log units. This is a special visual anchor only.

---

Step 5 — Handle x-axis break (MUST follow accumulated facts):
Check axis_facts.x_axis.breaks.

If axis_facts.x_axis.breaks is missing OR empty:
- set x_break.present = false
- omit all other x_break fields

If axis_facts.x_axis.breaks has >= 1 entry:
- set x_break.present = true
- Use ONLY the first break entry for geometry.
- Read break.start and break.end from axis_facts.x_axis.breaks[0]
  These are the data-unit boundary values already determined earlier.
- Locate the visual break symbol/gap on the x-axis and record:
  - break_px_start and break_px_end (integer pixel x-span of the break on the axis line)
- Create segment anchors:
  - left_segment_anchors: select 2 anchors from the LEFT side of the break:
      use {x_min_value, x_mid1_value} if both are < break.start, otherwise choose the two largest ticks < break.start.
  - right_segment_anchors: select 2 anchors from the RIGHT side of the break:
      use {x_mid2_value, x_max_value} if both are > break.end, otherwise choose the two smallest ticks > break.end.
For each segment anchor tick:
- record { "x_value_hours": ..., "x_px": ... } the same way as Step 2.

Abort if:
- break is declared in axis_facts but you cannot locate the break in pixels, OR
- you cannot find 2 anchors on each side.

---

Step 6 — Confidence (0–1):
Assign confidence based on:
- clarity of plot_box_px
- ease of locating tick/grid intersections for all required anchors
- consistency with visible grid (if grid.present true)

---

Abort JSON:
{
  "abort": true,
  "reason": "Stage 3A geometry extraction failed (plot box or required anchors/break not confidently located)",
  "confidence": 0
}

---

Output JSON (merge with accumulated facts; preserve existing keys):
Do NOT add empty arrays/objects. Omit fields you cannot determine.

{
  "geometry_facts": {
    "plot_box_px": { "left": 0, "top": 0, "right": 0, "bottom": 0 },
    "x_axis_geometry": {
      "scale": "linear",
      "anchors": [
        { "x_value_hours": 0.0, "x_px": 0 },
        { "x_value_hours": 0.0, "x_px": 0 },
        { "x_value_hours": 0.0, "x_px": 0 },
        { "x_value_hours": 0.0, "x_px": 0 }
      ],
      "x_break": {
        "present": true | false,
        "break_px_start": 0,
        "break_px_end": 0,
        "left_end_value_hours": 0.0,
        "right_start_value_hours": 0.0,
        "left_segment_anchors": [
          { "x_value_hours": 0.0, "x_px": 0 },
          { "x_value_hours": 0.0, "x_px": 0 }
        ],
        "right_segment_anchors": [
          { "x_value_hours": 0.0, "x_px": 0 },
          { "x_value_hours": 0.0, "x_px": 0 }
        ]
      }
    },
    "y_axis_geometry": {
      "scale": "log10",
      "decade_anchors": [
        { "log10_value": 0.0, "y_px": 0 },
        { "log10_value": 0.0, "y_px": 0 },
        { "log10_value": 0.0, "y_px": 0 },
        { "log10_value": 0.0, "y_px": 0 }
      ],
      "y0_tick": {
        "present": true | false,
        "y_px": 0
      }
    },
    "confidence": 0-1
  }
}
"""
# Extract stage 4: Curve definitions (style metadata)
EXTRACT_STAGE_4 = r"""
You have access to the complete extraction schema and accumulated facts.

COMPLETE SCHEMA:
{complete_schema}

ACCUMULATED FACTS SO FAR:
{accumulated_facts}

---

STAGE 4 — SET CURVE DEFINITIONS (STYLE METADATA)

GOAL:
Populate curve style metadata (colour, line_type, marker_symbol) for every curve and store it in curve_style_facts.

This stage defines curve identity for all downstream extraction.
Curve styles MUST be final and authoritative after this stage.

---

Step 1 — Check article info first (author-declared)

Inspect article_info, including:
- figure caption
- curve legend descriptions
- any explicit textual references to curve appearance

Extract ONLY explicitly stated attributes:
- colour
- line type (e.g. solid, dashed, dotted)
- marker symbol (e.g. circle, square, triangle, none)

Record source as "article_text".

Do NOT infer, guess, or normalise wording.

---

Step 2 — Inspect figure legend (visual legend)

For any missing attributes:
- Examine the visual legend symbols.
- Match each legend entry to its curve label.
- Extract colour, line type, and marker symbol.

Cross-check against article text.
Any contradiction → abort.

Record source as "figure_legend".

---

Step 3 — Inspect curves directly in the graph

For any remaining missing attributes:
- Visually inspect the plotted curves.
- Determine:
  - colour from stroke
  - line type from stroke pattern
  - marker symbol from point shapes

Ensure consistency across the entire curve.
Record source as "direct_visual".

---

Step 4 — Validate completeness and consistency

Every curve MUST have:
- colour
- line type
- marker symbol (or explicit "none")

If any attribute is missing → abort.

If multiple curves share identical styles:
- This must be explicitly confirmed visually.
- Otherwise → abort.

---

Step 5 — Assign confidence (0–1)

Base confidence on:
- agreement between sources
- legend clarity
- visual distinctness of curves

---

Abort output if any hard requirement fails:

{{
  "abort": true,
  "reason": "Curve style metadata incomplete or inconsistent (colour / line type / marker symbol)",
  "confidence": 0.0-1.0
}}

---

Output JSON (merge with accumulated facts only):

Populate curve_style_facts exactly as follows:

{{
  "curve_style_facts": {{
    "curves": [
      {{
        "curve_label": "string",
        "colour": "string",
        "line_type": "string",
        "marker_symbol": "string",
        "source": "article_text" | "figure_legend" | "direct_visual",
        "confidence": 0-1
      }}
    ]
  }}
}}

---

STRICT OUTPUT RULES:
Do NOT add commentary or explanations.
Do NOT invent attributes.
Merge with accumulated_facts — preserve existing keys.
Add/update only fields you can confidently determine.
Do NOT use triple backticks in the final model output.
Do NOT add empty objects or empty arrays.
Remove any triple backticks before proceeding to the next stage.

VERY IMPORTANT: Before moving on the the next stage ensure that any triple backticks have been removed from your response.
"""


# Stage 6: Extraction evaluation (uses Stage 6a diagnostics)
EXTRACT_STAGE_6 = r"""
You have access to:
- Accumulated facts (includes plot metadata and, when available, expected trend per curve)
- The original extracted CSV (wide format)
- Stage 6a diagnostics JSON (computed deterministically from the CSV)

ACCUMULATED FACTS:
{accumulated_facts}

CSV (ORIGINAL, WIDE FORMAT):
{csv_text}

CSV_DIAGNOSTICS_JSON (FROM STAGE 6a):
{csv_diagnostics_json}

---

STAGE 6 — EXTRACTION EVALUATION (USES STAGE 6a DIAGNOSTICS)

GOAL:
Evaluate the extracted CSV and produce targeted re-extraction requests for ONLY the problematic points.

HARD RULES:
- Do NOT do any arithmetic yourself if the diagnostics already provides it.
- Do NOT request re-extraction for an entire curve unless it has < 2 points.
- Always target specific time points (x-values).
- If expected trend exists for a curve in accumulated facts, use it.
- If expected trend is missing/unknown, you MAY still flag issues but MUST label them as suspicious (not wrong).
- Output MUST be JSON only. No commentary. No backticks.

DEFINITIONS (use these labels):
- direction_flip_full: curve direction strongly contradicts expected trend for most steps.
- truncation: curve is missing points after the last extracted time point.
- bad_intercept: curve's value near time 0 is an outlier vs other curves (baseline delta large).
- midcurve_switch: direction reverses with a large jump then continues wrong.

CHECKS (IN THIS ORDER):

1) truncation / early stop (highest priority)
Flag truncation ONLY if ALL hold:
- n_points >= 2 AND expected_n_points >= 4
- n_points < 0.60 * expected_n_points
- All missing times occur after the last extracted time point for that curve

2) bad_intercept at time ~0 (baseline outlier)
Apply ONLY if diagnostics provides y_at_time0 and baseline_delta_from_median.
Flag bad_intercept if:
- y_at_time0 is present AND baseline_delta_from_median is not null
- abs(baseline_delta_from_median) >= 0.75
Re-extraction targets MUST include time 0 (or the nearest reference time) and the next time.

3) trend mismatch (only when expected trend exists)
If expected trend indicates growth/increase:
- Flag direction_flip_full if pct_increasing <= 0.70
If expected trend indicates decrease/kill:
- Flag direction_flip_full if pct_decreasing <= 0.70
Re-extraction targets MUST include:
- last two times in reference_times
- first two times in reference_times
- and times around the largest_jump (from_time, to_time) if largest_jump.abs_jump is not null

4) midcurve_switch (early correct then wrong)
Apply ONLY if:
- expected trend exists OR (expected trend missing but pattern is extremely strong)
- first_direction_change_step_index is not null
- largest_jump.abs_jump >= 1.0
Label midcurve_switch_suspicious_unknown_expectation if expected trend is unknown.

---

OUTPUT JSON SCHEMA (MERGE INTO ACCUMULATED FACTS):
{
  "re_extraction": {
    "requests": [
      {
        "curve_label": "Condition 1",
        "issue_type": "truncated_curve | bad_intercept_time0 | direction_flip_full | direction_suspicious_unknown_expectation | midcurve_switch | midcurve_switch_suspicious_unknown_expectation",
        "target_times_hours": [0, 2, 24],
        "expected_trend": "kill/decrease | growth/increase | unknown",
        "evidence": {
          "n_points": 0,
          "expected_n_points": 0,
          "times_missing": [],
          "pct_increasing": 0.0,
          "pct_decreasing": 0.0,
          "baseline_delta_from_median": null,
          "largest_jump": {"abs_jump": null, "from_time": null, "to_time": null, "delta": null}
        },
        "visual_guidance": "One short sentence telling Stage 7 how to re-find the correct point (use curve style, continuity, avoid wrong decade)."
      }
    ]
  }
}

If there are NO problems, output exactly:
{}
"""


# Stage 7: Targeted re-extraction (image required)
EXTRACT_STAGE_7 = r"""
You have access to:
- Plot image
- Accumulated facts (curve definitions, colours/line/marker, expected trends when available)
- Stage 6 output JSON containing re-extraction requests

ACCUMULATED FACTS:
{accumulated_facts}

STAGE 6 OUTPUT (RE-EXTRACTION REQUESTS):
{stage6_output_json}

---

STAGE 7 — RE-EXTRACTION (TARGETED POINTS ONLY)

GOAL:
Re-extract ONLY the specified (curve_id, time_hours) points.

HARD RULES:
- Do NOT add new time points.
- Do NOT add points for curves not requested.
- Use curve definitions (colour/line/marker) from accumulated facts to select the correct curve.
- If a curve is ambiguous (overlap/crossing), lower confidence and explain why in ONE short note.
- Output JSON only. No extra text. No backticks.

METHOD (for each request):
1) Locate the curve (use colour/line/marker) and ensure continuity from nearby points.
2) Locate the specified x-value (time in hours) and read the y-value precisely.
3) Sanity check: compare against adjacent extracted points from the original CSV and/or expected trend.
4) Record only the corrected value.

OUTPUT JSON SCHEMA (MERGE INTO ACCUMULATED FACTS):
{
  "re_extraction": {
    "results": [
      {
        "curve_label": "Condition 1",
        "time_hours": 6,
        "new_log10_cfu_ml": 5.2,
        "confidence": "high | medium | low",
        "reason": "One short sentence: what was fixed (e.g., wrong curve / wrong decade / truncation).",
        "checks": {
          "used_curve_definition": true,
          "matches_expected_trend_if_known": true,
          "avoids_previous_issue": "short phrase"
        }
      }
    ]
  }
}
"""


# Stage 8: Compile new CSV (no image)
EXTRACT_STAGE_8 = r"""
You have access to:
- Original CSV (wide format)
- Stage 7 output JSON (re-extraction results)

ORIGINAL CSV:
{csv_text}

STAGE 7 OUTPUT (RE-EXTRACTION RESULTS):
{stage7_output_json}

---

STAGE 8 — COMPILE NEW CSV (MERGE FIXES)

GOAL:
Produce a NEW CSV with corrected values applied.

HARD RULES:
- Keep the exact same CSV structure and column ordering.
- Replace ONLY the y-value cell(s) for the specified curve_id at the specified time_hours.
- Do NOT add or remove rows.
- Do NOT add new columns.
- Do NOT modify other values.
- Output MUST be the full updated CSV text only (no JSON, no commentary, no backticks).

NOTES:
- Match time_hours exactly against the Time (hours) column for that curve.
- Match curve_label to the correct y-column by its header label (text in parentheses).
- Read re-extraction results from stage7_output_json.re_extraction.results.
- If a requested (curve_id, time_hours) cannot be found, leave the CSV unchanged for that entry.

---

OUTPUT:
Return only the updated CSV text.
"""

# Extract stage 5: Data point extraction (curve-by-curve, CSV + JSON)
EXTRACT_STAGE_5 = r"""
You have access to the complete extraction schema and accumulated facts
(including axes, plot type, grid facts, and curve style descriptions).

COMPLETE SCHEMA:
{complete_schema}

ACCUMULATED FACTS SO FAR:
{accumulated_facts}

---

STAGE 5 — DATA POINT EXTRACTION (CURVE-BY-CURVE, CSV + JSON)

GENERAL RULES (HARD):
- Do NOT interpolate, estimate, smooth, or infer missing points.
- Use only verified x-axis tick values.
- If a grid is present, use grid alignment only to assist value reading.
- Axis breaks are NOT valid x-values.
- Curve identity MUST come from accumulated facts; do not redefine curves.

---

Step 1 — Preconditions (abort if any fail):

- X and Y axes verified, numeric, ordered, and monotonic.
- Verified x-axis tick values exist.
- Accumulated facts contain curve descriptions (marker symbol, line colour, line type) for every curve.

If any precondition fails → abort.

---

Step 2 — Load curve definitions:

For each curve listed in accumulated facts:

- Read the curve description exactly as stored:
  - marker symbol
  - line colour
  - line type
- Treat this description as fixed and authoritative.
- Do NOT modify, infer, or override curve descriptions.

---

Step 3 — Initialise extraction for a curve:

Begin at the first verified x-axis value.

At this x-value:
- Look for a marker matching the stored curve description.
- If a marker is not clearly identifiable:
  - Check whether ALL curves share the same visible starting point.
  - If yes, use this shared point as the first point for the curve.
  - If no, do NOT extract a value at this x.

---

Step 4 — Sequential extraction across x-values:

For each subsequent verified x-axis value:

- Look for a marker matching the stored curve description.
- If a candidate marker is found:
  - Verify local curve continuity using distance D:
    D = 0.5 × (current x − previous verified x).
  - Within ±D, the curve colour and line type must match the stored curve description.
  - If continuity is confirmed → extract the y-value.
- If no marker clearly matches:
  - Do NOT report a value.
  - Move to the next x-value.
- Ensure that a dotted line or a brocken or dashed line styles are not mistaken for points. You must ensure that there is a valid marker fitting the description of the curve at each x-axis value before extracting a point.

---

Step 5 — Completion of curve:

Repeat Step 4 until all verified x-axis values are processed.

Then repeat Steps 3–5 for the next curve listed in accumulated facts.

---

Step 6 — Output construction

CSV OUTPUT (WIDE FORMAT — CRITICAL AND NON-NEGOTIABLE):

- The CSV MUST be in WIDE format.
- Long / tidy format is NOT allowed under any circumstances.

STRUCTURE RULES:

- Each curve occupies two adjacent columns:
  - Column A: X values
  - Column B: Y values for that specific curve
- Column pairs are placed side-by-side for all curves.

HEADER ROW:

- The header MUST alternate X and Y columns:
  - x_col_1, y_col_1, x_col_2, y_col_2, x_col_3, y_col_3, …
- X column headers MUST use the x-axis label (e.g. Time (hours)).
- Y column headers MUST use the y-axis label with curve name appended:
  - log10 CFU/mL (Condition 1)
  - log10 CFU/mL (Condition 2)
  - etc.

DATA ROWS:

- Each row corresponds to a single verified x-axis value.
- The same x-value is repeated in every X column for that row.
- Y values are populated only if a point was extracted for that curve at that x.
- If a curve has NO point at a given x-value:
  - Leave the corresponding Y cell blank.
  - Do NOT shift values or remove columns.
- Do NOT insert placeholder values (e.g. 0, null, NA).

---

CRITICAL OUTPUT REQUIREMENT:
- Your entire response MUST be a single valid JSON object.
- Do NOT include any prose, headings, or CSV outside the JSON.
- The response MUST start with "{{" and end with "}}".
- If you violate this, the pipeline will treat your response as CSV and fail.

---

JSON OUTPUT:

- Merge with accumulated facts.
- Populate marker_facts.curves using extracted points only.
- Include the full CSV as a single string in marker_facts.csv_output.
- Do NOT add commentary, explanations, or inferred values.
- Do NOT use triple backticks.

---

Abort output if extraction is impossible:

{{
  "abort": true,
  "reason": "Data extraction impossible due to missing labels, unreadable points, or invalid CSV construction",
  "confidence": 0-1
}}

---

Before proceeding to the next stage:
- Ensure ALL triple backticks have been removed from the response.

VERY IMPORTANT: Before moving on the the next stage ensure that any triple backticks have been removed from your response.
"""



# Code generation stage: Create matplotlib code to replot
CODE_PLOT = r"""
You are provided with:
- The same plot image.
- Extracted data and facts:
{accumulated_facts}

Generate Python (matplotlib) code that exactly replots the figure, matching styles, colors, markers, axis labels, limits, ticks, and legend.
- You MUST save the plot to EXACTLY this path: {replot_path}
- Do NOT save anywhere else and do NOT invent your own output_dir.
- The save call MUST be: plt.savefig(r"{replot_path}", ...)
- If you create directories, you may ONLY create: os.path.dirname(r"{replot_path}")
- Do not display the plot.
- Respond with code only so it can be executed directly.

CRITICAL PATH REQUIREMENTS:
- Do NOT hardcode any absolute paths other than the provided replot_path.
- Do NOT derive folders from the image name.
- Include this exact snippet near the top:
  replot_path = r"{replot_path}"
  os.makedirs(os.path.dirname(replot_path), exist_ok=True)
- And save with:
  plt.savefig(replot_path, bbox_inches='tight', dpi=300)

CRITICAL INSTRUCTIONS FOR DATA POINTS:
- Use ONLY the points listed in marker_facts.curves[].points
- Do NOT add any additional points you see in the image
- Do NOT interpolate or infer missing points
- Each curve must plot exactly the x,y coordinates from the JSON, nothing more
- Do NOT add any extra points beyond those extracted inlcuding at either the start or end of the axis break if a break is present.

CRITICAL X-AXIS POLICY FOR PROMPT_14:
- Always render a single continuous x-axis (NO broken-axis layout).
- Do NOT create split subplots for left/right x segments.
- Ignore any axis break metadata for plotting layout.
- Plot all points directly in their numeric x-values on one axes.


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
