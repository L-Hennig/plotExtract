# =====================================================================
# PlotExtract V2 - Prompt 9
# Stage renumbering:
# - New STAGE 4 = curve definitions (style metadata)
# - Old STAGE 4 (data extraction) is now STAGE 5
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
