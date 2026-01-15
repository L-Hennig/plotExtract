# =====================================================================
# PlotExtract V2 - Prompt Set 1
# Three-stage extraction with accumulated facts
# =====================================================================

# Extract stage 1: X-axis verification
EXTRACT_STAGE_1 = r"""
You have access to the complete extraction schema (showing all possible fields).
You will output ONLY the fields you determine, following this structure.

COMPLETE SCHEMA:
{complete_schema}

ACCUMULATED FACTS SO FAR:
{accumulated_facts}

---

Analyze the provided plot image. Determine if the x-axis represents time in hours. Identify discrete tick marks and labels.

Output ONLY the populated fields as JSON (do not include empty fields):

{{
  "axis_facts": {{
    "x_axis": {{
      "quantity": "time",
      "unit": "hours",
      "discrete": true/false,
      "ticks_verified": [list of tick marks] or [],
      "tick_labels": [labels if present],
      "confidence": 0–1
    }}
  }}
}}

If the x-axis cannot be confidently verified, set "discrete": false, "ticks_verified": [], confidence: 0.
If verification completely fails, output only "None".
"""

# Extract stage 2: Y-axis verification
EXTRACT_STAGE_2 = r"""
You have access to the complete extraction schema (showing all possible fields).
You will output ONLY the fields you determine, building on previous facts.

COMPLETE SCHEMA:
{complete_schema}

ACCUMULATED FACTS SO FAR:
{accumulated_facts}

---

Determine if the y-axis represents bacterial burden on a log10 scale. Identify tick marks, labels, and plausible range.

Output ONLY the new/updated fields as JSON (merge with previous facts in the output):

{{
  "axis_facts": {{
    "y_axis": {{
      "quantity": "bacterial burden",
      "unit": "log10 CFU/mL",
      "scale": "log10",
      "upper_plausible_limit": 9,
      "ticks_verified": [list of tick marks],
      "tick_labels": [labels if present],
      "confidence": 0–1
    }}
  }}
}}

If the y-axis cannot be verified, set ticks_verified: [], confidence: 0.
If verification completely fails, output only "None".
"""

# Extract stage 3: Marker extraction + CSV output
EXTRACT_STAGE_3 = r"""
You have access to the complete extraction schema (showing all possible fields).
You will output ONLY the fields you determine, building on previous facts.

COMPLETE SCHEMA:
{complete_schema}

ACCUMULATED FACTS SO FAR:
{accumulated_facts}

---

Identify explicit markers on the plot (ignore lines connecting points). Extract the x and y coordinates of each marker using the verified axes. Output the data in CSV format.

Output as JSON with marker_facts and CSV:

{{
  "marker_facts": {{
    "markers_detected": true/false,
    "curves": [
      {{
        "curve_label": "string if identifiable",
        "points": [
          {{"x": number, "y": number, "confidence": 0-1}}
        ]
      }}
    ],
    "csv_output": "actual CSV content here",
    "confidence": 0–1
  }}
}}

CSV Requirements:
- First row contains axis labels. Include curve label in the y-axis header if available.
- Subsequent rows present numeric data only, one row per data point.
- Each curve gets two columns (x and y). Multiple curves produce additional pairs of columns.
- Both axes must be labelled, tick values must increase logically; otherwise, output "None".

If extraction fails, output only "None".
"""

# Code generation stage: Create matplotlib code to replot
CODE_PLOT = r"""
You are provided with:
- The same plot image.
- Extracted data and facts:
{accumulated_facts}

Generate Python (matplotlib) code that exactly replots the figure, matching styles, colors, markers, axis labels, limits, ticks, and legend.

CRITICAL REQUIREMENTS:
- If axis_facts.y_axis.scale is "log10", MUST use plt.yscale('log') to set logarithmic scale
- If axis_facts.y_axis.scale is "linear", use linear scale (default)
- Match all axis labels, limits, and legend from the extracted facts
- Save the plot only to: {replot_path}
- Do not display the plot.
- Respond with code only so it can be executed directly.
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
