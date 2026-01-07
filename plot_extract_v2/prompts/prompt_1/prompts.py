# =====================================================================
# PlotExtract V2 - Prompt Set 1
# Three-stage extraction with accumulated facts
# =====================================================================

# Extract stage 1: X-axis verification
EXTRACT_STAGE_1 = r"""
Analyze the provided plot image. Determine if the x-axis represents time in hours. Identify discrete tick marks and labels.

Output JSON:

{{
  "axis_facts": {{
    "x_axis": {{
      "quantity": "time",
      "unit": "hours",
      "discrete": true/false,
      "ticks_verified": [list of tick marks] or [],
      "confidence": 0–1
    }}
  }}
}}

If the x-axis cannot be confidently verified, set "discrete": false, "ticks_verified": [], confidence: 0.
If verification completely fails, output only "None".
"""

# Extract stage 2: Y-axis verification
EXTRACT_STAGE_2 = r"""
Use the facts from Step 1: {data_context}

Determine if the y-axis represents bacterial burden on a log10 scale. Identify tick marks, labels, and plausible range.

Output JSON (merge previous facts):

{{
  "axis_facts": {{
    "x_axis": {{ ...previous x_axis facts... }},
    "y_axis": {{
      "quantity": "bacterial burden",
      "unit": "log10 CFU/mL",
      "scale": "log10",
      "upper_plausible_limit": 9,
      "ticks_verified": [list of tick marks],
      "confidence": 0–1
    }}
  }}
}}

If the y-axis cannot be verified, set ticks_verified: [], confidence: 0.
If verification completely fails, output only "None".
"""

# Extract stage 3: Marker extraction + CSV output
EXTRACT_STAGE_3 = r"""
Use all previous facts: {data_context}

Identify explicit markers on the plot (ignore lines connecting points). Extract the x and y coordinates of each marker using the verified axes. Output the data in CSV format.

Output CSV Requirements:

- First row contains axis labels. Include curve label in the y-axis header if available.
- Subsequent rows present numeric data only, one row per data point.
- Each curve gets two columns (x and y). Multiple curves produce additional pairs of columns.
- If extraction fails, output only "None".
- Both axes must be labelled, tick values must increase logically; otherwise, output "None".

Example CSV format (two curves):

x-axis label,y-axis label (Curve 1),x-axis label,y-axis label (Curve 2)
0.1,0.5,0.1,0.75
0.2,0.52,0.2,0.77
...

After CSV, also output JSON for tracking:

{{
  "axis_facts": {{ ...all axis facts from steps 1–2... }},
  "marker_facts": {{
    "markers_detected": true/false,
    "csv_output": "actual CSV content here",
    "confidence": 0–1
  }}
}}
"""

# Code generation stage: Create matplotlib code to replot
CODE_PLOT = r"""
You are provided with:
- The same plot image.
- Prior extracted CSV data:
{data_context}

Generate Python (matplotlib) code that exactly replots the figure, matching styles, colors, markers, axis labels, limits, ticks, and legend.
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
