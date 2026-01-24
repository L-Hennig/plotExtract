# ChatGPT prompt
prompts = {'extract': """ The image shows a time-kill plot from a research paper. Before extracting any numeric data, determine whether the figure is structurally suitable for time-kill data extraction.

Carefully examine the axes and confirm that:
- The x-axis represents time and has clearly readable numeric tick labels.
- The y-axis represents bacterial burden on a log₁₀ scale (e.g. log₁₀ CFU/mL).
- The y-axis tick labels increase monotonically in log₁₀ units (e.g. 0, 1, 2, … or 2, 4, 6, 8).
- All axis tick marks have explicit numeric values and follow a consistent, readable progression appropriate for time-kill plots.
- Any axis breaks (commonly used in time-kill plots) are clearly indicated and identifiable.
- Any limit-of-detection (LOD) or detection-threshold lines are visually distinguishable from data curves.

Use only the axis tick marks and labels to determine numeric values. Do not infer values based on visual spacing between ticks, as spacing on a log₁₀ axis is non-linear.  
If the axis labels, log₁₀ scale, axis continuity, or special indicators (such as axis breaks or LOD lines) cannot be confidently identified, do not attempt numeric extraction and respond with "None".

Definition of valid data points (time-kill plots):

For time-kill plots, a valid data point is defined as a clearly visible, explicit marker (e.g. circle, triangle, square, diamond) that denotes a measured observation at a specific timepoint.

Apply the following rules strictly:
- Extract numeric values only from explicit markers.
- Do not extract values from line segments, fitted curves, or visual interpolations between markers.
- Connecting lines are visual aids only and must not be treated as data.
- If a curve contains no visible markers, treat that curve as non-extractable.
- If markers overlap, are partially occluded, or their exact position cannot be confidently determined, omit those points rather than estimating them.
- Do not infer or create data points to complete a curve or to match the number of points in other curves.

Only points that are clearly identifiable as individual markers with unambiguous positions relative to the axis tick marks should be extracted.

Axis breaks and disjoint time regions (time-kill plots):

Time-kill plots may contain one or more axis breaks, most commonly on the time (x) axis to separate early and late timepoints.

Apply the following rules:
- Explicitly identify any axis breaks before extracting numeric data.
- Treat regions separated by an axis break as disjoint; no data exists within the break.
- Do not interpolate, extrapolate, or infer data points across an axis break.
- Do not assume continuity of trends, slopes, or time spacing across a break.
- Extract only markers that lie fully within a visible axis segment.

If an axis break is present but its location or extent is ambiguous, do not attempt extraction and respond with "None".

Limit of detection (LOD) handling (time-kill plots):

Time-kill plots commonly include a limit-of-detection (LOD) or detection-threshold indicator, often shown as a dashed horizontal line.

Apply the following rules strictly:
- Identify any LOD or detection-threshold indicators before numeric extraction.
- LOD lines are not data curves and must never be digitised as data.
- Data points plotted exactly on an LOD line represent censored measurements, not true numeric values below the LOD.
- Do not infer values below the LOD.
- Do not extrapolate trends past the LOD line.
- If it is unclear whether a horizontal line represents an LOD indicator or a data curve, do not attempt extraction and respond with "None".

Only explicit data markers should be extracted; LOD indicators must be excluded from numeric data.

Conservative extraction and refusal behaviour:

The goal of extraction is maximal precision, not maximal completeness.

Apply the following rules:
- Extract numeric values only when the marker position relative to axis tick marks is unambiguous.
- If a data point cannot be confidently read, omit it rather than estimating.
- Do not assume evenly spaced timepoints or values unless explicitly indicated by visible tick labels.
- Do not infer missing points, smooth curves, or adjust values to improve apparent trends.
- Treat each curve independently; do not force curves to share the same number of points or timepoints.
- If multiple violations or ambiguities are encountered within a figure, do not partially extract data; respond with "None".

Skipping uncertain points is correct behaviour. Guessing is incorrect behaviour.

Data Reporting Requirements:
1. Separate each curve’s data so that every curve has its own two columns of x and y.
2. The first row of the output must contain the axis labels (e.g. "Time (h)" for the x-column and "log₁₀ CFU/mL (Curve X)" for the y-column). Include the curve label in the y-axis header if available.
3. Subsequent rows must present the extracted numeric data in comma-separated values (CSV) format, with:
   - No additional words or commentary.
   - One row per confidently extractable data point.
   - For multiple curves, additional pairs of columns (two columns per curve: one for x-values, one for y-values).
4. If, for any reason, you are unable to extract the data, respond with only the word "None".

Remember: The sole output should be either the CSV table (with all columns for the curves) or "None". Nothing else. Do NOT use triple backticks anywhere.
""",
'code_fix': f'The text above is an error produced by your code, please fix the code so that this error does not appear. Repeat the whole code and only the code so that your whole response can be directly copied and executed. Do not explain and do not say anything else, respond with just the code.',
'compare_x': 'You are provided with two images of research plots extracted from academic papers. Do these two plots have the same x-axis (horizontal)? Do they have the same ranges, labels, etc.? Answer with a single word, "yes" or "no" only."',
'compare_y': 'You are provided with two images of research plots extracted from academic papers. Do these two plots have the same y-axis (vertical)? Do they have the same ranges, labels, etc.? Answer with a single word, "yes" or "no" only."',
'compare_number': 'You are provided with two images of research plots extracted from academic papers. Do these two plots have the same number of points (for point plots)? Do the curves look like they connect the same amount of points (for line plots)?. Answer with a single word, "yes" or "no" only."',
'compare_trend': 'You are provided with two images of research plots extracted from academic papers. Do these sets of points or curves on these two plots represent the same trends? Do they follow the same patterns? Are points distributed in the same way? Answer with a single word, "yes" or "no" only."',
'code_plot': 'Please analyze the figure and create a python code that will reproduce the plot exactly, including colors, line types, point shapes, axis labels, axis ranges, etc. Save the plot as a file "{replot_plot}" only and do not show it. Respond with the code only so that it can be directly copied and executed. Do NOT use triple backticks anywhere. \n\nUse the following data on the plot: {data}'}
