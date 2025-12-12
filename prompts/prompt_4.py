# Copy of prompt 3
# Added handeling of axis breaks in requirement 5
prompts = {'extract': """The image depicts a figure from a research paper. Please carefully analyze the figure and extract the horizontal ("x") and vertical ("y") coordinates of each individual data point from each curve. Pay extremely close attention to the axes:
- Identify the tick marks and labels on both axes, and use them to determine the numeric values for every point.
- Read the labels of both axes (the text or symbols denoting what each axis represents).

Data Reporting Requirements:
1. Separate each curve’s data so that every curve has its own two columns of x and y.
2. The first row of your output must contain the axis labels (for example, "x-axis label" for the x-column and "y-axis label (Curve X)" for the y-column). Include the curve label in the y-axis header if it’s available.
3. Subsequent rows must present the extracted numeric data in comma-separated values (CSV) format, with:
   - No additional words or commentary.
   - Exactly one row per data point in each curve.
   - For multiple curves, produce additional pairs of columns (two columns per curve: one for x-values, one for y-values).
4. If, for any reason, you are unable to extract the data, respond with only the word "None".
5. Make sure that both axis are labeled and there are individual values assigned to each tickmark, and that they increase in a logical monotonous fashion. If an axis break is present, identify the exact location of the break and treat the segments on each side as continuous but numerically discontinuous regions. Do not interpolate across the break, do not assume missing values, and always map data points strictly according to the numeric labels on each side of the break. If the break makes it impossible to determine precise values for any points, output "None". Never invent or infer data points at the edges of an axis break. Only record a point if a clear plotted marker (e.g., circle, triangle, square, dot) is visibly present. If a line segment ends before the break and resumes after it, you must not create any point at the cutoff or restart unless an explicit marker is drawn.



Key Instructions:
- Look at each point individually; do not assume a pattern or formula. Each data point must be read precisely from its plotted location.
- Verify the numerical axis labels carefully so the extracted data is as accurate as possible.
- Do not include any explanations or text other than the required CSV table.

Output Format Example (if you have two curves):
x-axis label,y-axis label (Curve 1),x-axis label,y-axis label (Curve 2)
0.1,0.5,0.1,0.75
0.2,0.52,0.2,0.77
...

(Or "None" if you cannot extract the data.)

Remember: The sole output should be either the CSV table (with all columns for the curves) or "None". Nothing else. Do NOT use triple backticks anywhere.
""",
'code_fix': f'The text above is an error produced by your code, please fix the code so that this error does not appear. Repeat the whole code and only the code so that your whole response can be directly copied and executed. Do not explain and do not say anything else, respond with just the code.',
'compare_x': 'You are provided with two images of research plots extracted from academic papers. Do these two plots have the same x-axis (horizontal)? Do they have the same ranges, labels, etc.? Answer with a single word, "yes" or "no" only."',
'compare_y': 'You are provided with two images of research plots extracted from academic papers. Do these two plots have the same y-axis (vertical)? Do they have the same ranges, labels, etc.? Answer with a single word, "yes" or "no" only."',
'compare_number': 'You are provided with two images of research plots extracted from academic papers. Do these two plots have the same number of points (for point plots)? Do the curves look like they connect the same amount of points (for line plots)?. Answer with a single word, "yes" or "no" only."',
'compare_trend': 'You are provided with two images of research plots extracted from academic papers. Do these sets of points or curves on these two plots represent the same trends? Do they follow the same patterns? Are points distributed in the same way? Answer with a single word, "yes" or "no" only."',
'code_plot': 'Please analyze the figure and create a python code that will reproduce the plot exactly, including colors, line types, point shapes, axis labels, axis ranges, etc. Save the plot as a file "{replot_plot}" only and do not show it. Respond with the code only so that it can be directly copied and executed. Do NOT use triple backticks anywhere. \n\nUse the following data on the plot: {data}'}
