PROMPT_TEMPLATE = r"""
You are provided with:
- The same plot image.
- Prior extracted CSV data:
{data_context}

Generate Python (matplotlib) code that exactly replots the figure, matching styles, colors, markers, axis labels, limits, ticks, and legend.
- Save the plot only to: {replot_path}
- Do not display the plot.
- Respond with code only so it can be executed directly.
"""

CODE_FIX = (
    "The text above is an error produced by your code. "
    "Rewrite the full corrected code only; no explanations; the response must be executable as-is."
)
