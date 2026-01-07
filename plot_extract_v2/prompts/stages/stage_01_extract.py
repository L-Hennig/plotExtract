PROMPT_TEXT = r"""
You will extract all numeric data from the provided research plot image.
- Output must be CSV only (no commentary).
- First row: axis labels. Include curve names in y headers when available.
- Subsequent rows: numeric x,y pairs per curve (two columns per curve).
- If you cannot extract, respond exactly with the word "None".
- Use the axis tick marks/labels; do not assume patterns.
- Verify both axes are monotonic and labeled; if not possible, output "None".
"""
