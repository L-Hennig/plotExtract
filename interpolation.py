import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import ast
import io

def _safe_reconfigure_text_stream(stream, encoding: str = 'utf-8', errors: str = 'replace'):
    # On Windows, re-wrapping sys.stdout/sys.stderr using stream.buffer can invalidate
    # the underlying handle (especially when launched with subprocess capture_output).
    # Prefer reconfigure() (py3.7+), otherwise detach() then wrap.
    try:
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding=encoding, errors=errors)
            return stream
    except Exception:
        return stream

    try:
        if hasattr(stream, 'detach'):
            raw = stream.detach()
            return io.TextIOWrapper(raw, encoding=encoding, errors=errors)
    except Exception:
        return stream

    return stream


# Configure stdout/stderr to use UTF-8 encoding on Windows
if sys.platform == 'win32':
    sys.stdout = _safe_reconfigure_text_stream(sys.stdout)
    sys.stderr = _safe_reconfigure_text_stream(sys.stderr)

# Import Mistral for LLM-based curve matching
from mistralai import Mistral
from dotenv import load_dotenv
load_dotenv(override=True)

if len(sys.argv) < 7:
    print("Usage: python interpolation.py <original.csv> <extracted.csv> leftX rightX bottomY topY [output_dir]")
    sys.exit(1)

# Loads API key from .env file
api_key = os.getenv("API_KEY_1")
client = Mistral(api_key=api_key)

def prompt_mistral(prompt_text):
    """Send a prompt to Mistral and return the response."""
    response = client.chat.complete(
        model="mistral-large-2512",
        messages=[{"role": "user", "content": prompt_text}],
        max_tokens=4096,
        temperature=0,
    )
    return response.choices[0].message.content


def _read_csv_tolerant(filepath: str, encoding: str) -> pd.DataFrame:
    """Read CSVs that may contain ragged rows (extra commas / missing fields).

    Some extraction outputs occasionally include rows with an extra empty field
    (e.g. `,,,,`) which makes the row have more columns than the header.
    Pandas' default CSV reader raises a ParserError in that case.

    Strategy:
    - Determine expected number of columns from the header line.
    - For each subsequent row:
      - If too many fields: remove empty fields until the count matches.
      - If still too many: truncate the row.
      - If too few: pad with empty fields.
    - Feed the normalized CSV text into pandas.

    Note: This assumes no embedded commas inside quoted fields (true for our
    generated numeric CSVs).
    """
    with open(filepath, 'r', encoding=encoding, errors='replace') as f:
        lines = f.read().splitlines()

    if not lines:
        return pd.DataFrame()

    header = lines[0]
    expected_cols = len(header.split(','))
    if expected_cols <= 0:
        return pd.DataFrame()

    out_lines = [header]
    for raw in lines[1:]:
        if raw.strip() == '':
            continue
        fields = raw.split(',')
        if len(fields) > expected_cols:
            # Prefer removing empty tokens first (these are almost always the issue).
            while len(fields) > expected_cols and '' in fields:
                fields.remove('')
        if len(fields) > expected_cols:
            fields = fields[:expected_cols]
        elif len(fields) < expected_cols:
            fields = fields + [''] * (expected_cols - len(fields))
        out_lines.append(','.join(fields))

    return pd.read_csv(io.StringIO('\n'.join(out_lines)), encoding=encoding)

def load_multi_curve_csv(filepath):
    """
    Load a CSV with multiple curves (pairs of x,y columns).
    Returns a dict: {curve_label: DataFrame with 'x' and 'y' columns}
    """
    # Try UTF-8 first (for extracted CSVs), fall back to latin1 if needed
    try:
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
        except pd.errors.ParserError:
            df = _read_csv_tolerant(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filepath, encoding='latin1')
        except pd.errors.ParserError:
            df = _read_csv_tolerant(filepath, encoding='latin1')
    headers = df.columns.tolist()
    
    curves = {}
    # Assume columns are in pairs: x1, y1, x2, y2, ...
    num_curves = len(headers) // 2
    
    for i in range(num_curves):
        x_col = headers[i * 2]
        y_col = headers[i * 2 + 1]
        
        # Extract the curve data, dropping NaN rows
        curve_data = df[[x_col, y_col]].dropna()
        curve_data.columns = ['x', 'y']
        curve_data = curve_data.sort_values(by='x').reset_index(drop=True)
        
        # Use the y-column header as the curve label
        curves[y_col] = {
            'data': curve_data,
            'x_label': x_col,
            'y_label': y_col
        }
    
    return curves


def _expand_limits(lo, hi, pad_ratio=0.05):
    """Expand [lo, hi] by a fraction for nicer plotting."""
    if lo is None or hi is None:
        return lo, hi
    if not np.isfinite(lo) or not np.isfinite(hi):
        return lo, hi
    if lo == hi:
        pad = (abs(lo) * pad_ratio) if lo != 0 else 1.0
        return lo - pad, hi + pad
    span = hi - lo
    pad = span * pad_ratio
    return lo - pad, hi + pad


def _nice_upper_bound(v: float) -> float:
    """Return a 'nice' upper axis bound >= v.

    - Uses 1/2/5/10 * 10^k rounding.
    - If v is exactly a power of 10 (e.g. 1e9), returns the next decade (1e10).
    """
    if v is None:
        return 1.0
    try:
        v = float(v)
    except Exception:
        return 1.0

    if not np.isfinite(v) or v <= 0:
        return 1.0

    exp = int(np.floor(np.log10(v)))
    base = 10.0 ** exp
    frac = v / base

    # Exact power-of-ten: only bump to next decade for large ranges.
    # For small ranges (e.g. 10), a decade bump (100) is far too aggressive.
    if np.isclose(frac, 1.0, rtol=1e-12, atol=0.0):
        return (10.0 * base) if v >= 1e3 else (2.0 * base)

    if frac <= 2.0:
        return 2.0 * base
    if frac <= 5.0:
        return 5.0 * base
    return 10.0 * base


def _compute_data_bounds(*curves_dicts):
    """Return (min_x, max_x, min_y, max_y) across all curves in dicts."""
    min_x = min_y = np.inf
    max_x = max_y = -np.inf

    for curves in curves_dicts:
        for curve_info in curves.values():
            df = curve_info.get('data')
            if df is None or df.empty:
                continue
            xs = pd.to_numeric(df['x'], errors='coerce').dropna()
            ys = pd.to_numeric(df['y'], errors='coerce').dropna()
            if xs.empty or ys.empty:
                continue
            min_x = min(min_x, float(xs.min()))
            max_x = max(max_x, float(xs.max()))
            min_y = min(min_y, float(ys.min()))
            max_y = max(max_y, float(ys.max()))

    if min_x == np.inf:
        return None, None, None, None
    return min_x, max_x, min_y, max_y

def match_curves_with_llm(original_curves, extracted_curves, original_file, extracted_file):
    """
    Use LLM to match curves between original and extracted CSVs.
    Returns a dict mapping original curve labels to extracted curve labels.
    """
    # Build a summary of both CSVs for the LLM
    original_summary = "Original CSV curves:\n"
    for label, curve_info in original_curves.items():
        data = curve_info['data']
        original_summary += f"  - '{label}': x_label='{curve_info['x_label']}', "
        original_summary += f"x_range=[{data['x'].min():.4f}, {data['x'].max():.4f}], "
        original_summary += f"y_range=[{data['y'].min():.4f}, {data['y'].max():.4f}], "
        original_summary += f"num_points={len(data)}\n"
        # Add sample points
        sample = data.head(3)
        original_summary += f"    Sample points: {list(zip(sample['x'], sample['y']))}\n"
    
    extracted_summary = "Extracted CSV curves:\n"
    for label, curve_info in extracted_curves.items():
        data = curve_info['data']
        extracted_summary += f"  - '{label}': x_label='{curve_info['x_label']}', "
        extracted_summary += f"x_range=[{data['x'].min():.4f}, {data['x'].max():.4f}], "
        extracted_summary += f"y_range=[{data['y'].min():.4f}, {data['y'].max():.4f}], "
        extracted_summary += f"num_points={len(data)}\n"
        # Add sample points
        sample = data.head(3)
        extracted_summary += f"    Sample points: {list(zip(sample['x'], sample['y']))}\n"
    
    prompt = f"""You are given two CSV files: the first is the original data and the second is extracted data. 
Each CSV may contain multiple curves. The curves in the extracted file may appear in a different order 
or have slightly different y-axis labels compared to the original file. 

Your task is to identify which curve in the extracted CSV corresponds to which curve in the original CSV.

{original_summary}

{extracted_summary}

Instructions:
- Use the column headers and sample data points to match curves between the two files.
- Output the mapping as a Python dictionary where the keys are the original curve labels and the values 
  are the corresponding extracted curve labels.
- The output must be valid Python syntax, parseable directly with `eval()` or `ast.literal_eval()`.

Example output format:
{{
    "Original Curve 1": "Extracted Curve B",
    "Original Curve 2": "Extracted Curve A",
}}

Make sure to match all curves accurately, even if the names or order differ slightly.
Do not add extra explanations or text—only output the dictionary."""

    print("Matching curves using LLM... ", end='', flush=True)
    response = prompt_mistral(prompt)
    print("DONE")
    
    # Parse the response
    try:
        # Clean up response - remove markdown code blocks if present
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("python"):
                response = response[6:]
        response = response.strip()
        
        mapping = ast.literal_eval(response)
        return mapping
    except Exception as e:
        print(f"Warning: Could not parse LLM response: {e}")
        print(f"Response was: {response}")
        # Fallback: try to match by order
        original_labels = list(original_curves.keys())
        extracted_labels = list(extracted_curves.keys())
        mapping = {}
        for i, orig_label in enumerate(original_labels):
            if i < len(extracted_labels):
                mapping[orig_label] = extracted_labels[i]
        return mapping

def interpolate_and_compare(orig_data, extr_data, xrange_arg, yrange_arg):
    """
    Perform interpolation comparison between two curves.
    Returns dict with MAE, left_miss, right_miss, and data for plotting.
    """
    # Finding the overlapping x range
    overlap_x_min = max(orig_data['x'].min(), extr_data['x'].min())
    overlap_x_max = min(orig_data['x'].max(), extr_data['x'].max())
    
    # Determine the non-overlapping ranges
    left_miss = extr_data['x'].min() - orig_data['x'].min()
    right_miss = orig_data['x'].max() - extr_data['x'].max()
    
    # Creating a common x range for the overlapping area
    common_x_overlap = np.linspace(overlap_x_min, overlap_x_max, num=1000)
    
    # Interpolating y values for both datasets
    interpolated_y1_overlap = np.interp(common_x_overlap, orig_data['x'], orig_data['y'])
    interpolated_y2_overlap = np.interp(common_x_overlap, extr_data['x'], extr_data['y'])
    
    # Calculating the absolute differences
    differences_overlap = interpolated_y2_overlap - interpolated_y1_overlap
    
    # Calculate the average difference (MAE)
    average_difference_overlap = np.mean(np.abs(differences_overlap))
    
    return {
        'mae': average_difference_overlap / yrange_arg,
        'left_miss': left_miss / xrange_arg,
        'right_miss': right_miss / xrange_arg,
        'common_x': common_x_overlap,
        'differences': differences_overlap,
        'interp_orig': interpolated_y1_overlap,
        'interp_extr': interpolated_y2_overlap,
        'orig_data': orig_data,
        'extr_data': extr_data
    }

# Load command line arguments
original_file = sys.argv[1]
extracted_file = sys.argv[2]
leftX = float(sys.argv[3])
rightX = float(sys.argv[4])
bottomY = float(sys.argv[5])
topY = float(sys.argv[6])

# Optional output directory (if provided, outputs go there instead of original file's directory)
if len(sys.argv) > 7:
    output_dir = sys.argv[7]
else:
    output_dir = os.path.dirname(original_file)

# Load both CSV files
print(f"Loading original CSV: {original_file}")
original_curves = load_multi_curve_csv(original_file)
print(f"  Found {len(original_curves)} curve(s): {list(original_curves.keys())}")

print(f"Loading extracted CSV: {extracted_file}")
extracted_curves = load_multi_curve_csv(extracted_file)
print(f"  Found {len(extracted_curves)} curve(s): {list(extracted_curves.keys())}")

# Axis ranges
# - Always start at origin for interpolation plots.
# - If the caller provided explicit caps (rightX/topY > 0), treat them as hard caps.
# - Otherwise, auto-expand to include all data and round up to a "nice" bound.
data_min_x, data_max_x, data_min_y, data_max_y = _compute_data_bounds(original_curves, extracted_curves)

eff_leftX = 0.0
eff_bottomY = 0.0

if rightX > 0:
    eff_rightX = rightX
else:
    raw_right_x = data_max_x if data_max_x is not None else 1.0
    eff_rightX = _nice_upper_bound(raw_right_x)

if topY > 0:
    eff_topY = topY
else:
    raw_top_y = data_max_y if data_max_y is not None else 1.0
    eff_topY = _nice_upper_bound(raw_top_y)

print(f"Effective plot bounds: X=[{eff_leftX}, {eff_rightX}], Y=[{eff_bottomY}, {eff_topY}]")

xrange_arg = eff_rightX - eff_leftX
yrange_arg = eff_topY - eff_bottomY
if xrange_arg == 0:
    xrange_arg = 1e-12
if yrange_arg == 0:
    yrange_arg = 1e-12

# Match curves using LLM
curve_mapping = match_curves_with_llm(original_curves, extracted_curves, original_file, extracted_file)
print(f"Curve mapping: {curve_mapping}")

# Perform interpolation for each matched curve
results = {}
for orig_label, extr_label in curve_mapping.items():
    if orig_label not in original_curves:
        print(f"Warning: '{orig_label}' not found in original curves, skipping")
        continue
    if extr_label not in extracted_curves:
        print(f"Warning: '{extr_label}' not found in extracted curves, skipping")
        continue
    
    print(f"Comparing '{orig_label}' vs '{extr_label}'... ", end='', flush=True)
    orig_data = original_curves[orig_label]['data']
    extr_data = extracted_curves[extr_label]['data']
    
    result = interpolate_and_compare(orig_data, extr_data, xrange_arg, yrange_arg)
    result['orig_label'] = orig_label
    result['extr_label'] = extr_label
    result['x_label'] = original_curves[orig_label]['x_label']
    results[orig_label] = result
    print(f"MAE = {result['mae']:.6f}")

num_curves = len(results)
if num_curves == 0:
    print("Error: No curves matched successfully")
    sys.exit(1)

# Determine subplot layout
if num_curves <= 8:
    ncols = min(2, num_curves)  # Don't create more columns than curves
else:
    ncols = 3
nrows = (num_curves + ncols - 1) // ncols  # Ceiling division

# Dynamic figure size
fig_width = 6 * ncols
fig_height = 4 * nrows
fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)

# Flatten axes for easy iteration
axes = axes.flatten()

# Plot each curve comparison
colors = plt.cm.tab10.colors
for idx, (orig_label, result) in enumerate(results.items()):
    ax = axes[idx]
    
    orig_data = result['orig_data']
    extr_data = result['extr_data']
    common_x = result['common_x']
    interp_orig = result['interp_orig']
    interp_extr = result['interp_extr']
    
    # Plot original and extracted curves
    ax.plot(orig_data['x'], orig_data['y'], label='original', linestyle='-', color=colors[0])
    ax.plot(extr_data['x'], extr_data['y'], label='llm', linestyle='-', color=colors[1])
    
    # Fill area between interpolated curves (keeps y-axis range consistent with the curves)
    ax.fill_between(
        common_x,
        interp_orig,
        interp_extr,
        color='gray',
        alpha=0.35,
        label=f'MAE={result["mae"]:.4f}',
    )
    
    # Add miss info to legend
    ax.scatter([None], [None], label=f'Miss L:{result["left_miss"]:.2f} R:{result["right_miss"]:.2f}')
    
    ax.set_title(f'{orig_label}')
    ax.set_xlabel(result['x_label'])
    ax.set_ylabel(orig_label)
    ax.legend(fontsize='small')
    ax.grid(True)
    
    # Axes: origin start and shared, nice upper bounds
    ax.set_xlim(eff_leftX, eff_rightX)
    ax.set_ylim(eff_bottomY, eff_topY)

# Hide unused subplots
for idx in range(num_curves, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Interpolation Comparison: Original vs Extracted Data', fontsize=14)
plt.tight_layout()

# Save outputs
f1 = os.path.basename(original_file)
f2 = os.path.basename(extracted_file)
output_base = os.path.join(output_dir, f"interpolated_{f1}_VS_{f2}")

# Save figure
plt.savefig(f"{output_base}.png", dpi=150)
plt.close()
print(f"Saved figure: {output_base}.png")

# Save statistics
with open(f"{output_base}.stats", 'w', encoding='utf-8') as file:
    file.write("# Curve-by-curve statistics\n")
    mae_values = []
    for orig_label, result in results.items():
        file.write(f"Curve '{orig_label}' -> '{result['extr_label']}':\n")
        file.write(f"  MAE: {result['mae']:.6f}\n")
        file.write(f"  LeftMissed: {result['left_miss']:.6f}\n")
        file.write(f"  RightMissed: {result['right_miss']:.6f}\n")
        mae_values.append(result['mae'])
    
    mean_mae = np.mean(mae_values)
    file.write(f"\n# Summary\n")
    file.write(f"Number of curves: {num_curves}\n")
    file.write(f"Mean MAE: {mean_mae:.6f}\n")

print(f"Saved statistics: {output_base}.stats")
print(f"\nMean MAE across all curves: {mean_mae:.6f}")
