import sys
import os
import math
import ast
import io
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Configure stdout/stderr to use UTF-8 encoding on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from dotenv import load_dotenv

load_dotenv(override=True)

MAX_NORM_DIST = float(os.getenv("PLOTEXTRACT_POINTWISE_MAX_NORM_DIST", "0.1"))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_curve_matcher() -> str:
    # heuristic (default) | llm
    return os.getenv("PLOTEXTRACT_POINTWISE_CURVE_MATCHER", "heuristic").strip().lower()


def _get_llm_timeout_s() -> float:
    try:
        return float(os.getenv("PLOTEXTRACT_POINTWISE_LLM_TIMEOUT_S", "20"))
    except Exception:
        return 20.0


def _try_init_mistral_client():
    """Optional Mistral client. Pointwise should still work without it."""
    try:
        from mistralai import Mistral  # type: ignore
    except Exception:
        return None

    api_key = os.getenv("API_KEY_1")
    if not api_key:
        return None
    try:
        return Mistral(api_key=api_key)
    except Exception:
        return None

def prompt_mistral(prompt_text: str, timeout_s: float = 20.0) -> str:
    """Send a prompt to Mistral and return the response.

    Runs in a thread with a hard timeout so pointwise never hangs indefinitely.
    """
    client = _try_init_mistral_client()
    if client is None:
        raise RuntimeError("Mistral client unavailable (missing package or API_KEY_1)")

    def _call() -> str:
        response = client.chat.complete(
            model=os.getenv("PLOTEXTRACT_POINTWISE_MISTRAL_MODEL", "mistral-large-2512"),
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=4096,
            temperature=0,
        )
        return response.choices[0].message.content

    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_call)
        return future.result(timeout=timeout_s)


def _read_csv_tolerant(filepath: str, encoding: str) -> pd.DataFrame:
    """Read CSVs that may contain ragged rows (extra commas / missing fields).

    Extraction outputs sometimes contain rows with an extra empty field (e.g.
    `,,,,`) which makes pandas raise a ParserError. We normalize each row to the
    header's column count by removing empty tokens first, then truncating/padding.

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
    Returns a dict: {curve_label: {'data': DataFrame, 'x_label': str, 'y_label': str, 'coords': list}}
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
        
        # Create coords list
        coords = list(zip(curve_data['x'], curve_data['y']))
        
        # Use the y-column header as the curve label
        curves[y_col] = {
            'data': curve_data,
            'x_label': x_col,
            'y_label': y_col,
            'coords': coords
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

    Uses 1/2/5/10 * 10^k rounding. If v is exactly a power of 10 (e.g. 1e9),
    returns the next decade (1e10).
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

def match_curves_with_llm(original_curves, extracted_curves):
    """Use LLM to match curves between original and extracted CSVs.

    Returns a dict mapping original curve labels to extracted curve labels.
    This is optional and should not be relied on for runtime stability.
    """
    original_summary = "Original CSV curves:\n"
    for label, curve_info in original_curves.items():
        data = curve_info['data']
        original_summary += (
            f"  - '{label}': x_label='{curve_info['x_label']}', "
            f"x_range=[{data['x'].min():.4f}, {data['x'].max():.4f}], "
            f"y_range=[{data['y'].min():.4f}, {data['y'].max():.4f}], "
            f"num_points={len(data)}\n"
        )
        sample = data.head(3)
        original_summary += f"    Sample points: {list(zip(sample['x'], sample['y']))}\n"

    extracted_summary = "Extracted CSV curves:\n"
    for label, curve_info in extracted_curves.items():
        data = curve_info['data']
        extracted_summary += (
            f"  - '{label}': x_label='{curve_info['x_label']}', "
            f"x_range=[{data['x'].min():.4f}, {data['x'].max():.4f}], "
            f"y_range=[{data['y'].min():.4f}, {data['y'].max():.4f}], "
            f"num_points={len(data)}\n"
        )
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
- The output must be valid Python syntax, parseable directly with `ast.literal_eval()`.

Only output the dictionary."""

    timeout_s = _get_llm_timeout_s()
    print(f"Matching curves using LLM (timeout={timeout_s:.0f}s)... ", end='', flush=True)
    response = prompt_mistral(prompt, timeout_s=timeout_s)
    print("DONE")

    # Parse response
    try:
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("python"):
                response = response[6:]
        response = response.strip()
        return ast.literal_eval(response)
    except Exception as e:
        print(f"Warning: Could not parse LLM response: {e}")
        print(f"Response was: {response}")
        original_labels = list(original_curves.keys())
        extracted_labels = list(extracted_curves.keys())
        mapping = {}
        for i, orig_label in enumerate(original_labels):
            if i < len(extracted_labels):
                mapping[orig_label] = extracted_labels[i]
        return mapping


def match_curves_heuristic(original_curves, extracted_curves):
    """Fast, deterministic curve mapping.

    Strategy:
    - If labels overlap, match by exact label.
    - Otherwise match by order.
    """
    original_labels = list(original_curves.keys())
    extracted_labels = list(extracted_curves.keys())

    mapping = {}

    extracted_set = set(extracted_labels)
    for orig_label in original_labels:
        if orig_label in extracted_set:
            mapping[orig_label] = orig_label

    used = set(mapping.values())
    remaining_extracted = [lab for lab in extracted_labels if lab not in used]
    for orig_label in original_labels:
        if orig_label in mapping:
            continue
        if remaining_extracted:
            mapping[orig_label] = remaining_extracted.pop(0)

    return mapping


def normalize_point(x, y, leftX, rightX, bottomY, topY):
    """Normalize a point into [0,1]x[0,1] using provided axis ranges."""
    try:
        x = float(x)
        y = float(y)
    except Exception:
        return 0.0, 0.0

    denom_x = (rightX - leftX)
    denom_y = (topY - bottomY)
    if denom_x == 0:
        denom_x = 1e-12
    if denom_y == 0:
        denom_y = 1e-12

    x_norm = (x - leftX) / denom_x
    y_norm = (y - bottomY) / denom_y
    return x_norm, y_norm

def normalized_distance(p1, p2, leftX, rightX, bottomY, topY):
    x1n, y1n = normalize_point(p1[0], p1[1], leftX, rightX, bottomY, topY)
    x2n, y2n = normalize_point(p2[0], p2[1], leftX, rightX, bottomY, topY)
    return math.sqrt((x1n - x2n)**2 + (y1n - y2n)**2)

def find_and_match_closest_pairs(coords_extracted, coords_original,
                                 leftX, rightX, bottomY, topY):
    """Efficient greedy matching in normalized space.

    The previous implementation did repeated global min searches with nested loops,
    which can be extremely slow for large point sets.

    This uses a uniform grid in normalized space with cell size ~= MAX_NORM_DIST.
    For each original point, we search only nearby grid buckets.
    """
    if not coords_extracted or not coords_original:
        return [], list(coords_extracted), list(coords_original)

    cell = MAX_NORM_DIST
    if cell <= 0:
        cell = 0.1

    def cell_key(xn: float, yn: float) -> tuple[int, int]:
        return (int(math.floor(xn / cell)), int(math.floor(yn / cell)))

    # Build grid for extracted points
    grid: dict[tuple[int, int], list[tuple[float, float, tuple[float, float]]]] = {}
    for p_ex in coords_extracted:
        xn, yn = normalize_point(p_ex[0], p_ex[1], leftX, rightX, bottomY, topY)
        key = cell_key(xn, yn)
        grid.setdefault(key, []).append((xn, yn, p_ex))

    matched_pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    leftover_original: list[tuple[float, float]] = []

    for p_or in coords_original:
        xon, yon = normalize_point(p_or[0], p_or[1], leftX, rightX, bottomY, topY)
        base_key = cell_key(xon, yon)

        best_dist = float('inf')
        best_bucket_key = None
        best_bucket_idx = None
        best_p_ex = None

        # Search neighbor cells
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                k = (base_key[0] + dx, base_key[1] + dy)
                bucket = grid.get(k)
                if not bucket:
                    continue
                for idx, (xen, yen, p_ex) in enumerate(bucket):
                    dist = math.hypot(xon - xen, yon - yen)
                    if dist < best_dist:
                        best_dist = dist
                        best_bucket_key = k
                        best_bucket_idx = idx
                        best_p_ex = p_ex

        if best_p_ex is not None and best_dist <= MAX_NORM_DIST:
            matched_pairs.append((best_p_ex, p_or))
            bucket = grid.get(best_bucket_key)
            if bucket is not None:
                del bucket[best_bucket_idx]
                if not bucket:
                    del grid[best_bucket_key]
        else:
            leftover_original.append(p_or)

    # Remaining extracted points are leftovers
    leftover_extracted: list[tuple[float, float]] = []
    for bucket in grid.values():
        leftover_extracted.extend([p_ex for (_xn, _yn, p_ex) in bucket])

    return matched_pairs, leftover_extracted, leftover_original

def compute_mae(matched_pairs, leftX, rightX, bottomY, topY):
    """Compute MAE in normalized coords, returns (mae_x_pct, mae_y_pct)"""
    if matched_pairs:
        x_errors = []
        y_errors = []
        for (p_ex, p_or) in matched_pairs:
            x_exn, y_exn = normalize_point(p_ex[0], p_ex[1], leftX, rightX, bottomY, topY)
            x_orn, y_orn = normalize_point(p_or[0], p_or[1], leftX, rightX, bottomY, topY)
            x_errors.append(abs(x_orn - x_exn))
            y_errors.append(abs(y_orn - y_exn))

        mae_x = sum(x_errors) / len(x_errors)
        mae_y = sum(y_errors) / len(y_errors)
    else:
        mae_x = 0.0
        mae_y = 0.0

    return 100.0 * mae_x, 100.0 * mae_y

def plot_curve_comparison(ax, curve_label, df_extracted, df_original,
                          matched_pairs, leftover_extracted, leftover_original,
                          mae_x_pct, mae_y_pct, precision, recall,
                          leftX, rightX, bottomY, topY):
    """Plot a single curve comparison on the given axis."""
    
    # Plot extracted (blue line + markers)
    ax.plot(
        df_extracted['x'], df_extracted['y'],
        '-o', color='blue', linewidth=0.5, markersize=4, zorder=1,
        label='extracted'
    )

    # Plot original (red line + markers)
    ax.plot(
        df_original['x'], df_original['y'],
        '-o', color='red', linewidth=0.5, markersize=4, zorder=1,
        label='original'
    )

    # leftover_extracted => extra extracted points that didn't match any original
    if leftover_extracted:
        x_ex_left = [pt[0] for pt in leftover_extracted]
        y_ex_left = [pt[1] for pt in leftover_extracted]
        ax.scatter(
            x_ex_left, y_ex_left,
            marker='X', s=60, color='blue',
            edgecolors='black', linewidth=1.0,
            label=f'extra extr ({len(leftover_extracted)})',
            zorder=5
        )

    # leftover_original => original points not found in extracted
    if leftover_original:
        x_or_left = [pt[0] for pt in leftover_original]
        y_or_left = [pt[1] for pt in leftover_original]
        ax.scatter(
            x_or_left, y_or_left,
            marker='X', s=60, color='red',
            edgecolors='black', linewidth=1.0,
            label=f'missed orig ({len(leftover_original)})',
            zorder=5
        )

    # Draw arrows with consistent head size
    for (p_ex, p_or) in matched_pairs:
        arrow = mpatches.FancyArrowPatch(
            posA=(p_ex[0], p_ex[1]),
            posB=(p_or[0], p_or[1]),
            arrowstyle='->',
            mutation_scale=10,
            color='limegreen',
            alpha=0.8,
            linewidth=1,
            zorder=10
        )
        ax.add_patch(arrow)

    # Place MAE% text
    text_str = (
        f"MAE X: {mae_x_pct:.2f}%\n"
        f"MAE Y: {mae_y_pct:.2f}%\n"
        f"Prec: {precision:.2f}\n"
        f"Rec: {recall:.2f}"
    )
    ax.text(
        0.02, 0.98, text_str,
        transform=ax.transAxes,
        va='top', ha='left',
        fontsize=8, color='black',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )

    # Axes:
    # - Always start at origin.
    # - Treat caller-provided rightX/topY (> 0) as hard caps.
    # - If caps aren't provided, auto-expand to data maxima with "nice" rounding.
    if rightX > 0:
        eff_rightX = rightX
    else:
        x_max_data = max(
            float(pd.to_numeric(df_extracted['x'], errors='coerce').max()),
            float(pd.to_numeric(df_original['x'], errors='coerce').max()),
        )
        eff_rightX = _nice_upper_bound(x_max_data)

    if topY > 0:
        eff_topY = topY
    else:
        y_max_data = max(
            float(pd.to_numeric(df_extracted['y'], errors='coerce').max()),
            float(pd.to_numeric(df_original['y'], errors='coerce').max()),
        )
        eff_topY = _nice_upper_bound(y_max_data)

    ax.set_xlim(0.0, eff_rightX)
    ax.set_ylim(0.0, eff_topY)

    ax.set_title(curve_label, fontsize=10)
    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Y", fontsize=8)
    ax.legend(fontsize=6, loc='lower right')
    ax.grid(True, alpha=0.3)

def main():
    if len(sys.argv) < 7:
        print("Usage: python pointwise.py <extracted.csv> <original.csv> leftX rightX bottomY topY [output_dir]")
        sys.exit(1)

    file_extracted = sys.argv[1]
    file_original  = sys.argv[2]

    leftX   = float(sys.argv[3])
    rightX  = float(sys.argv[4])
    bottomY = float(sys.argv[5])
    topY    = float(sys.argv[6])
    
    # Optional output directory (if provided, outputs go there instead of extracted file's directory)
    if len(sys.argv) > 7:
        output_dir = sys.argv[7]
    else:
        output_dir = os.path.dirname(file_extracted)

    # 1. Load both CSV files
    print(f"Loading extracted CSV: {file_extracted}")
    extracted_curves = load_multi_curve_csv(file_extracted)
    print(f"  Found {len(extracted_curves)} curve(s): {list(extracted_curves.keys())}")

    print(f"Loading original CSV: {file_original}")
    original_curves = load_multi_curve_csv(file_original)
    print(f"  Found {len(original_curves)} curve(s): {list(original_curves.keys())}")

    # Keep user-provided axis limits as-is. We used to expand them to include all points,
    # but that breaks "hard cap" expectations in the UI.
    # We still force the plotting origin to (0,0) in plot_curve_comparison.

    # For visibility in logs/debugging, print the effective caps we're using.
    print(f"Effective plot bounds (caps): X=[0, {rightX}], Y=[0, {topY}]")

    # 2. Match curves (fast heuristic by default; optional LLM)
    matcher = _get_curve_matcher()
    if matcher == "llm":
        try:
            curve_mapping = match_curves_with_llm(original_curves, extracted_curves)
        except Exception as e:
            print(f"[WARN] LLM curve matching failed ({e}); falling back to heuristic.")
            curve_mapping = match_curves_heuristic(original_curves, extracted_curves)
    else:
        curve_mapping = match_curves_heuristic(original_curves, extracted_curves)
    print(f"Curve mapping: {curve_mapping}")

    # 3. Process each matched curve
    results = {}
    for orig_label, extr_label in curve_mapping.items():
        if orig_label not in original_curves:
            print(f"Warning: '{orig_label}' not found in original curves, skipping")
            continue
        if extr_label not in extracted_curves:
            print(f"Warning: '{extr_label}' not found in extracted curves, skipping")
            continue
        
        print(f"Processing '{orig_label}' vs '{extr_label}'... ", end='', flush=True)
        
        orig_info = original_curves[orig_label]
        extr_info = extracted_curves[extr_label]
        
        # Match pairs
        matched_pairs, leftover_extracted, leftover_original = find_and_match_closest_pairs(
            extr_info['coords'], orig_info['coords'],
            leftX, rightX, bottomY, topY
        )
        
        # Compute MAE
        mae_x_pct, mae_y_pct = compute_mae(matched_pairs, leftX, rightX, bottomY, topY)
        
        # Compute precision and recall
        num_matched = len(matched_pairs)
        num_extracted = len(extr_info['coords'])
        num_original = len(orig_info['coords'])
        
        precision = num_matched / num_extracted if num_extracted > 0 else 0.0
        recall = num_matched / num_original if num_original > 0 else 0.0
        
        results[orig_label] = {
            'extr_label': extr_label,
            'df_original': orig_info['data'],
            'df_extracted': extr_info['data'],
            'matched_pairs': matched_pairs,
            'leftover_extracted': leftover_extracted,
            'leftover_original': leftover_original,
            'mae_x_pct': mae_x_pct,
            'mae_y_pct': mae_y_pct,
            'num_matched': num_matched,
            'num_missing_orig': len(leftover_extracted),
            'num_missing_extr': len(leftover_original),
            'precision': precision,
            'recall': recall
        }
        
        print(f"MAE X={mae_x_pct:.2f}%, MAE Y={mae_y_pct:.2f}%, Prec={precision:.2f}, Rec={recall:.2f}")

    num_curves = len(results)
    if num_curves == 0:
        print("Error: No curves matched successfully")
        sys.exit(1)

    # 4. Determine subplot layout
    if num_curves <= 8:
        ncols = min(2, num_curves)  # Don't create more columns than curves
    else:
        ncols = 3
    nrows = (num_curves + ncols - 1) // ncols

    # Dynamic figure size
    fig_width = 6 * ncols
    fig_height = 5 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # 5. Plot each curve comparison
    all_mae_x = []
    all_mae_y = []
    all_precision = []
    all_recall = []
    
    for idx, (orig_label, result) in enumerate(results.items()):
        ax = axes[idx]
        
        plot_curve_comparison(
            ax, orig_label,
            result['df_extracted'],
            result['df_original'],
            result['matched_pairs'],
            result['leftover_extracted'],
            result['leftover_original'],
            result['mae_x_pct'],
            result['mae_y_pct'],
            result['precision'],
            result['recall'],
            leftX, rightX, bottomY, topY
        )
        
        all_mae_x.append(result['mae_x_pct'])
        all_mae_y.append(result['mae_y_pct'])
        all_precision.append(result['precision'])
        all_recall.append(result['recall'])

    # Hide unused subplots
    for idx in range(num_curves, len(axes)):
        axes[idx].set_visible(False)

    # Calculate mean MAE
    mean_mae_x = np.mean(all_mae_x)
    mean_mae_y = np.mean(all_mae_y)
    mean_precision = np.mean(all_precision)
    mean_recall = np.mean(all_recall)

    # Add overall title with mean MAE - use multiple lines for narrow figures
    if num_curves == 1:
        # For single curve, split stats across multiple lines
        fig.suptitle(
            f'Pointwise Comparison: Original vs Extracted\n'
            f'MAE X: {mean_mae_x:.2f}%  |  MAE Y: {mean_mae_y:.2f}%\n'
            f'Prec: {mean_precision:.2f}  |  Rec: {mean_recall:.2f}  |  Threshold: {MAX_NORM_DIST}',
            fontsize=12
        )
        plt.tight_layout(rect=[0, 0, 1, 0.90])
    else:
        fig.suptitle(
            f'Pointwise Comparison: Original vs Extracted\n'
            f'Mean MAE X: {mean_mae_x:.2f}%  |  Mean MAE Y: {mean_mae_y:.2f}%  |  '
            f'Mean Prec: {mean_precision:.2f}  |  Mean Rec: {mean_recall:.2f}  |  Threshold: {MAX_NORM_DIST}',
            fontsize=14
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 6. Save outputs
    f1 = os.path.basename(file_extracted)
    f2 = os.path.basename(file_original)
    base_ex = os.path.splitext(f1)[0]
    base_or = os.path.splitext(f2)[0]
    output_base = os.path.join(output_dir, f"pointwise_{base_ex}_VS_{base_or}")

    # Save figure
    plt.savefig(f"{output_base}.png", dpi=150)
    plt.close()
    print(f"Saved figure: {output_base}.png")

    # 7. Save statistics
    with open(f"{output_base}.stats", 'w', encoding='utf-8') as f:
        f.write("# Per-curve statistics\n")
        for orig_label, result in results.items():
            f.write(f"\nCurve '{orig_label}' -> '{result['extr_label']}':\n")
            f.write(f"  MAE X (percent): {result['mae_x_pct']:.2f}\n")
            f.write(f"  MAE Y (percent): {result['mae_y_pct']:.2f}\n")
            f.write(f"  Precision: {result['precision']:.4f}\n")
            f.write(f"  Recall: {result['recall']:.4f}\n")
            f.write(f"  MatchedPairs: {result['num_matched']}\n")
            f.write(f"  Extra extracted (unmatched): {result['num_missing_orig']}\n")
            f.write(f"  Missed original (not found): {result['num_missing_extr']}\n")
        
        f.write(f"\n# Summary\n")
        f.write(f"Number of curves: {num_curves}\n")
        f.write(f"Mean MAE X (percent): {mean_mae_x:.2f}\n")
        f.write(f"Mean MAE Y (percent): {mean_mae_y:.2f}\n")
        f.write(f"Mean Precision: {mean_precision:.4f}\n")
        f.write(f"Mean Recall: {mean_recall:.4f}\n")
        f.write(f"Threshold (MAX_NORM_DIST): {MAX_NORM_DIST}\n")

    print(f"Saved statistics: {output_base}.stats")
    print(f"\nMean MAE X: {mean_mae_x:.2f}%")
    print(f"Mean MAE Y: {mean_mae_y:.2f}%")
    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Recall: {mean_recall:.4f}")

if __name__ == "__main__":
    main()
