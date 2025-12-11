import sys
import os
import math
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import Mistral for LLM-based curve matching
from mistralai import Mistral
from dotenv import load_dotenv
load_dotenv(override=True)

MAX_NORM_DIST = 0.2  # Example value; adjust as needed

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

def load_multi_curve_csv(filepath):
    """
    Load a CSV with multiple curves (pairs of x,y columns).
    Returns a dict: {curve_label: {'data': DataFrame, 'x_label': str, 'y_label': str, 'coords': list}}
    """
    df = pd.read_csv(filepath, encoding='latin1')
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

def match_curves_with_llm(original_curves, extracted_curves):
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

def normalize_point(x, y, leftX, rightX, bottomY, topY):
    denom_x = rightX - leftX
    denom_y = topY - bottomY
    # Avoid zero division
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
    matched_pairs = []

    data_extracted = coords_extracted[:]
    data_original = coords_original[:]

    while data_extracted and data_original:
        min_dist = float('inf')
        best_pair = (None, None)

        # Find the closest pair in normalized space
        for i, p_ex in enumerate(data_extracted):
            for j, p_or in enumerate(data_original):
                dist = normalized_distance(p_ex, p_or, leftX, rightX, bottomY, topY)
                if dist < min_dist:
                    min_dist = dist
                    best_pair = (i, j)

        # If the best pair is still too far, stop matching
        if min_dist > MAX_NORM_DIST:
            break

        # Otherwise, match them
        i, j = best_pair
        p_ex = data_extracted[i]
        p_or = data_original[j]
        matched_pairs.append((p_ex, p_or))

        # Remove matched points
        if i > j:
            data_extracted.pop(i)
            data_original.pop(j)
        else:
            data_original.pop(j)
            data_extracted.pop(i)

    leftover_extracted = data_extracted
    leftover_original  = data_original
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

    # Set plot limits
    ax.set_xlim(leftX, rightX)
    ax.set_ylim(bottomY, topY)

    ax.set_title(curve_label, fontsize=10)
    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Y", fontsize=8)
    ax.legend(fontsize=6, loc='lower right')
    ax.grid(True, alpha=0.3)

def main():
    if len(sys.argv) < 7:
        print("Usage: python pointwise.py <extracted.csv> <original.csv> leftX rightX bottomY topY")
        sys.exit(1)

    file_extracted = sys.argv[1]
    file_original  = sys.argv[2]

    leftX   = float(sys.argv[3])
    rightX  = float(sys.argv[4])
    bottomY = float(sys.argv[5])
    topY    = float(sys.argv[6])

    # 1. Load both CSV files
    print(f"Loading extracted CSV: {file_extracted}")
    extracted_curves = load_multi_curve_csv(file_extracted)
    print(f"  Found {len(extracted_curves)} curve(s): {list(extracted_curves.keys())}")

    print(f"Loading original CSV: {file_original}")
    original_curves = load_multi_curve_csv(file_original)
    print(f"  Found {len(original_curves)} curve(s): {list(original_curves.keys())}")

    # 2. Match curves using LLM
    curve_mapping = match_curves_with_llm(original_curves, extracted_curves)
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
        ncols = 2
    else:
        ncols = 3
    nrows = (num_curves + ncols - 1) // ncols

    # Dynamic figure size
    fig_width = 6 * ncols
    fig_height = 5 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))

    # Flatten axes for easy iteration
    if num_curves == 1:
        axes = [axes]
    else:
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

    # Add overall title with mean MAE
    fig.suptitle(
        f'Pointwise Comparison: Original vs Extracted\n'
        f'Mean MAE X: {mean_mae_x:.2f}%  |  Mean MAE Y: {mean_mae_y:.2f}%  |  '
        f'Mean Prec: {mean_precision:.2f}  |  Mean Rec: {mean_recall:.2f}  |  Threshold: {MAX_NORM_DIST}',
        fontsize=14
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 6. Save outputs
    output_dir = os.path.dirname(file_extracted)
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
    with open(f"{output_base}.stats", 'w') as f:
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
