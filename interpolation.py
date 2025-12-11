import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import ast

# Import Mistral for LLM-based curve matching
from mistralai import Mistral
from dotenv import load_dotenv
load_dotenv(override=True)

if len(sys.argv) < 7:
    print("Usage: python interpolation.py <original.csv> <extracted.csv> leftX rightX bottomY topY")
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

def load_multi_curve_csv(filepath):
    """
    Load a CSV with multiple curves (pairs of x,y columns).
    Returns a dict: {curve_label: DataFrame with 'x' and 'y' columns}
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
        
        # Use the y-column header as the curve label
        curves[y_col] = {
            'data': curve_data,
            'x_label': x_col,
            'y_label': y_col
        }
    
    return curves

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

xrange_arg = rightX - leftX
yrange_arg = topY - bottomY

# Load both CSV files
print(f"Loading original CSV: {original_file}")
original_curves = load_multi_curve_csv(original_file)
print(f"  Found {len(original_curves)} curve(s): {list(original_curves.keys())}")

print(f"Loading extracted CSV: {extracted_file}")
extracted_curves = load_multi_curve_csv(extracted_file)
print(f"  Found {len(extracted_curves)} curve(s): {list(extracted_curves.keys())}")

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
    ncols = 2
else:
    ncols = 3
nrows = (num_curves + ncols - 1) // ncols  # Ceiling division

# Dynamic figure size
fig_width = 6 * ncols
fig_height = 4 * nrows
fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))

# Flatten axes for easy iteration (handle single subplot case)
if num_curves == 1:
    axes = [axes]
else:
    axes = axes.flatten()

# Plot each curve comparison
colors = plt.cm.tab10.colors
for idx, (orig_label, result) in enumerate(results.items()):
    ax = axes[idx]
    
    orig_data = result['orig_data']
    extr_data = result['extr_data']
    common_x = result['common_x']
    differences = result['differences']
    
    # Plot original and extracted curves
    ax.plot(orig_data['x'], orig_data['y'], label='original', linestyle='-', color=colors[0])
    ax.plot(extr_data['x'], extr_data['y'], label='llm', linestyle='-', color=colors[1])
    
    # Fill area for differences
    bottom_y = min(orig_data['y'].min(), extr_data['y'].min())
    ax.fill_between(common_x, np.abs(differences) + bottom_y, bottom_y, 
                    color='gray', alpha=0.5, label=f'MAE={result["mae"]:.4f}')
    
    # Add miss info to legend
    ax.scatter([None], [None], label=f'Miss L:{result["left_miss"]:.2f} R:{result["right_miss"]:.2f}')
    
    ax.set_title(f'{orig_label}')
    ax.set_xlabel(result['x_label'])
    ax.set_ylabel(orig_label)
    ax.legend(fontsize='small')
    ax.grid(True)
    
    # Set axis limits
    ax.set_xlim(min(leftX, ax.get_xlim()[0]), max(rightX, ax.get_xlim()[1]))
    ax.set_ylim(min(bottomY, ax.get_ylim()[0]), max(topY, ax.get_ylim()[1]))

# Hide unused subplots
for idx in range(num_curves, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Interpolation Comparison: Original vs Extracted Data', fontsize=14)
plt.tight_layout()

# Save outputs
output_dir = os.path.dirname(original_file)
f1 = os.path.basename(original_file)
f2 = os.path.basename(extracted_file)
output_base = os.path.join(output_dir, f"interpolated_{f1}_VS_{f2}")

# Save figure
plt.savefig(f"{output_base}.png", dpi=150)
plt.close()
print(f"Saved figure: {output_base}.png")

# Save statistics
with open(f"{output_base}.stats", 'w') as file:
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
