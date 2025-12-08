import sys
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

MAX_NORM_DIST = 0.2  # Example value; adjust as needed

def read_csv_coordinates_pandas(filepath):
    df = pd.read_csv(filepath, header=0, usecols=[0, 1], names=['x', 'y'])
    coords = list(zip(df['x'], df['y']))
    return df, coords

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
            # We decide not to match any more pairs
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

    # All remaining points become leftover
    leftover_extracted = data_extracted
    leftover_original  = data_original
    return matched_pairs, leftover_extracted, leftover_original

def plot_results(
    file_extracted,
    file_original,
    df_extracted,
    df_original,
    matched_pairs,
    leftover_extracted,
    leftover_original,
    mae_x_pct,
    mae_y_pct,
    leftX, rightX, bottomY, topY
):
    base_ex = os.path.splitext(os.path.basename(file_extracted))[0]
    base_or = os.path.splitext(os.path.basename(file_original))[0]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot extracted (blue line + markers)
    ax.plot(
        df_extracted['x'], df_extracted['y'],
        '-o', color='blue', linewidth=0.5, zorder=1,
        label='extracted'
    )

    # Plot original (red line + markers)
    ax.plot(
        df_original['x'], df_original['y'],
        '-o', color='red', linewidth=0.5, zorder=1,
        label='original'
    )

    # leftover_extracted => "missing from original"
    if leftover_extracted:
        x_ex_left = [pt[0] for pt in leftover_extracted]
        y_ex_left = [pt[1] for pt in leftover_extracted]
        ax.scatter(
            x_ex_left, y_ex_left,
            marker='X', s=80, color='blue',
            edgecolors='black', linewidth=1.0,
            label=f'missing from original ({len(leftover_extracted)})',
            zorder=5
        )

    # leftover_original => "missing from extracted"
    if leftover_original:
        x_or_left = [pt[0] for pt in leftover_original]
        y_or_left = [pt[1] for pt in leftover_original]
        ax.scatter(
            x_or_left, y_or_left,
            marker='X', s=80, color='red',
            edgecolors='black', linewidth=1.0,
            label=f'missing from extracted ({len(leftover_original)})',
            zorder=5
        )

    # Draw arrows with consistent head size using FancyArrowPatch
    for (p_ex, p_or) in matched_pairs:
        arrow = mpatches.FancyArrowPatch(
            posA=(p_ex[0], p_ex[1]),
            posB=(p_or[0], p_or[1]),
            arrowstyle='->',
            mutation_scale=15,  # consistent arrowhead size
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
        f"Threshold: {MAX_NORM_DIST} (norm)"
    )
    ax.text(
        0.02, 0.98, text_str,
        transform=ax.transAxes,
        va='top', ha='left',
        fontsize=10, color='black',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )

    # Set plot limits
    ax.set_xlim(leftX, rightX)
    ax.set_ylim(bottomY, topY)

    ax.set_title("Closest Pairs (Plot-Scaled Distances)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True)

    # Save and show
    output_filename = f"pointwise_{base_ex}_VS_{base_or}.png"
    plt.savefig(output_filename, dpi=300)
    #plt.show()
    plt.close()

def write_stats_file(
    base_ex, base_or,
    mae_x_pct, mae_y_pct,
    matched_pairs,
    leftover_count_extracted,
    leftover_count_original
):
    output_stats = f"pointwise_{base_ex}_VS_{base_or}.stats"
    with open(output_stats, 'w') as f:
        f.write(f"MAE X (percent): {mae_x_pct:.2f}\n")
        f.write(f"MAE Y (percent): {mae_y_pct:.2f}\n")
        f.write(f"MatchedPairs: {len(matched_pairs)}\n")
        f.write(f"Missing from original: {leftover_count_extracted}\n")
        f.write(f"Missing from extracted: {leftover_count_original}\n")

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

    # 1. Read CSV
    df_extracted, coords_extracted = read_csv_coordinates_pandas(file_extracted)
    df_original, coords_original   = read_csv_coordinates_pandas(file_original)

    # 2. Match pairs, with threshold-check
    matched_pairs, leftover_extracted, leftover_original = find_and_match_closest_pairs(
        coords_extracted, coords_original,
        leftX, rightX, bottomY, topY
    )

    leftover_count_extracted = len(leftover_extracted)  # "missing from original"
    leftover_count_original  = len(leftover_original)   # "missing from extracted"

    # 3. Compute MAE in normalized coords
    if matched_pairs:
        x_errors = []
        y_errors = []
        for (p_ex, p_or) in matched_pairs:
            x_exn, y_exn = normalize_point(p_ex[0], p_ex[1], leftX, rightX, bottomY, topY)
            x_orn, y_orn = normalize_point(p_or[0], p_or[1], leftX, rightX, bottomY, topY)
            x_errors.append(abs(x_orn - x_exn))
            y_errors.append(abs(y_orn - y_exn))

        mae_x = sum(x_errors) / len(x_errors)  # fraction in [0..1]
        mae_y = sum(y_errors) / len(y_errors)  # fraction in [0..1]
    else:
        mae_x = 0.0
        mae_y = 0.0

    mae_x_pct = 100.0 * mae_x
    mae_y_pct = 100.0 * mae_y

    # Print stats
    print(f"MAE X (as % of plot range): {mae_x_pct:.2f}")
    print(f"MAE Y (as % of plot range): {mae_y_pct:.2f}")
    print(f"MatchedPairs: {len(matched_pairs)}")
    print(f"Missing from original : {leftover_count_extracted}")
    print(f"Missing from extracted: {leftover_count_original}")

    # 4. Write stats
    base_ex = os.path.splitext(os.path.basename(file_extracted))[0]
    base_or = os.path.splitext(os.path.basename(file_original))[0]
    write_stats_file(
        base_ex, base_or,
        mae_x_pct, mae_y_pct,
        matched_pairs,
        leftover_count_extracted,
        leftover_count_original
    )

    # 5. Plot
    plot_results(
        file_extracted,
        file_original,
        df_extracted,
        df_original,
        matched_pairs,
        leftover_extracted,
        leftover_original,
        mae_x_pct,
        mae_y_pct,
        leftX, rightX, bottomY, topY
    )

if __name__ == "__main__":
    main()
