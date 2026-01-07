import os
import subprocess
import glob
import json
import numpy as np

# Set matplotlib backend to Agg (non-interactive) BEFORE importing pyplot
# This prevents threading issues with Tkinter when using Flask
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import io
import base64
import threading
import time
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
PROMPTS_DIR = os.path.join(BASE_DIR, 'prompts')
SYNTHETIC_DIR = os.path.join(PLOTS_DIR, 'synthetic')

# Create synthetic folder if it doesn't exist
os.makedirs(SYNTHETIC_DIR, exist_ok=True)

# Settings file for synthetic generator
SETTINGS_FILE = os.path.join(BASE_DIR, 'synthetic_settings.json')

# File to persist extraction results
EXTRACTION_STATE_FILE = os.path.join(BASE_DIR, 'extraction_state.json')

# =============================================================================
# Background Task Management
# =============================================================================

# In-memory task storage (for running tasks)
extraction_tasks = {}
extraction_tasks_lock = threading.Lock()

def load_extraction_state():
    """Load the last extraction result from file."""
    if os.path.exists(EXTRACTION_STATE_FILE):
        try:
            with open(EXTRACTION_STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading extraction state: {e}")
    return None

def save_extraction_state(state):
    """Save extraction result to file for persistence."""
    try:
        with open(EXTRACTION_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"Error saving extraction state: {e}")

# =============================================================================
# Synthetic Generator Configuration
# =============================================================================

DEFAULT_SETTINGS = {
    'num_curves': 3,
    'num_points': 8,
    'x_values_mode': 'auto',
    'x_values_manual': '0, 2, 4, 6, 8, 12, 18, 24',
    'x_spacing': 3,
    'x_label': 'Time',
    'x_unit': 'hours',
    'y_label': 'Bacterial Count',
    'y_unit': 'CFU/mL',
    'y_scale': 'log',
    'x_min': '',
    'x_max': '',
    'y_min': '',
    'y_max': '',
    'title': '',
    'show_legend': True,
    'show_grid': True,
    'figure_width': 10,
    'figure_height': 6,
    'dpi': 150,
    'save_svg': False,
    'curves': []
}

DEFAULT_CURVE = {
    'name': 'Condition',
    'initial_y': 6.0,
    'trend': 'stable',
    'trend_magnitude': 1.0,
    'noise_level': 0.1,
    'color': '#1f77b4',
    'marker': 'o',
    'line_style': '-',
    'show_line': True,
    'line_width': 1.5,
    'marker_size': 6
}

COLOR_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
]

# =============================================================================
# Synthetic Generator Helper Functions
# =============================================================================

def load_synthetic_settings():
    """Load settings from file, or return defaults if file doesn't exist."""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                saved = json.load(f)
                settings = DEFAULT_SETTINGS.copy()
                settings.update(saved)
                return settings
        except Exception as e:
            print(f"Error loading settings: {e}")
    return DEFAULT_SETTINGS.copy()

def save_synthetic_settings(settings):
    """Save settings to file for persistence."""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"Error saving settings: {e}")

def get_default_curves(num_curves):
    """Generate default curve configurations."""
    curves = []
    for i in range(num_curves):
        curve = DEFAULT_CURVE.copy()
        curve['name'] = f'Condition {i + 1}'
        curve['color'] = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        trends = ['stable', 'down', 'up', 'kill_regrowth', 'mixed']
        curve['trend'] = trends[i % len(trends)]
        curves.append(curve)
    return curves

def generate_x_values(settings):
    """Generate X-axis values based on settings."""
    if settings['x_values_mode'] == 'manual':
        try:
            x_vals = [float(x.strip()) for x in settings['x_values_manual'].split(',')]
            return np.array(x_vals)
        except:
            pass
    num_points = settings['num_points']
    spacing = settings['x_spacing']
    return np.arange(0, num_points * spacing, spacing)

def generate_curve_data(x_values, curve_config, y_scale='log'):
    """Generate Y-values for a single curve based on its configuration."""
    n_points = len(x_values)
    initial_y = curve_config['initial_y']
    trend = curve_config['trend']
    magnitude = curve_config['trend_magnitude']
    noise = curve_config['noise_level']
    
    x_norm = (x_values - x_values.min()) / (x_values.max() - x_values.min() + 1e-10)
    
    if trend == 'stable':
        base = np.zeros(n_points)
        for i in range(1, n_points):
            base[i] = base[i-1] + np.random.normal(0, 0.05 * magnitude)
    elif trend == 'down':
        decay_rate = 0.5 + magnitude * 0.5
        base = -decay_rate * x_norm * (2 + magnitude)
        base = base * (1 + 0.3 * np.sin(x_norm * np.pi))
    elif trend == 'up':
        growth_rate = 0.3 + magnitude * 0.3
        base = magnitude * (1 - np.exp(-growth_rate * x_norm * 5)) * 2
    elif trend == 'mixed':
        peak_pos = np.random.uniform(0.3, 0.7)
        base = np.zeros(n_points)
        for i, xn in enumerate(x_norm):
            if xn < peak_pos:
                base[i] = magnitude * (xn / peak_pos)
            else:
                base[i] = magnitude * (1 - (xn - peak_pos) / (1 - peak_pos))
    elif trend == 'kill_regrowth':
        nadir_pos = np.random.uniform(0.3, 0.5)
        nadir_depth = 1.5 + magnitude
        base = np.zeros(n_points)
        for i, xn in enumerate(x_norm):
            if xn < nadir_pos:
                base[i] = -nadir_depth * (xn / nadir_pos) ** 0.8
            else:
                regrowth_x = (xn - nadir_pos) / (1 - nadir_pos)
                regrowth_amount = (magnitude * 0.5) * regrowth_x ** 1.5
                base[i] = -nadir_depth + regrowth_amount
    else:
        base = np.zeros(n_points)
    
    noise_vals = np.random.normal(0, noise, n_points)
    
    if y_scale == 'log':
        y_values = initial_y + base + noise_vals
        y_values = np.maximum(y_values, 0.1)
    else:
        y_values = (10 ** initial_y) * (10 ** (base + noise_vals))
        y_values = np.maximum(y_values, 0)
    
    return y_values

def generate_all_curves(settings):
    """Generate data for all curves based on settings."""
    x_values = generate_x_values(settings)
    curves_data = []
    
    for curve_config in settings['curves']:
        y_values = generate_curve_data(x_values, curve_config, settings['y_scale'])
        curves_data.append({
            'x': x_values.tolist(),
            'y': y_values.tolist(),
            'config': curve_config
        })
    
    return x_values.tolist(), curves_data

def create_synthetic_plot(settings, curves_data, x_values):
    """Create the matplotlib figure based on settings and data."""
    # Helper: draw a broken line across the axis break
    def draw_broken_line(ax1, ax2, x_before, y_before, x_after, y_after, break_start, break_end, color, linestyle, linewidth):
        # Find the last point before the break and first after
        if len(x_before) == 0 or len(x_after) == 0:
            return
        x0, y0 = x_before[-1], y_before[-1]
        x1, y1 = x_after[0], y_after[0]
        # Calculate the gap width in data coordinates
        gap_left = break_start
        gap_right = break_end
        # Draw line from last point to gap_left (on ax1)
        if x0 < gap_left:
            ax1.plot([x0, gap_left], [y0, np.interp(gap_left, [x0, x1], [y0, y1])],
                     color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.7, zorder=10, dashes=(5, 5))
        # Draw line from gap_right to first point (on ax2)
        if x1 > gap_right:
            ax2.plot([gap_right, x1], [np.interp(gap_right, [x0, x1], [y0, y1]), y1],
                     color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.7, zorder=10, dashes=(5, 5))
    
    fig, ax = plt.subplots(figsize=(settings['figure_width'], settings['figure_height']))
    
    for curve_data in curves_data:
        config = curve_data['config']
        x = curve_data['x']
        y = curve_data['y']
        
        if settings['y_scale'] == 'log':
            y_plot = y
        else:
            y_plot = y
        
        if config['show_line']:
            ax.plot(x, y_plot,
                   linestyle=config['line_style'],
                   color=config['color'],
                   linewidth=config['line_width'],
                   label=config['name'])
        
        # Always draw markers with no line (line was drawn above if show_line is True)
        # Fix black border issue: if color is black, don't add edge color
        marker_kwargs = {
            'marker': config['marker'],
            'linestyle': 'none',
            'color': config['color'],
            'markersize': config['marker_size'],
            'label': config['name'] if not config['show_line'] else None
        }
        # If color is black (or very dark), set edge color to 'none' to avoid invisible border
        color_lower = config['color'].lower()
        if color_lower == '#000000' or color_lower == '#000' or color_lower == 'black':
            marker_kwargs['markeredgecolor'] = 'none'
        
        ax.plot(x, y_plot, **marker_kwargs)
    
    x_label = settings['x_label']
    if settings['x_unit']:
        x_label += f" ({settings['x_unit']})"
    ax.set_xlabel(x_label, fontsize=11)
    
    y_label = settings['y_label']
    if settings['y_unit']:
        y_label += f" ({settings['y_unit']})"
    ax.set_ylabel(y_label, fontsize=11)
    
    # Set axis limits - x starts at 0 by default
    x_min = 0
    if settings['x_min'] != '':
        try:
            x_min = max(0, float(settings['x_min']))
        except:
            x_min = 0
    
    if settings['x_max'] != '':
        try:
            x_max = float(settings['x_max'])
        except:
            all_x = [val for curve in curves_data for val in curve['x']]
            x_max = max(all_x) if all_x else 24
    else:
        all_x = [val for curve in curves_data for val in curve['x']]
        x_max = max(all_x) if all_x else 24
    
    # Calculate y limits first (needed for both regular and broken axis)
    if settings['y_scale'] == 'log':
        if settings['y_min'] != '':
            try:
                y_min = float(settings['y_min'])
            except:
                y_min = 0
        else:
            y_min = 0
        
        if settings['y_max'] != '':
            try:
                y_max = float(settings['y_max'])
            except:
                all_y = [val for curve in curves_data for val in curve['y']]
                y_max = max(all_y) + 1
        else:
            all_y = [val for curve in curves_data for val in curve['y']]
            y_max = max(all_y) + 1
        
        ax.set_ylim(y_min, y_max)
        
        y_range = y_max - y_min
        if y_range <= 6:
            tick_spacing = 1
        elif y_range <= 12:
            tick_spacing = 2
        else:
            tick_spacing = max(1, int(y_range / 6))
        
        y_ticks = np.arange(int(y_min), int(y_max) + 1, tick_spacing)
        ax.set_yticks(y_ticks)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        ax.spines['bottom'].set_position(('data', y_min))
    else:
        y_min = 0
        if settings['y_min'] != '' and settings['y_max'] != '':
            try:
                y_min = float(settings['y_min'])
                y_max = float(settings['y_max'])
                ax.set_ylim(y_min, y_max)
            except:
                all_y = [val for curve in curves_data for val in curve['y']]
                y_max = max(all_y) + 1 if all_y else 1
                ax.set_ylim(y_min, y_max)
        else:
            all_y = [val for curve in curves_data for val in curve['y']]
            y_max = max(all_y) + 1 if all_y else 1
            ax.set_ylim(y_min, y_max)
        ax.spines['bottom'].set_position(('data', y_min))
    
    # Handle axis break if enabled
    axis_break_enabled = settings.get('axis_break_enabled', False)
    # Handle both boolean and string 'true'/'false' from JSON
    if isinstance(axis_break_enabled, str):
        axis_break_enabled = axis_break_enabled.lower() in ('true', '1', 'yes')
    axis_break_type = settings.get('axis_break_type', 'x')
    axis_break_start = settings.get('axis_break_start', '')
    axis_break_end = settings.get('axis_break_end', '')
    
    # Handle both string and numeric inputs from JavaScript
    axis_break_start_valid = (axis_break_start is not None and 
                             axis_break_start != '' and 
                             str(axis_break_start).strip() != '')
    axis_break_end_valid = (axis_break_end is not None and 
                           axis_break_end != '' and 
                           str(axis_break_end).strip() != '')
    
    if axis_break_enabled and axis_break_type == 'x' and axis_break_start_valid and axis_break_end_valid:
        try:
            # Convert to float (handles both string and numeric)
            break_start = float(axis_break_start)
            break_end = float(axis_break_end)

            if break_start < break_end and x_min < break_start and break_end < x_max:
                # Create a broken x-axis using subplots (y-axis only on left, x-axis break only on bottom)
                fig.clf()
                y_plot_min = y_min
                y_plot_max = y_max
                from matplotlib import gridspec
                gs = gridspec.GridSpec(1, 2, width_ratios=[break_start - x_min, x_max - break_end], 
                                     wspace=0.05, left=0.1, right=0.95, top=0.9, bottom=0.1)
                ax1 = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
                # Hide the spines between the axes, but keep top border continuous
                ax1.spines['right'].set_visible(False)
                ax2.spines['left'].set_visible(False)
                # Ensure top spines are visible (no gap at top)
                ax1.spines['top'].set_visible(True)
                ax2.spines['top'].set_visible(True)
                # Only show y-axis on the left
                ax2.yaxis.set_visible(False)
                ax1.yaxis.tick_left()
                ax2.tick_params(labelleft=False)
                # Only show x-axis ticks on bottom, and remove any top x-axis
                ax1.xaxis.set_ticks_position('bottom')
                ax2.xaxis.set_ticks_position('bottom')
                ax1.xaxis.set_ticks([] if not (len(ax1.get_xticks()) > 0) else ax1.get_xticks())
                ax2.xaxis.set_ticks([] if not (len(ax2.get_xticks()) > 0) else ax2.get_xticks())
                ax1.xaxis.set_label_position('bottom')
                ax2.xaxis.set_label_position('bottom')
                ax1.xaxis.set_visible(True)
                ax2.xaxis.set_visible(True)
                # Remove any top x-axis
                ax1.xaxis.set_ticks_position('bottom')
                ax2.xaxis.set_ticks_position('bottom')
                # Add diagonal lines to indicate break (bottom only)
                d = 0.015  # Size of diagonal lines
                kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linewidth=1)
                # Bottom break (right edge of left axis)
                ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
                # No top break
                kwargs2 = dict(transform=ax2.transAxes, color='k', clip_on=False, linewidth=1)
                # Bottom break (left edge of right axis)
                ax2.plot((-d, +d), (-d, +d), **kwargs2)
                # No top break
                # Plot curves on both axes
                for curve_data in curves_data:
                    config = curve_data['config']
                    x = np.array(curve_data['x'])
                    y = np.array(curve_data['y'])
                    y_plot = y
                    mask_before = x <= break_start
                    mask_after = x >= break_end
                    marker_kwargs = {
                        'marker': config['marker'],
                        'linestyle': 'none',
                        'color': config['color'],
                        'markersize': config['marker_size']
                    }
                    color_lower = config['color'].lower()
                    if color_lower == '#000000' or color_lower == '#000' or color_lower == 'black':
                        marker_kwargs['markeredgecolor'] = 'none'
                    # Plot on first axis
                    x_before = x[mask_before]
                    y_before = y_plot[mask_before]
                    if len(x_before) > 0:
                        if config['show_line']:
                            ax1.plot(x_before, y_before,
                                   linestyle=config['line_style'],
                                   color=config['color'],
                                   linewidth=config['line_width'],
                                   label=config['name'])
                        ax1.plot(x_before, y_before, **marker_kwargs)
                    # Plot on second axis
                    x_after = x[mask_after]
                    y_after = y_plot[mask_after]
                    if len(x_after) > 0:
                        if config['show_line']:
                            ax2.plot(x_after, y_after,
                                   linestyle=config['line_style'],
                                   color=config['color'],
                                   linewidth=config['line_width'],
                                   label=None)
                        marker_kwargs_copy = marker_kwargs.copy()
                        marker_kwargs_copy['label'] = None
                        ax2.plot(x_after, y_after, **marker_kwargs_copy)
                    # Draw broken connecting line if line is enabled and both sides exist
                    if config['show_line'] and len(x_before) > 0 and len(x_after) > 0:
                        draw_broken_line(
                            ax1, ax2,
                            x_before, y_before,
                            x_after, y_after,
                            break_start, break_end,
                            config['color'], config['line_style'], config['line_width']
                        )
                # Set axis limits
                ax1.set_xlim(x_min, break_start)
                ax2.set_xlim(break_end, x_max)
                ax1.set_ylim(y_plot_min, y_plot_max)
                ax2.set_ylim(y_plot_min, y_plot_max)
                # Set x-axis ticks for broken axis
                x_tick_mode_break = settings.get('x_tick_mode', 'custom')
                x_tick_interval_break = settings.get('x_tick_interval', 2)
                if isinstance(x_tick_interval_break, str):
                    try:
                        x_tick_interval_break = float(x_tick_interval_break)
                    except:
                        x_tick_interval_break = 2
                elif x_tick_interval_break is None or x_tick_interval_break == 0:
                    x_tick_interval_break = 2
                if x_tick_mode_break == 'custom' and x_tick_interval_break > 0:
                    tick_interval = float(x_tick_interval_break)
                    ticks_before = np.arange(x_min, break_start + tick_interval, tick_interval)
                    ticks_before = ticks_before[(ticks_before >= x_min) & (ticks_before <= break_start)]
                    ticks_before = [tick for tick in ticks_before if int(round(tick)) % 2 == 0]
                    first_tick_after = int(np.ceil(break_end))
                    if first_tick_after % 2 != 0:
                        first_tick_after += 1
                    ticks_after = np.arange(first_tick_after, x_max + tick_interval, tick_interval)
                    ticks_after = [tick for tick in ticks_after if int(round(tick)) % 2 == 0 and tick >= break_end and tick <= x_max]
                    if len(ticks_before) > 0:
                        ax1.set_xticks(ticks_before)
                    if len(ticks_after) > 0:
                        ax2.set_xticks(ticks_after)
                # Set labels
                x_label = settings['x_label']
                if settings['x_unit']:
                    x_label += f" ({settings['x_unit']})"
                fig.text(0.5, 0.02, x_label, ha='center', fontsize=11)
                y_label = settings['y_label']
                if settings['y_unit']:
                    y_label += f" ({settings['y_unit']})"
                ax1.set_ylabel(y_label, fontsize=11)
                # Title
                if settings['title']:
                    fig.suptitle(settings['title'], fontsize=12, fontweight='bold')
                # Legend
                if settings['show_legend']:
                    ax1.legend(loc='best', framealpha=0.9)
                # Grid (only on left axis)
                if settings['show_grid']:
                    ax1.grid(True, alpha=0.3, linestyle='--')
                plt.tight_layout()
                return fig
        except Exception as e:
            print(f"Warning: Could not create axis break: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to regular plot
    
    # Regular plot (no axis break)
    ax.set_xlim(x_min, x_max)
    ax.spines['left'].set_position(('data', x_min))
    
    # Set x-axis tick marks (default: every 2 hours)
    x_tick_mode = settings.get('x_tick_mode', 'custom')
    x_tick_interval = settings.get('x_tick_interval', 2)
    
    # Convert to appropriate types if needed
    if isinstance(x_tick_interval, str):
        try:
            x_tick_interval = float(x_tick_interval)
        except:
            x_tick_interval = 2
    elif x_tick_interval is None or x_tick_interval == 0:
        x_tick_interval = 2
    
    # Apply custom tick marks if specified (default is custom with interval 2)
    if x_tick_mode == 'custom' and x_tick_interval > 0:
        tick_interval = float(x_tick_interval)
        # Generate ticks starting from x_min
        x_ticks = np.arange(x_min, x_max + tick_interval, tick_interval)
        # Filter ticks to be within the x-axis range
        x_ticks = x_ticks[(x_ticks >= x_min) & (x_ticks <= x_max)]
        if len(x_ticks) > 0:
            ax.set_xticks(x_ticks)
            ax.tick_params(axis='x', which='major', labelsize=10)
    # Otherwise, use auto ticks (matplotlib default)
    
    if settings['title']:
        ax.set_title(settings['title'], fontsize=12, fontweight='bold')
    
    if settings['show_legend']:
        ax.legend(loc='best', framealpha=0.9)
    
    if settings['show_grid']:
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for web display."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

def get_next_synthetic_name():
    """Generate the next available name for a synthetic plot (AA, AB, AC, ...)."""
    import string
    existing = set()
    if os.path.exists(SYNTHETIC_DIR):
        for item in os.listdir(SYNTHETIC_DIR):
            if os.path.isdir(os.path.join(SYNTHETIC_DIR, item)):
                existing.add(item.upper())
    
    for first in string.ascii_uppercase:
        for second in string.ascii_uppercase:
            name = f"{first}{second}"
            if name not in existing:
                return name
    
    from datetime import datetime
    return f"ZZ_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def save_synthetic_plot_and_data(settings, curves_data, x_values):
    """Save the plot as PNG and the data as CSV."""
    base_filename = get_next_synthetic_name()
    plot_folder = os.path.join(SYNTHETIC_DIR, base_filename)
    os.makedirs(plot_folder, exist_ok=True)
    
    fig = create_synthetic_plot(settings, curves_data, x_values)
    
    png_path = os.path.join(plot_folder, f'{base_filename}.png')
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    
    svg_path = None
    if settings.get('save_svg', False):
        svg_path = os.path.join(plot_folder, f'{base_filename}.svg')
        fig.savefig(svg_path, format='svg', bbox_inches='tight')
    
    plt.close(fig)
    
    csv_path = os.path.join(plot_folder, f'{base_filename}-original.csv')
    save_synthetic_csv(curves_data, x_values, settings, csv_path)
    
    return {
        'png': png_path,
        'svg': svg_path,
        'csv': csv_path,
        'filename': base_filename,
        'folder': plot_folder
    }

def save_synthetic_csv(curves_data, x_values, settings, filepath):
    """Save curve data to CSV file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        headers = []
        for curve_data in curves_data:
            config = curve_data['config']
            x_col = settings['x_label'] if settings['x_label'] else 'x'
            y_col = config['name']
            headers.extend([x_col, y_col])
        f.write(','.join(headers) + '\n')
        
        n_points = len(x_values)
        for i in range(n_points):
            row = []
            for curve_data in curves_data:
                x_val = curve_data['x'][i]
                y_val = curve_data['y'][i]
                row.extend([str(x_val), f'{y_val:.4f}'])
            f.write(','.join(row) + '\n')

# =============================================================================
# Plot Extraction Helper Functions
# =============================================================================

def get_input_images(directory):
    """
    Recursively find only original input image files (.png, .svg) in the plots directory.
    Excludes output files like replot, comparison, interpolated, pointwise images.
    Also excludes files inside version folders (e.g., A-1.p2.v1/)
    Returns a dict grouped by top-level folder for easier selection.
    """
    images_set = set()  # Use set to avoid duplicates on case-insensitive filesystems
    exclude_patterns = [
        '-replot', 'comparison_', 'interpolated_', 'pointwise_',
        '.mistral.out', '.claude.out', '_VS_'
    ]
    # Pattern to detect version folders: {name}.p{n}.v{n}
    import re
    version_folder_pattern = re.compile(r'\.p\d+\.v\d+[\\/]')
    
    for ext in ['*.png', '*.PNG', '*.svg', '*.SVG']:
        for img_path in glob.glob(os.path.join(directory, '**', ext), recursive=True):
            rel_path = os.path.relpath(img_path, directory)
            filename = os.path.basename(img_path)
            
            # Skip if inside a version folder
            if version_folder_pattern.search(rel_path):
                continue
            
            # Skip if filename contains any exclude pattern
            if not any(pattern in filename for pattern in exclude_patterns):
                images_set.add(os.path.relpath(img_path, PLOTS_DIR).replace('\\', '/'))
    
    # Group images by top-level folder
    grouped = {}
    for img_path in sorted(images_set):
        parts = img_path.split('/')
        if len(parts) > 1:
            folder = parts[0]
        else:
            folder = '(root)'
        
        if folder not in grouped:
            grouped[folder] = []
        grouped[folder].append(img_path)
    
    return grouped

def get_prompts():
    """Get all prompt files from the prompts directory."""
    prompts = []
    for f in os.listdir(PROMPTS_DIR):
        if f.endswith('.py') and not f.startswith('__'):
            prompts.append(f)
    return sorted(prompts)

def get_csv_paths(image_path):
    """
    Given an image path like 'first_examples/A/A-1/A-1.png',
    return the expected paths for original and extracted CSVs.
    """
    image_dir = os.path.dirname(image_path)
    image_name = os.path.basename(image_path)
    base_name = os.path.splitext(image_name)[0]  # e.g., 'A-1'
    
    original_csv = os.path.join(image_dir, f"{base_name}-original.csv")
    
    return {
        'original': original_csv,
        'base_name': base_name,
        'image_dir': image_dir
    }

def find_extracted_csv(image_path, prompt_file):
    """Find the actual extracted data file for a given image and prompt.
    Now searches inside version folders and returns the latest version.
    Files have version in filename: {image}.{prompt}.v{n}.mistral.out_data
    Folder names use underscores instead of dots for the extension: A-1_png.p2.v1/"""
    import re
    
    image_dir = os.path.dirname(os.path.join(PLOTS_DIR, image_path))
    image_name = os.path.basename(image_path)
    base_name = os.path.splitext(image_name)[0]
    # Folder naming uses underscore: A-1.png -> A-1_png
    name_for_folder = image_name.replace('.', '_')
    
    # Get prompt short name (e.g., prompt_1.py -> p1)
    prompt_name = os.path.splitext(prompt_file)[0].replace('prompt_', 'p')
    
    # Look for version folders matching this image+prompt (new format with extension underscore)
    # Also check old format (without extension) for backwards compatibility
    version_pattern_new = re.compile(rf'^{re.escape(name_for_folder)}\.{re.escape(prompt_name)}\.v(\d+)$')
    version_pattern_old = re.compile(rf'^{re.escape(base_name)}\.{re.escape(prompt_name)}\.v(\d+)$')
    
    latest_version = 0
    latest_file = None
    
    if os.path.exists(image_dir):
        for item in os.listdir(image_dir):
            # Try new format first
            match = version_pattern_new.match(item)
            is_new_format = True
            if not match:
                # Fall back to old format
                match = version_pattern_old.match(item)
                is_new_format = False
            
            if match:
                version_num = int(match.group(1))
                version_dir = os.path.join(image_dir, item)
                # Filename still includes version: {image}.{prompt}.v{n}.mistral.out_data
                extracted_file = os.path.join(version_dir, f"{image_name}.{prompt_name}.v{version_num}.mistral.out_data")
                
                if os.path.exists(extracted_file) and version_num > latest_version:
                    latest_version = version_num
                    latest_file = extracted_file
    
    if latest_file:
        return os.path.relpath(latest_file, PLOTS_DIR).replace('\\', '/')
    return None

def get_output_files(image_path, prompt_file=None, version_dir=None):
    """Get output files related to an image. If version_dir is provided, only show that version."""
    image_dir = os.path.dirname(os.path.join(PLOTS_DIR, image_path))
    image_name = os.path.basename(image_path)
    base_name = os.path.splitext(image_name)[0]
    
    outputs = {
        'images': [],
        'stats': [],
        'data': [],
        'other': [],
        'summary': {}
    }
    
    if not os.path.exists(image_dir):
        return outputs
    
    # Add the original image
    original_path = os.path.join(image_dir, image_name)
    if os.path.exists(original_path):
        outputs['images'].append({
            'path': os.path.relpath(original_path, PLOTS_DIR).replace('\\', '/'),
            'label': 'Original Input',
            'filename': image_name
        })
    
    # Add original CSV if exists
    original_csv = os.path.join(image_dir, f"{base_name}-original.csv")
    if os.path.exists(original_csv):
        outputs['data'].append({
            'path': os.path.relpath(original_csv, PLOTS_DIR).replace('\\', '/'),
            'label': 'Original Data',
            'filename': f"{base_name}-original.csv"
        })
    
    # If version_dir is specified, only scan that folder
    if version_dir and os.path.exists(version_dir):
        version_label = os.path.basename(version_dir)
        _scan_version_folder(version_dir, version_label, outputs, PLOTS_DIR)
        outputs['summary'] = _parse_summary_stats(version_dir)
    else:
        # Scan for all version folders matching pattern: {base_name}.p*.v* or {name_for_folder}.p*.v*
        # name_for_folder uses underscore instead of dot for extension (A-1_png)
        import re
        name_for_folder = image_name.replace('.', '_')
        version_pattern_new = re.compile(rf'^{re.escape(name_for_folder)}\.p\d+\.v\d+$')
        version_pattern_old = re.compile(rf'^{re.escape(base_name)}\.p\d+\.v\d+$')
        
        for item in os.listdir(image_dir):
            item_path = os.path.join(image_dir, item)
            if os.path.isdir(item_path) and (version_pattern_new.match(item) or version_pattern_old.match(item)):
                version_label = item
                _scan_version_folder(item_path, version_label, outputs, PLOTS_DIR)
                # Use summary from the last scanned folder
                outputs['summary'] = _parse_summary_stats(item_path)
    
    return outputs

def _scan_version_folder(folder_path, version_label, outputs, plots_dir):
    """Helper to scan a version folder and add files to outputs."""
    for f in os.listdir(folder_path):
        full_path = os.path.join(folder_path, f)
        rel_path = os.path.relpath(full_path, plots_dir).replace('\\', '/')
        
        label = None
        
        if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.svg'):
            if '-replot' in f:
                label = f'Extracted Replot ({version_label})'
            elif f.startswith('comparison_'):
                label = f'Comparison ({version_label})'
            elif f.startswith('interpolated_'):
                label = f'Interpolation ({version_label})'
            elif f.startswith('pointwise_'):
                label = f'Pointwise ({version_label})'
            else:
                label = f'Output Image ({version_label})'
            
            outputs['images'].append({'path': rel_path, 'label': label, 'filename': f})
            
        elif f.endswith('.stats'):
            if 'interpolated_' in f:
                label = f'Interpolation Stats ({version_label})'
            elif 'pointwise_' in f:
                label = f'Pointwise Stats ({version_label})'
            else:
                label = f'Statistics ({version_label})'
            outputs['stats'].append({'path': rel_path, 'label': label, 'filename': f})
            
        elif f.endswith('_data'):
            outputs['data'].append({'path': rel_path, 'label': f'Extracted Data ({version_label})', 'filename': f})
            
        elif f.endswith('_code') or f.endswith('_conversation') or f.endswith('_validate') or f.endswith('_validate_why'):
            outputs['other'].append({'path': rel_path, 'filename': f, 'version': version_label})

def _parse_validation_why(code):
    """Convert validation code like 'X; N; T' to human-readable reasons."""
    reasons_map = {
        'X': 'X-axis',
        'Y': 'Y-axis', 
        'N': 'Number of points',
        'T': 'Trends'
    }
    if not code:
        return 'N/A'
    parts = [p.strip() for p in code.replace(';', ',').split(',') if p.strip()]
    reasons = [reasons_map.get(p, p) for p in parts]
    return ', '.join(reasons) if reasons else 'N/A'

def _parse_summary_stats(folder_path):
    """Parse validation and comparison stats from a version folder."""
    summary = {
        'validation_result': None,
        'validation_reason': None,
        'interpolation_mae': None,
        'pointwise_mae_x': None,
        'pointwise_mae_y': None,
        'precision': None,
        'recall': None
    }
    
    if not folder_path or not os.path.exists(folder_path):
        return summary
    
    # Find and parse validation files
    for f in os.listdir(folder_path):
        full_path = os.path.join(folder_path, f)
        
        if f.endswith('_validate') and not f.endswith('_validate_why'):
            try:
                with open(full_path, 'r') as file:
                    content = file.read().strip().lower()
                    summary['validation_result'] = 'Yes' if 'yes' in content else 'No'
            except:
                pass
                
        elif f.endswith('_validate_why'):
            try:
                with open(full_path, 'r') as file:
                    content = file.read().strip()
                    summary['validation_reason'] = _parse_validation_why(content)
            except:
                pass
                
        elif f.endswith('.stats') and 'interpolated_' in f:
            try:
                with open(full_path, 'r', encoding='latin1') as file:
                    for line in file:
                        if 'Mean MAE:' in line:
                            val = line.split(':')[1].strip()
                            summary['interpolation_mae'] = float(val)
                            break
            except:
                pass
                
        elif f.endswith('.stats') and 'pointwise_' in f:
            try:
                with open(full_path, 'r', encoding='latin1') as file:
                    for line in file:
                        if 'Mean MAE X (percent):' in line:
                            val = line.split(':')[1].strip()
                            summary['pointwise_mae_x'] = f"{float(val):.2f}%"
                        elif 'Mean MAE Y (percent):' in line:
                            val = line.split(':')[1].strip()
                            summary['pointwise_mae_y'] = f"{float(val):.2f}%"
                        elif 'Mean Precision:' in line:
                            val = line.split(':')[1].strip()
                            summary['precision'] = f"{float(val) * 100:.1f}%"
                        elif 'Mean Recall:' in line:
                            val = line.split(':')[1].strip()
                            summary['recall'] = f"{float(val) * 100:.1f}%"
            except:
                pass
    
    # If validation was Yes, set reason to N/A
    if summary['validation_result'] == 'Yes':
        summary['validation_reason'] = 'N/A'
    
    return summary

def check_csv_exists(image_path, prompt_file=None):
    """Check if original and extracted CSVs exist."""
    csv_info = get_csv_paths(image_path)
    
    original_full = os.path.join(PLOTS_DIR, csv_info['original'])
    original_exists = os.path.exists(original_full)
    
    extracted_exists = False
    extracted_path = None
    
    if prompt_file:
        extracted_rel = find_extracted_csv(image_path, prompt_file)
        if extracted_rel:
            extracted_exists = True
            extracted_path = extracted_rel
    
    return {
        'original': {
            'path': csv_info['original'],
            'exists': original_exists
        },
        'extracted': {
            'path': extracted_path,
            'exists': extracted_exists
        },
        'base_name': csv_info['base_name'],
        'image_dir': csv_info['image_dir']
    }

@app.route('/')
def index():
    images_grouped = get_input_images(PLOTS_DIR)
    prompts = get_prompts()
    return render_template('index.html', images_grouped=images_grouped, prompts=prompts)

@app.route('/plots/<path:filename>')
def serve_plot(filename):
    """Serve files from the plots directory."""
    return send_from_directory(PLOTS_DIR, filename)

@app.route('/check_csv', methods=['POST'])
def check_csv():
    """Check if CSV files exist for the selected image."""
    image_path = request.json.get('image_path')
    prompt_file = request.json.get('prompt_file')
    
    result = check_csv_exists(image_path, prompt_file)
    return jsonify(result)

@app.route('/get_axis_ranges', methods=['POST'])
def get_axis_ranges():
    """Get axis ranges from the original CSV file for an image."""
    import pandas as pd
    
    image_path = request.json.get('image_path')
    
    # Get the original CSV path
    csv_paths = get_csv_paths(os.path.join(PLOTS_DIR, image_path))
    original_csv_path = os.path.join(PLOTS_DIR, csv_paths['original'])
    
    if not os.path.exists(original_csv_path):
        return jsonify({
            'success': False, 
            'error': f'Original CSV not found: {csv_paths["original"]}',
            'has_original': False
        })
    
    try:
        # Read CSV and auto-detect axis ranges
        df = pd.read_csv(original_csv_path)
        
        # First column is X values
        x_col = df.columns[0]
        x_values = df[x_col].dropna()
        
        # Find Y columns (all except first)
        y_cols = df.columns[1:]
        y_values = df[y_cols].values.flatten()
        y_values = y_values[~pd.isna(y_values)]
        
        # Get min/max with some padding for better visualization
        left_x = float(x_values.min())
        right_x = float(x_values.max())
        bottom_y = float(y_values.min())
        top_y = float(y_values.max())
        
        return jsonify({
            'success': True,
            'has_original': True,
            'leftX': left_x,
            'rightX': right_x,
            'bottomY': bottom_y,
            'topY': top_y,
            'csv_path': csv_paths['original']
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'has_original': True
        })

@app.route('/get_outputs', methods=['POST'])
def get_outputs():
    """Get output files for a selected image. Shows latest version by default."""
    import re
    
    image_path = request.json.get('image_path')
    prompt_file = request.json.get('prompt_file')
    version_dir_param = request.json.get('version_dir')  # Optional: specific version to show
    
    # If no specific version requested, find the latest version for this image+prompt
    if not version_dir_param and prompt_file:
        image_dir = os.path.dirname(os.path.join(PLOTS_DIR, image_path))
        image_name = os.path.basename(image_path)
        base_name = os.path.splitext(image_name)[0]
        name_for_folder = image_name.replace('.', '_')  # New format with extension underscore
        prompt_name = os.path.splitext(prompt_file)[0].replace('prompt_', 'p')
        
        # Find latest version folder (check both new and old formats)
        version_pattern_new = re.compile(rf'^{re.escape(name_for_folder)}\.{re.escape(prompt_name)}\.v(\d+)$')
        version_pattern_old = re.compile(rf'^{re.escape(base_name)}\.{re.escape(prompt_name)}\.v(\d+)$')
        latest_version = 0
        latest_dir = None
        
        if os.path.exists(image_dir):
            for item in os.listdir(image_dir):
                match = version_pattern_new.match(item) or version_pattern_old.match(item)
                if match:
                    version_num = int(match.group(1))
                    if version_num > latest_version:
                        latest_version = version_num
                        latest_dir = os.path.join(image_dir, item)
        
        version_dir_param = latest_dir
    
    outputs = get_output_files(image_path, prompt_file, version_dir_param)
    return jsonify({'outputs': outputs, 'version_dir': version_dir_param})

@app.route('/read_file', methods=['POST'])
def read_file_route():
    """Read contents of a text file."""
    file_path = request.json.get('file_path')
    # Convert forward slashes to OS-appropriate separators
    file_path = file_path.replace('/', os.sep)
    full_path = os.path.join(PLOTS_DIR, file_path)
    try:
        with open(full_path, 'r', encoding='latin1') as f:
            content = f.read()
        return jsonify({'success': True, 'content': content})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/run_all', methods=['POST'])
def run_all():
    """
    Start extraction in background and return task ID immediately.
    Client will poll /task_status/<task_id> for updates.
    """
    import re
    
    data = request.json
    image_path = data.get('image')
    prompt_file = data.get('prompt')
    run_interpolation = data.get('runInterpolation', False)
    run_pointwise = data.get('runPointwise', False)
    left_x = str(data.get('leftX', 0))
    right_x = str(data.get('rightX', 100))
    bottom_y = str(data.get('bottomY', 0))
    top_y = str(data.get('topY', 100))
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())[:8]
    
    # Initialize task state
    with extraction_tasks_lock:
        extraction_tasks[task_id] = {
            'status': 'running',
            'progress': 'Starting...',
            'console': [],
            'started_at': time.time(),
            'image_path': image_path,
            'prompt_file': prompt_file
        }
    
    # Start background thread
    thread = threading.Thread(
        target=run_extraction_task,
        args=(task_id, image_path, prompt_file, run_interpolation, run_pointwise, 
              left_x, right_x, bottom_y, top_y)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({'task_id': task_id, 'status': 'started'})

def run_extraction_task(task_id, image_path, prompt_file, run_interpolation, run_pointwise,
                        left_x, right_x, bottom_y, top_y):
    """Background task that runs the extraction pipeline."""
    import re
    
    def update_task(progress=None, console_line=None):
        with extraction_tasks_lock:
            if task_id in extraction_tasks:
                if progress:
                    extraction_tasks[task_id]['progress'] = progress
                if console_line:
                    extraction_tasks[task_id]['console'].append(console_line)
    
    total_start_time = time.time()
    
    full_image_path = os.path.join(PLOTS_DIR, image_path)
    full_prompt_path = os.path.join(PROMPTS_DIR, prompt_file)
    
    console_output = []
    success = True
    version_dir = None
    timings = {}
    step_status = {
        'extraction': 'pending',
        'interpolation': 'skipped',
        'pointwise': 'skipped'
    }
    
    # Get CSV info
    csv_info = check_csv_exists(image_path, prompt_file)
    original_csv = os.path.join(PLOTS_DIR, csv_info['original']['path'])
    
    # Get prompt short name for extracted file
    prompt_name = os.path.splitext(prompt_file)[0].replace('prompt_', 'p')
    image_dir = os.path.dirname(full_image_path)
    image_name = os.path.basename(image_path)
    
    # Step 1: Run extraction
    update_task(progress='Running extraction...')
    console_output.append("=" * 60)
    console_output.append("STEP 1: Running Plot Extraction")
    console_output.append("=" * 60)
    console_output.append(f"Image: {image_path}")
    console_output.append(f"Prompt: {prompt_file}")
    console_output.append("")
    
    step1_start = time.time()
    try:
        result = subprocess.run(
            ['python', 'plotExtract.py', full_image_path, full_prompt_path],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.stdout:
            console_output.append(result.stdout)
            # Parse VERSION_DIR from output
            for line in result.stdout.split('\n'):
                if line.startswith('VERSION_DIR:'):
                    version_dir = line.replace('VERSION_DIR:', '').strip()
                    break
        if result.stderr:
            console_output.append(f"[STDERR] {result.stderr}")
        
        if result.returncode != 0:
            success = False
            step_status['extraction'] = f"failed (exit code {result.returncode})"
            console_output.append(f"\n[ERROR] Extraction failed with exit code {result.returncode}")
        else:
            step_status['extraction'] = 'success'
            console_output.append("\n[SUCCESS] Extraction completed.")
            
    except subprocess.TimeoutExpired:
        success = False
        step_status['extraction'] = 'failed (timeout)'
        console_output.append("[ERROR] Extraction timed out after 5 minutes")
    except Exception as e:
        success = False
        step_status['extraction'] = 'failed (exception)'
        console_output.append(f"[ERROR] {str(e)}")
    
    step1_time = time.time() - step1_start
    timings['extraction'] = step1_time
    console_output.append(f"[TIME] Extraction took {step1_time:.2f} seconds")
    
    # Determine extracted CSV path
    version_num = 1
    if version_dir:
        version_match = re.search(r'\.v(\d+)$', os.path.basename(version_dir))
        if version_match:
            version_num = int(version_match.group(1))
        extracted_csv = os.path.join(version_dir, f"{image_name}.{prompt_name}.v{version_num}.mistral.out_data")
    else:
        name_for_folder = image_name.replace('.', '_')
        fallback_dir = os.path.join(image_dir, f"{name_for_folder}.{prompt_name}.v{version_num}")
        extracted_csv = os.path.join(fallback_dir, f"{image_name}.{prompt_name}.v{version_num}.mistral.out_data")
    
    # Step 2: Run interpolation if requested
    if run_interpolation and success:
        update_task(progress='Running interpolation...')
        console_output.append("")
        console_output.append("=" * 60)
        console_output.append("STEP 2: Running Interpolation")
        console_output.append("=" * 60)
        
        if not os.path.exists(original_csv):
            success = False
            step_status['interpolation'] = 'failed (missing original CSV)'
            console_output.append(f"[WARNING] Original CSV not found: {original_csv}")
            console_output.append("[ERROR] Interpolation skipped - missing original CSV")
        elif not os.path.exists(extracted_csv):
            success = False
            step_status['interpolation'] = 'failed (missing extracted data)'
            console_output.append(f"[WARNING] Extracted data not found: {extracted_csv}")
            console_output.append("[ERROR] Interpolation skipped - missing extracted data")
        else:
            console_output.append(f"Original: {original_csv}")
            console_output.append(f"Extracted: {extracted_csv}")
            console_output.append(f"Output dir: {version_dir}")
            console_output.append(f"Axis range: X=[{left_x}, {right_x}], Y=[{bottom_y}, {top_y}]")
            console_output.append("")
            
            step2_start = time.time()
            try:
                cmd = ['python', 'interpolation.py', original_csv, extracted_csv,
                       left_x, right_x, bottom_y, top_y]
                if version_dir:
                    cmd.append(version_dir)
                
                result = subprocess.run(
                    cmd,
                    cwd=BASE_DIR,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.stdout:
                    console_output.append(result.stdout)
                if result.stderr:
                    console_output.append(f"[STDERR] {result.stderr}")
                
                if result.returncode != 0:
                    success = False
                    step_status['interpolation'] = f"failed (exit code {result.returncode})"
                    console_output.append(f"\n[ERROR] Interpolation failed with exit code {result.returncode}")
                else:
                    step_status['interpolation'] = 'success'
                    console_output.append("\n[SUCCESS] Interpolation completed.")
                    
            except subprocess.TimeoutExpired:
                success = False
                step_status['interpolation'] = 'failed (timeout)'
                console_output.append("[ERROR] Interpolation timed out after 5 minutes")
            except Exception as e:
                success = False
                step_status['interpolation'] = 'failed (exception)'
                console_output.append(f"[ERROR] {str(e)}")
            
            step2_time = time.time() - step2_start
            timings['interpolation'] = step2_time
            console_output.append(f"[TIME] Interpolation took {step2_time:.2f} seconds")
    elif run_interpolation:
        step_status['interpolation'] = 'skipped (earlier step failed)'
    
    # Step 3: Run pointwise if requested
    if run_pointwise and success:
        update_task(progress='Running pointwise comparison...')
        console_output.append("")
        console_output.append("=" * 60)
        console_output.append("STEP 3: Running Pointwise Comparison")
        console_output.append("=" * 60)
        
        if not os.path.exists(original_csv):
            success = False
            step_status['pointwise'] = 'failed (missing original CSV)'
            console_output.append(f"[WARNING] Original CSV not found: {original_csv}")
            console_output.append("[ERROR] Pointwise skipped - missing original CSV")
        elif not os.path.exists(extracted_csv):
            success = False
            step_status['pointwise'] = 'failed (missing extracted data)'
            console_output.append(f"[WARNING] Extracted data not found: {extracted_csv}")
            console_output.append("[ERROR] Pointwise skipped - missing extracted data")
        else:
            console_output.append(f"Extracted: {extracted_csv}")
            console_output.append(f"Original: {original_csv}")
            console_output.append(f"Output dir: {version_dir}")
            console_output.append(f"Axis range: X=[{left_x}, {right_x}], Y=[{bottom_y}, {top_y}]")
            console_output.append("")
            
            step3_start = time.time()
            try:
                cmd = ['python', 'pointwise.py', extracted_csv, original_csv,
                       left_x, right_x, bottom_y, top_y]
                if version_dir:
                    cmd.append(version_dir)
                
                result = subprocess.run(
                    cmd,
                    cwd=BASE_DIR,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.stdout:
                    console_output.append(result.stdout)
                if result.stderr:
                    console_output.append(f"[STDERR] {result.stderr}")
                
                if result.returncode != 0:
                    success = False
                    step_status['pointwise'] = f"failed (exit code {result.returncode})"
                    console_output.append(f"\n[ERROR] Pointwise comparison failed with exit code {result.returncode}")
                else:
                    step_status['pointwise'] = 'success'
                    console_output.append("\n[SUCCESS] Pointwise comparison completed.")
                    
            except subprocess.TimeoutExpired:
                success = False
                step_status['pointwise'] = 'failed (timeout)'
                console_output.append("[ERROR] Pointwise comparison timed out after 5 minutes")
            except Exception as e:
                success = False
                step_status['pointwise'] = 'failed (exception)'
                console_output.append(f"[ERROR] {str(e)}")
            
            step3_time = time.time() - step3_start
            timings['pointwise'] = step3_time
            console_output.append(f"[TIME] Pointwise took {step3_time:.2f} seconds")
    elif run_pointwise:
        step_status['pointwise'] = 'skipped (earlier step failed)'
    
    total_time = time.time() - total_start_time
    timings['total'] = total_time
    
    console_output.append("")
    console_output.append("=" * 60)
    if success:
        console_output.append("PIPELINE FINISHED SUCCESSFULLY")
    else:
        console_output.append("PIPELINE FINISHED WITH ERRORS")
    console_output.append(f"Extraction: {step_status['extraction']}")
    console_output.append(f"Interpolation: {step_status['interpolation']}")
    console_output.append(f"Pointwise: {step_status['pointwise']}")
    console_output.append(f"Total time: {total_time:.2f} seconds")
    console_output.append("=" * 60)
    
    # Get updated outputs
    outputs = get_output_files(image_path, prompt_file, version_dir)
    csv_status = check_csv_exists(image_path, prompt_file)
    
    # Build final result
    final_result = {
        'success': success,
        'console': '\n'.join(console_output),
        'outputs': outputs,
        'csv_status': csv_status,
        'version_dir': version_dir,
        'timings': timings,
        'completed_at': time.time(),
        'image_path': image_path,
        'prompt_file': prompt_file
    }
    
    # Update task state
    with extraction_tasks_lock:
        if task_id in extraction_tasks:
            extraction_tasks[task_id]['status'] = 'completed'
            extraction_tasks[task_id]['result'] = final_result
    
    # Save to file for persistence
    save_extraction_state(final_result)

@app.route('/task_status/<task_id>')
def task_status(task_id):
    """Get the status of a running task."""
    with extraction_tasks_lock:
        if task_id in extraction_tasks:
            task = extraction_tasks[task_id]
            return jsonify({
                'status': task['status'],
                'progress': task.get('progress', ''),
                'elapsed': time.time() - task['started_at'],
                'result': task.get('result')
            })
    return jsonify({'status': 'not_found'})

@app.route('/last_extraction_result')
def last_extraction_result():
    """Get the last completed extraction result (persisted to file)."""
    state = load_extraction_state()
    if state:
        return jsonify({'exists': True, 'result': state})
    return jsonify({'exists': False})

# =============================================================================
# Batch Extraction Routes
# =============================================================================

# Directory for batch uploads
BATCH_DIR = os.path.join(PLOTS_DIR, 'batch_uploads')
os.makedirs(BATCH_DIR, exist_ok=True)

@app.route('/run_batch_single', methods=['POST'])
def run_batch_single():
    """
    Process a single image for batch extraction.
    Accepts file upload, saves to batch_uploads folder, runs extraction, returns results.
    """
    import re
    from datetime import datetime
    
    try:
        # Get the uploaded file
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Get parameters
        prompt_file = request.form.get('prompt', 'prompt_1.py')
        run_interpolation = request.form.get('runInterpolation', 'false') == 'true'
        run_pointwise = request.form.get('runPointwise', 'false') == 'true'
        left_x = request.form.get('leftX', '0')
        right_x = request.form.get('rightX', '100')
        bottom_y = request.form.get('bottomY', '0')
        top_y = request.form.get('topY', '100')
        
        # Create a unique batch subfolder for this image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_subfolder = f"batch_{timestamp}_{file.filename.replace('.', '_')}"
        batch_image_dir = os.path.join(BATCH_DIR, batch_subfolder)
        os.makedirs(batch_image_dir, exist_ok=True)
        
        # Save the uploaded file
        original_filename = file.filename
        image_path = os.path.join(batch_image_dir, original_filename)
        file.save(image_path)
        
        # Get relative path for the extraction pipeline
        rel_image_path = os.path.relpath(image_path, PLOTS_DIR).replace('\\', '/')
        
        # Build the full paths
        full_prompt_path = os.path.join(PROMPTS_DIR, prompt_file)
        
        console_output = []
        success = True
        version_dir = None
        timings = {}
        total_start_time = time.time()
        
        # Get prompt short name
        prompt_name = os.path.splitext(prompt_file)[0].replace('prompt_', 'p')
        
        # Step 1: Run extraction
        console_output.append("=" * 60)
        console_output.append("STEP 1: Running Plot Extraction")
        console_output.append("=" * 60)
        console_output.append(f"Image: {original_filename}")
        console_output.append(f"Prompt: {prompt_file}")
        console_output.append("")
        
        step1_start = time.time()
        try:
            result = subprocess.run(
                ['python', 'plotExtract.py', image_path, full_prompt_path],
                cwd=BASE_DIR,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.stdout:
                console_output.append(result.stdout)
                # Parse VERSION_DIR from output
                for line in result.stdout.split('\n'):
                    if line.startswith('VERSION_DIR:'):
                        version_dir = line.replace('VERSION_DIR:', '').strip()
                        break
            if result.stderr:
                console_output.append(f"[STDERR] {result.stderr}")
            
            if result.returncode != 0:
                success = False
                console_output.append(f"\n[ERROR] Extraction failed with exit code {result.returncode}")
            else:
                console_output.append("\n[SUCCESS] Extraction completed!")
                
        except subprocess.TimeoutExpired:
            success = False
            console_output.append("[ERROR] Extraction timed out after 5 minutes")
        except Exception as e:
            success = False
            console_output.append(f"[ERROR] {str(e)}")
        
        step1_time = time.time() - step1_start
        timings['extraction'] = step1_time
        console_output.append(f"[TIME] Extraction took {step1_time:.2f} seconds")
        
        # Determine extracted CSV path
        version_num = 1
        if version_dir:
            version_match = re.search(r'\.v(\d+)$', os.path.basename(version_dir))
            if version_match:
                version_num = int(version_match.group(1))
            extracted_csv = os.path.join(version_dir, f"{original_filename}.{prompt_name}.v{version_num}.mistral.out_data")
        else:
            name_for_folder = original_filename.replace('.', '_')
            fallback_dir = os.path.join(batch_image_dir, f"{name_for_folder}.{prompt_name}.v{version_num}")
            extracted_csv = os.path.join(fallback_dir, f"{original_filename}.{prompt_name}.v{version_num}.mistral.out_data")
        
        # Try to find original CSV in batch_image_dir
        original_csv = None
        for f in os.listdir(batch_image_dir):
            if f.endswith('-original.csv'):
                original_csv = os.path.join(batch_image_dir, f)
                break
        
        # Step 2: Run interpolation if requested
        if run_interpolation and success and original_csv and os.path.exists(original_csv):
            console_output.append("")
            console_output.append("=" * 60)
            console_output.append("STEP 2: Running Interpolation")
            console_output.append("=" * 60)
            
            if not os.path.exists(extracted_csv):
                console_output.append(f"[WARNING] Extracted data not found: {extracted_csv}")
                console_output.append("[SKIPPED] Interpolation skipped - missing extracted data")
            else:
                console_output.append(f"Original: {original_csv}")
                console_output.append(f"Extracted: {extracted_csv}")
                console_output.append(f"Output dir: {version_dir}")
                console_output.append("")
                
                step2_start = time.time()
                try:
                    cmd = ['python', 'interpolation.py', original_csv, extracted_csv,
                           left_x, right_x, bottom_y, top_y]
                    if version_dir:
                        cmd.append(version_dir)
                    
                    result = subprocess.run(
                        cmd,
                        cwd=BASE_DIR,
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    
                    if result.stdout:
                        console_output.append(result.stdout)
                    if result.stderr:
                        console_output.append(f"[STDERR] {result.stderr}")
                    
                    if result.returncode != 0:
                        console_output.append(f"\n[ERROR] Interpolation failed with exit code {result.returncode}")
                    else:
                        console_output.append("\n[SUCCESS] Interpolation completed!")
                        
                except subprocess.TimeoutExpired:
                    console_output.append("[ERROR] Interpolation timed out after 5 minutes")
                except Exception as e:
                    console_output.append(f"[ERROR] {str(e)}")
                
                step2_time = time.time() - step2_start
                timings['interpolation'] = step2_time
                console_output.append(f"[TIME] Interpolation took {step2_time:.2f} seconds")
        elif run_interpolation and success:
            console_output.append("")
            console_output.append("[INFO] Interpolation skipped - no original CSV found for comparison")
        
        # Step 3: Run pointwise if requested
        if run_pointwise and success and original_csv and os.path.exists(original_csv):
            console_output.append("")
            console_output.append("=" * 60)
            console_output.append("STEP 3: Running Pointwise Comparison")
            console_output.append("=" * 60)
            
            if not os.path.exists(extracted_csv):
                console_output.append(f"[WARNING] Extracted data not found: {extracted_csv}")
                console_output.append("[SKIPPED] Pointwise skipped - missing extracted data")
            else:
                console_output.append(f"Extracted: {extracted_csv}")
                console_output.append(f"Original: {original_csv}")
                console_output.append(f"Output dir: {version_dir}")
                console_output.append("")
                
                step3_start = time.time()
                try:
                    cmd = ['python', 'pointwise.py', extracted_csv, original_csv,
                           left_x, right_x, bottom_y, top_y]
                    if version_dir:
                        cmd.append(version_dir)
                    
                    result = subprocess.run(
                        cmd,
                        cwd=BASE_DIR,
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    
                    if result.stdout:
                        console_output.append(result.stdout)
                    if result.stderr:
                        console_output.append(f"[STDERR] {result.stderr}")
                    
                    if result.returncode != 0:
                        console_output.append(f"\n[ERROR] Pointwise comparison failed with exit code {result.returncode}")
                    else:
                        console_output.append("\n[SUCCESS] Pointwise comparison completed!")
                        
                except subprocess.TimeoutExpired:
                    console_output.append("[ERROR] Pointwise comparison timed out after 5 minutes")
                except Exception as e:
                    console_output.append(f"[ERROR] {str(e)}")
                
                step3_time = time.time() - step3_start
                timings['pointwise'] = step3_time
                console_output.append(f"[TIME] Pointwise took {step3_time:.2f} seconds")
        elif run_pointwise and success:
            console_output.append("")
            console_output.append("[INFO] Pointwise comparison skipped - no original CSV found for comparison")
        
        total_time = time.time() - total_start_time
        timings['total'] = total_time
        
        # Get outputs relative to PLOTS_DIR for serving
        outputs = {'images': [], 'stats': [], 'data': [], 'summary': {}}
        
        if version_dir and os.path.exists(version_dir):
            # Get the relative path from PLOTS_DIR
            try:
                rel_version_dir = os.path.relpath(version_dir, PLOTS_DIR)
            except ValueError:
                rel_version_dir = None
            
            if rel_version_dir:
                # Build outputs from version_dir
                for f in os.listdir(version_dir):
                    file_path = os.path.join(rel_version_dir, f).replace('\\', '/')
                    
                    if f.endswith('.png'):
                        label = f
                        if f.startswith('comparison_'):
                            label = 'Comparison'
                        elif f.startswith('interpolated_'):
                            label = 'Interpolation'
                        elif f.startswith('pointwise_'):
                            label = 'Pointwise'
                        outputs['images'].append({
                            'filename': f,
                            'path': file_path,
                            'label': label
                        })
                    elif 'stats' in f.lower() or f.endswith('_stats'):
                        label = 'Statistics'
                        if 'interpolation' in f.lower():
                            label = 'Interpolation Stats'
                        elif 'pointwise' in f.lower():
                            label = 'Pointwise Stats'
                        outputs['stats'].append({
                            'filename': f,
                            'path': file_path,
                            'label': label
                        })
                    elif f.endswith('.out_data') or f.endswith('.csv'):
                        outputs['data'].append({
                            'filename': f,
                            'path': file_path,
                            'label': 'Extracted Data' if 'out_data' in f else 'Data'
                        })
                
                # Parse summary stats
                outputs['summary'] = _parse_summary_stats(version_dir)
        
        return jsonify({
            'success': success,
            'console': '\n'.join(console_output),
            'outputs': outputs,
            'version_dir': version_dir,
            'timings': timings,
            'filename': original_filename
        })
            
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })

# =============================================================================
# Synthetic Generator Routes
# =============================================================================

@app.route('/synthetic')
def synthetic():
    """Render the synthetic generator page."""
    settings = load_synthetic_settings()
    if not settings['curves']:
        settings['curves'] = get_default_curves(settings['num_curves'])
    return render_template('synthetic.html', settings=settings)

@app.route('/synthetic/editor')
def synthetic_editor():
    """Render the synthetic plot editor page."""
    return render_template('synthetic_editor.html')

@app.route('/synthetic/get_settings')
def synthetic_get_settings():
    """Return current synthetic settings as JSON."""
    settings = load_synthetic_settings()
    if not settings['curves']:
        settings['curves'] = get_default_curves(settings['num_curves'])
    return jsonify(settings)

@app.route('/synthetic/update_curves', methods=['POST'])
def synthetic_update_curves():
    """Update the number of curves and return new curve configs."""
    data = request.json
    num_curves = int(data.get('num_curves', 3))
    
    settings = load_synthetic_settings()
    current_curves = settings.get('curves', [])
    
    if num_curves > len(current_curves):
        for i in range(len(current_curves), num_curves):
            curve = DEFAULT_CURVE.copy()
            curve['name'] = f'Condition {i + 1}'
            curve['color'] = COLOR_PALETTE[i % len(COLOR_PALETTE)]
            trends = ['stable', 'down', 'up', 'kill_regrowth', 'mixed']
            curve['trend'] = trends[i % len(trends)]
            current_curves.append(curve)
    elif num_curves < len(current_curves):
        current_curves = current_curves[:num_curves]
    
    settings['curves'] = current_curves
    settings['num_curves'] = num_curves
    save_synthetic_settings(settings)
    
    return jsonify({'curves': current_curves})

@app.route('/synthetic/preview', methods=['POST'])
def synthetic_preview():
    """Generate a preview of the synthetic plot."""
    start_time = time.time()
    settings = request.json
    
    # Ensure all new settings have default values if missing
    settings.setdefault('x_tick_mode', 'custom')
    settings.setdefault('x_tick_interval', 2)
    settings.setdefault('axis_break_enabled', False)
    settings.setdefault('axis_break_type', 'x')
    settings.setdefault('axis_break_start', '')
    settings.setdefault('axis_break_end', '')
    
    x_values, curves_data = generate_all_curves(settings)
    fig = create_synthetic_plot(settings, curves_data, x_values)
    img_base64 = fig_to_base64(fig)
    save_synthetic_settings(settings)
    elapsed = time.time() - start_time
    
    return jsonify({
        'success': True,
        'image': img_base64,
        'x_values': x_values,
        'curves_data': curves_data,
        'time_seconds': round(elapsed, 2)
    })

@app.route('/synthetic/save', methods=['POST'])
def synthetic_save():
    """Save the synthetic plot and data to files."""
    start_time = time.time()
    settings = request.json
    
    # Ensure all new settings have default values if missing
    settings.setdefault('x_tick_mode', 'custom')
    settings.setdefault('x_tick_interval', 2)
    settings.setdefault('axis_break_enabled', False)
    settings.setdefault('axis_break_type', 'x')
    settings.setdefault('axis_break_start', '')
    settings.setdefault('axis_break_end', '')
    
    x_values, curves_data = generate_all_curves(settings)
    saved_files = save_synthetic_plot_and_data(settings, curves_data, x_values)
    save_synthetic_settings(settings)
    elapsed = time.time() - start_time
    
    return jsonify({
        'success': True,
        'files': saved_files,
        'message': f"Saved to {saved_files['filename']}",
        'time_seconds': round(elapsed, 2)
    })

@app.route('/synthetic/reset', methods=['POST'])
def synthetic_reset():
    """Reset all synthetic settings to defaults."""
    settings = DEFAULT_SETTINGS.copy()
    settings['curves'] = get_default_curves(settings['num_curves'])
    save_synthetic_settings(settings)
    return jsonify(settings)

@app.route('/synthetic/regenerate', methods=['POST'])
def synthetic_regenerate():
    """Regenerate curve data with same settings (new random values)."""
    start_time = time.time()
    settings = request.json
    
    # Ensure all new settings have default values if missing
    settings.setdefault('x_tick_mode', 'custom')
    settings.setdefault('x_tick_interval', 2)
    settings.setdefault('axis_break_enabled', False)
    settings.setdefault('axis_break_type', 'x')
    settings.setdefault('axis_break_start', '')
    settings.setdefault('axis_break_end', '')
    
    x_values, curves_data = generate_all_curves(settings)
    fig = create_synthetic_plot(settings, curves_data, x_values)
    img_base64 = fig_to_base64(fig)
    elapsed = time.time() - start_time
    
    return jsonify({
        'success': True,
        'image': img_base64,
        'x_values': x_values,
        'curves_data': curves_data,
        'time_seconds': round(elapsed, 2)
    })

# =============================================================================
# Plot Editor Routes
# =============================================================================

@app.route('/synthetic/get_existing_plots')
def get_existing_plots():
    """Get list of existing synthetic plots that can be edited."""
    plots = []
    
    if os.path.exists(SYNTHETIC_DIR):
        for item in sorted(os.listdir(SYNTHETIC_DIR)):
            item_path = os.path.join(SYNTHETIC_DIR, item)
            if os.path.isdir(item_path):
                png_file = os.path.join(item_path, f'{item}.png')
                csv_file = os.path.join(item_path, f'{item}-original.csv')
                
                if not os.path.exists(png_file):
                    for f in os.listdir(item_path):
                        if f.endswith('.png'):
                            png_file = os.path.join(item_path, f)
                            break
                
                if os.path.exists(png_file):
                    plots.append({
                        'name': item,
                        'folder': item_path,
                        'has_csv': os.path.exists(csv_file)
                    })
    
    return jsonify(plots)

@app.route('/synthetic/load_plot_for_edit/<plot_name>')
def load_plot_for_edit(plot_name):
    """Load an existing plot's data and settings for editing."""
    plot_folder = os.path.join(SYNTHETIC_DIR, plot_name)
    
    if not os.path.exists(plot_folder):
        return jsonify({'success': False, 'error': 'Plot folder not found'})
    
    csv_file = os.path.join(plot_folder, f'{plot_name}-original.csv')
    if not os.path.exists(csv_file):
        for f in os.listdir(plot_folder):
            if f.endswith('-original.csv'):
                csv_file = os.path.join(plot_folder, f)
                break
    
    if not os.path.exists(csv_file):
        return jsonify({'success': False, 'error': 'CSV file not found'})
    
    curves_data = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        header = lines[0].strip().split(',')
        num_curves = len(header) // 2
        
        curve_names = []
        for i in range(num_curves):
            y_col_name = header[i * 2 + 1].strip()
            curve_names.append(y_col_name)
        
        x_values = []
        y_values_per_curve = [[] for _ in range(num_curves)]
        
        for line in lines[1:]:
            if not line.strip():
                continue
            values = line.strip().split(',')
            for i in range(num_curves):
                x_val = float(values[i * 2])
                y_val = float(values[i * 2 + 1])
                if i == 0:
                    x_values.append(x_val)
                y_values_per_curve[i].append(y_val)
        
        for i in range(num_curves):
            curves_data.append({
                'x': x_values,
                'y': y_values_per_curve[i],
                'config': {
                    'name': curve_names[i],
                    'color': COLOR_PALETTE[i % len(COLOR_PALETTE)],
                    'marker': 'o',
                    'line_style': '-',
                    'show_line': True,
                    'line_width': 1.5,
                    'marker_size': 6,
                    'noise_level': 0.1
                }
            })
        
        png_file = os.path.join(plot_folder, f'{plot_name}.png')
        if not os.path.exists(png_file):
            for f in os.listdir(plot_folder):
                if f.endswith('.png') and not '_copy' in f:
                    png_file = os.path.join(plot_folder, f)
                    break
        
        img_base64 = None
        if os.path.exists(png_file):
            with open(png_file, 'rb') as f:
                img_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'plot_name': plot_name,
            'x_values': x_values,
            'curves_data': curves_data,
            'image': img_base64
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/synthetic/preview_edit', methods=['POST'])
def preview_edit():
    """Generate a preview of the edited plot with visual changes only."""
    try:
        start_time = time.time()
        data = request.json
        curves_data = data['curves_data']
        settings = data['settings']
        x_values = data['x_values']
        
        settings.setdefault('x_min', '')
        settings.setdefault('x_max', '')
        settings.setdefault('x_label', 'Time')
        settings.setdefault('x_unit', 'hours')
        settings.setdefault('y_label', 'Bacterial Count')
        settings.setdefault('y_unit', 'CFU/mL')
        settings.setdefault('y_scale', 'log')
        settings.setdefault('y_min', '0')
        settings.setdefault('y_max', '')
        settings.setdefault('title', '')
        settings.setdefault('figure_width', 10)
        settings.setdefault('figure_height', 6)
        settings.setdefault('show_legend', True)
        settings.setdefault('show_grid', True)
        
        fig = create_synthetic_plot(settings, curves_data, x_values)
        img_base64 = fig_to_base64(fig)
        elapsed = time.time() - start_time
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'time_seconds': round(elapsed, 2)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/synthetic/save_edit', methods=['POST'])
def save_edit():
    """Save the edited plot as a copy."""
    try:
        start_time = time.time()
        data = request.json
        original_name = data['original_name']
        curves_data = data['curves_data']
        settings = data['settings']
        x_values = data['x_values']
        
        settings.setdefault('x_min', '')
        settings.setdefault('x_max', '')
        settings.setdefault('x_label', 'Time')
        settings.setdefault('x_unit', 'hours')
        settings.setdefault('y_label', 'Bacterial Count')
        settings.setdefault('y_unit', 'CFU/mL')
        settings.setdefault('y_scale', 'log')
        settings.setdefault('y_min', '0')
        settings.setdefault('y_max', '')
        settings.setdefault('title', '')
        settings.setdefault('figure_width', 10)
        settings.setdefault('figure_height', 6)
        settings.setdefault('show_legend', True)
        settings.setdefault('show_grid', True)
        
        plot_folder = os.path.join(SYNTHETIC_DIR, original_name)
        
        if not os.path.exists(plot_folder):
            return jsonify({'success': False, 'error': 'Plot folder not found'})
        
        copy_num = 1
        while True:
            copy_name = f'{original_name}_copy{copy_num}'
            png_path = os.path.join(plot_folder, f'{copy_name}.png')
            if not os.path.exists(png_path):
                break
            copy_num += 1
        
        fig = create_synthetic_plot(settings, curves_data, x_values)
        
        png_path = os.path.join(plot_folder, f'{copy_name}.png')
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        
        svg_path = None
        if settings.get('save_svg', False):
            svg_path = os.path.join(plot_folder, f'{copy_name}.svg')
            fig.savefig(svg_path, format='svg', bbox_inches='tight')
        
        plt.close(fig)
        
        csv_path = os.path.join(plot_folder, f'{copy_name}-original.csv')
        save_synthetic_csv(curves_data, x_values, settings, csv_path)
        elapsed = time.time() - start_time
        
        return jsonify({
            'success': True,
            'files': {
                'png': png_path,
                'svg': svg_path,
                'csv': csv_path,
                'filename': copy_name,
                'folder': plot_folder
            },
            'message': f'Saved as {copy_name}',
            'time_seconds': round(elapsed, 2)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

# =============================================================================
# Results Comparison Page Routes
# =============================================================================

@app.route('/results')
def results_page():
    """Render the results comparison page"""
    return render_template('results.html')

@app.route('/results/list_replots')
def list_replots():
    """List all available replots from synthetic and first_examples folders"""
    source_filter = request.args.get('source', 'all')
    replots = []
    
    def find_replots_in_folder(folder_path, source_name):
        """Recursively find all replot images in a folder"""
        results = []
        if not os.path.exists(folder_path):
            return results
        
        for root, dirs, files in os.walk(folder_path):
            # Look for replot images
            for f in files:
                if '-replot' in f and f.endswith('.png'):
                    rel_path = os.path.relpath(os.path.join(root, f), PLOTS_DIR)
                    
                    # Extract display name from path
                    path_parts = rel_path.replace('\\', '/').split('/')
                    if len(path_parts) >= 2:
                        display_name = '/'.join(path_parts[-2:])  # Folder/filename
                    else:
                        display_name = f
                    
                    results.append({
                        'path': rel_path.replace('\\', '/'),
                        'display_name': display_name,
                        'source': source_name,
                        'folder': os.path.dirname(rel_path).replace('\\', '/')
                    })
        return results
    
    # Scan synthetic folder
    if source_filter in ['all', 'synthetic']:
        replots.extend(find_replots_in_folder(SYNTHETIC_DIR, 'synthetic'))
    
    # Scan first_examples folder
    if source_filter in ['all', 'first_examples']:
        first_examples_dir = os.path.join(PLOTS_DIR, 'first_examples')
        replots.extend(find_replots_in_folder(first_examples_dir, 'first_examples'))
    
    # Sort by display name
    replots.sort(key=lambda x: x['display_name'])
    
    return jsonify({'replots': replots})

@app.route('/results/get_replot_data', methods=['POST'])
def get_replot_data():
    """Get all data for a specific replot including images and stats"""
    data = request.json
    replot_path = data.get('replot_path', '')
    
    if not replot_path:
        return jsonify({'success': False, 'error': 'No replot path provided'})
    
    full_path = os.path.join(PLOTS_DIR, replot_path.replace('/', os.sep))
    if not os.path.exists(full_path):
        return jsonify({'success': False, 'error': 'Replot file not found'})
    
    folder = os.path.dirname(full_path)
    replot_filename = os.path.basename(full_path)
    
    result = {
        'success': True,
        'replot_name': replot_filename,
        'replot_image': f'/plots/{replot_path}',
        'pointwise_image': None,
        'pointwise_stats': None,
        'interpolation_image': None,
        'interpolation_stats': None,
        'visual_image': None
    }
    
    # Find associated comparison files
    folder_files = os.listdir(folder)
    
    for f in folder_files:
        full_file_path = os.path.join(folder, f)
        rel_file_path = os.path.relpath(full_file_path, PLOTS_DIR).replace('\\', '/')
        
        # Pointwise comparison image
        if f.startswith('pointwise_') and f.endswith('.png'):
            result['pointwise_image'] = f'/plots/{rel_file_path}'
        
        # Pointwise stats
        if f.startswith('pointwise_') and f.endswith('.stats'):
            result['pointwise_stats'] = parse_pointwise_stats(full_file_path)
        
        # Interpolation comparison image
        if f.startswith('interpolated_') and f.endswith('.png'):
            result['interpolation_image'] = f'/plots/{rel_file_path}'
        
        # Interpolation stats
        if f.startswith('interpolated_') and f.endswith('.stats'):
            result['interpolation_stats'] = parse_interpolation_stats(full_file_path)
        
        # Visual/side-by-side comparison
        if f.startswith('comparison_') and f.endswith('.png'):
            result['visual_image'] = f'/plots/{rel_file_path}'
    
    return jsonify(result)

def parse_pointwise_stats(stats_path):
    """Parse a pointwise stats file and return structured data"""
    try:
        with open(stats_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        curves = []
        current_curve = None
        overall = {}
        
        for line in content.split('\n'):
            line = line.strip()
            
            # Match curve header: "Curve 'name1' -> 'name2':"
            if line.startswith("Curve '"):
                if current_curve:
                    curves.append(current_curve)
                
                # Extract names
                parts = line.split("' -> '")
                if len(parts) == 2:
                    extracted_name = parts[0].replace("Curve '", "")
                    original_name = parts[1].rstrip("':")
                else:
                    extracted_name = line
                    original_name = ""
                
                current_curve = {
                    'extracted_name': extracted_name,
                    'original_name': original_name
                }
            
            # Parse stats
            elif current_curve and ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                try:
                    if key == 'MAE X (percent)':
                        current_curve['mae_x_percent'] = float(value)
                    elif key == 'MAE Y (percent)':
                        current_curve['mae_y_percent'] = float(value)
                    elif key == 'Precision':
                        current_curve['precision'] = float(value)
                    elif key == 'Recall':
                        current_curve['recall'] = float(value)
                    elif key == 'MatchedPairs':
                        current_curve['matched_pairs'] = int(value)
                except ValueError:
                    pass
            
            # Overall stats
            elif line.startswith('Overall') or 'Average' in line or 'Mean' in line:
                if ':' in line:
                    key, value = line.split(':', 1)
                    try:
                        overall[key.strip().lower().replace(' ', '_')] = float(value.strip().rstrip('%'))
                    except ValueError:
                        pass
        
        # Add last curve
        if current_curve:
            curves.append(current_curve)
        
        # Calculate overall if not present
        if not overall and curves:
            mae_y_values = [c['mae_y_percent'] for c in curves if 'mae_y_percent' in c]
            precision_values = [c['precision'] for c in curves if 'precision' in c]
            recall_values = [c['recall'] for c in curves if 'recall' in c]
            
            overall = {
                'avg_mae_y': sum(mae_y_values) / len(mae_y_values) if mae_y_values else None,
                'avg_precision': sum(precision_values) / len(precision_values) if precision_values else None,
                'avg_recall': sum(recall_values) / len(recall_values) if recall_values else None
            }
        
        return {'curves': curves, 'overall': overall}
    
    except Exception as e:
        print(f"Error parsing pointwise stats: {e}")
        return None

def parse_interpolation_stats(stats_path):
    """Parse an interpolation stats file and return structured data"""
    try:
        with open(stats_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        curves = []
        current_curve = None
        mean_mae = None
        
        for line in content.split('\n'):
            line = line.strip()
            
            # Match curve header
            if line.startswith("Curve '"):
                if current_curve:
                    curves.append(current_curve)
                
                # Extract names
                parts = line.split("' -> '")
                if len(parts) == 2:
                    extracted_name = parts[0].replace("Curve '", "")
                    original_name = parts[1].rstrip("':")
                else:
                    extracted_name = line
                    original_name = ""
                
                current_curve = {
                    'extracted_name': extracted_name,
                    'original_name': original_name
                }
            
            # Parse stats
            elif current_curve and ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                try:
                    if key == 'MAE':
                        current_curve['mae'] = float(value)
                    elif key == 'LeftMissed':
                        current_curve['left_missed'] = float(value)
                    elif key == 'RightMissed':
                        current_curve['right_missed'] = float(value)
                except ValueError:
                    pass
            
            # Mean MAE at the end
            elif 'Mean MAE' in line and ':' in line:
                try:
                    mean_mae = float(line.split(':')[1].strip())
                except ValueError:
                    pass
        
        # Add last curve
        if current_curve:
            curves.append(current_curve)
        
        # Calculate mean MAE if not present
        if mean_mae is None and curves:
            mae_values = [c['mae'] for c in curves if 'mae' in c]
            if mae_values:
                mean_mae = sum(mae_values) / len(mae_values)
        
        return {'curves': curves, 'mean_mae': mean_mae}
    
    except Exception as e:
        print(f"Error parsing interpolation stats: {e}")
        return None

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    print(f"Synthetic plots will be saved to: {SYNTHETIC_DIR}")
    print("Starting PlotExtract Web Application...")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True, port=5000)
