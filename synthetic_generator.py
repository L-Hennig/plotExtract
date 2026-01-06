"""
Synthetic Time-Kill Plot Generator
===================================
A Flask-based web application for generating synthetic time-kill plots
with customizable parameters for each curve.

Author: PlotExtract Project
Date: December 2025
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
import io
import base64

# =============================================================================
# Flask Application Setup
# =============================================================================

app = Flask(__name__)

# Base directory and output folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
SYNTHETIC_DIR = os.path.join(PLOTS_DIR, 'synthetic')

# Create synthetic folder if it doesn't exist
os.makedirs(SYNTHETIC_DIR, exist_ok=True)

# Settings file to persist user preferences
SETTINGS_FILE = os.path.join(BASE_DIR, 'synthetic_settings.json')

# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_SETTINGS = {
    # Global plot settings
    'num_curves': 3,
    'num_points': 8,
    'x_values_mode': 'auto',  # 'auto' or 'manual'
    'x_values_manual': '0, 2, 4, 6, 8, 12, 18, 24',
    'x_spacing': 3,  # For auto mode
    
    # Axis settings
    'x_label': 'Time',
    'x_unit': 'hours',
    'y_label': 'Bacterial Count',
    'y_unit': 'CFU/mL',
    'y_scale': 'log',  # 'log' or 'linear'
    'x_min': '',
    'x_max': '',
    'y_min': '',
    'y_max': '',
    'x_tick_interval': 2,  # Tick marks every N hours (0 or empty for auto)
    'x_tick_mode': 'custom',  # 'custom' or 'auto'
    
    # Axis break settings
    'axis_break_enabled': False,
    'axis_break_type': 'x',  # 'x' or 'y'
    'axis_break_start': '',
    'axis_break_end': '',
    
    # Plot appearance
    'title': '',
    'show_legend': True,
    'show_grid': True,
    'figure_width': 10,
    'figure_height': 6,
    
    # Output settings
    'save_svg': False,
    
    # Per-curve defaults (will be expanded when curves are added)
    'curves': []
}

DEFAULT_CURVE = {
    'name': 'Condition',
    'initial_y': 6.0,  # Log10 value for log scale, or actual value for linear
    'trend': 'stable',  # 'up', 'down', 'stable', 'mixed', 'kill_regrowth'
    'trend_magnitude': 1.0,  # How strong the trend is (0-3 scale)
    'noise_level': 0.1,  # Standard deviation of noise
    'color': '#1f77b4',
    'marker': 'o',  # 'o', 's', '^', 'D', 'v', 'p', '*'
    'line_style': '-',  # '-', '--', ':', '-.'
    'show_line': True,
    'line_width': 1.5,
    'marker_size': 6
}

# Color palette for curves
COLOR_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
]

# =============================================================================
# Settings Management
# =============================================================================

def load_settings():
    """Load settings from file, or return defaults if file doesn't exist."""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                saved = json.load(f)
                # Merge with defaults to handle new settings
                settings = DEFAULT_SETTINGS.copy()
                settings.update(saved)
                return settings
        except Exception as e:
            print(f"Error loading settings: {e}")
    return DEFAULT_SETTINGS.copy()

def save_settings(settings):
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
        # Vary the default trends
        trends = ['stable', 'down', 'up', 'kill_regrowth', 'mixed']
        curve['trend'] = trends[i % len(trends)]
        curves.append(curve)
    return curves

# =============================================================================
# Data Generation Functions
# =============================================================================

def generate_x_values(settings):
    """Generate X-axis values based on settings."""
    if settings['x_values_mode'] == 'manual':
        # Parse manual X values
        try:
            x_vals = [float(x.strip()) for x in settings['x_values_manual'].split(',')]
            return np.array(x_vals)
        except:
            pass
    
    # Auto-generate evenly spaced values
    num_points = settings['num_points']
    spacing = settings['x_spacing']
    return np.arange(0, num_points * spacing, spacing)

def generate_curve_data(x_values, curve_config, y_scale='log'):
    """
    Generate Y-values for a single curve based on its configuration.
    
    Parameters:
    -----------
    x_values : np.array
        The X-axis values
    curve_config : dict
        Configuration for this curve (trend, noise, etc.)
    y_scale : str
        'log' or 'linear' - affects how values are generated
    
    Returns:
    --------
    np.array : Y-values for the curve
    """
    n_points = len(x_values)
    initial_y = curve_config['initial_y']
    trend = curve_config['trend']
    magnitude = curve_config['trend_magnitude']
    noise = curve_config['noise_level']
    
    # Normalize x to 0-1 range for trend calculation
    x_norm = (x_values - x_values.min()) / (x_values.max() - x_values.min() + 1e-10)
    
    # Generate base trend
    if trend == 'stable':
        # Mostly flat with slight random walk
        base = np.zeros(n_points)
        for i in range(1, n_points):
            base[i] = base[i-1] + np.random.normal(0, 0.05 * magnitude)
    
    elif trend == 'down':
        # Decreasing trend (killing curve)
        # Exponential decay pattern typical of bactericidal action
        decay_rate = 0.5 + magnitude * 0.5
        base = -decay_rate * x_norm * (2 + magnitude)
        # Add some curvature
        base = base * (1 + 0.3 * np.sin(x_norm * np.pi))
    
    elif trend == 'up':
        # Increasing trend (growth curve)
        growth_rate = 0.3 + magnitude * 0.3
        # Logistic-like growth
        base = magnitude * (1 - np.exp(-growth_rate * x_norm * 5)) * 2
    
    elif trend == 'mixed':
        # Variable - goes up then down or vice versa
        peak_pos = np.random.uniform(0.3, 0.7)
        base = np.zeros(n_points)
        for i, xn in enumerate(x_norm):
            if xn < peak_pos:
                base[i] = magnitude * (xn / peak_pos)
            else:
                base[i] = magnitude * (1 - (xn - peak_pos) / (1 - peak_pos))
    
    elif trend == 'kill_regrowth':
        # Classic time-kill pattern: initial drop then regrowth
        # Common in antibiotic time-kill assays
        nadir_pos = np.random.uniform(0.3, 0.5)  # Lowest point position
        nadir_depth = 1.5 + magnitude  # How deep the kill goes
        
        base = np.zeros(n_points)
        for i, xn in enumerate(x_norm):
            if xn < nadir_pos:
                # Kill phase - exponential decay
                base[i] = -nadir_depth * (xn / nadir_pos) ** 0.8
            else:
                # Regrowth phase
                regrowth_x = (xn - nadir_pos) / (1 - nadir_pos)
                regrowth_amount = (magnitude * 0.5) * regrowth_x ** 1.5
                base[i] = -nadir_depth + regrowth_amount
    
    else:
        base = np.zeros(n_points)
    
    # Add noise
    noise_vals = np.random.normal(0, noise, n_points)
    
    # Combine: initial value + trend + noise
    if y_scale == 'log':
        # For log scale, work in log space
        y_values = initial_y + base + noise_vals
        # Ensure values stay positive in log scale
        y_values = np.maximum(y_values, 0.1)
    else:
        # For linear scale
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

# =============================================================================
# Plotting Functions
# =============================================================================

def create_plot(settings, curves_data, x_values):
    """
    Create the matplotlib figure based on settings and data.
    
    Returns:
    --------
    matplotlib.figure.Figure : The generated figure
    """
    fig, ax = plt.subplots(figsize=(settings['figure_width'], settings['figure_height']))
    
    # Plot each curve
    for curve_data in curves_data:
        config = curve_data['config']
        x = curve_data['x']
        y = curve_data['y']
        
        # For log scale, plot the log10 values directly (y is already in log10 units)
        # For linear scale, y contains actual values
        if settings['y_scale'] == 'log':
            y_plot = y  # Keep as log10 values
        else:
            y_plot = y
        
        # Plot line if enabled
        if config['show_line']:
            ax.plot(x, y_plot,
                   linestyle=config['line_style'],
                   color=config['color'],
                   linewidth=config['line_width'],
                   label=config['name'])
        
        # Plot markers (always with no line - line was drawn above if needed)
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
    
    # Set axis labels
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
    
    # Calculate x_max from data if not specified
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
                y_max = max(all_y) + 1 if all_y else 8
        else:
            all_y = [val for curve in curves_data for val in curve['y']]
            y_max = max(all_y) + 1 if all_y else 8
    else:
        y_min = 0
        if settings['y_min'] != '' and settings['y_max'] != '':
            try:
                y_min = float(settings['y_min'])
                y_max = float(settings['y_max'])
            except:
                all_y = [val for curve in curves_data for val in curve['y']]
                y_max = max(all_y) + 1 if all_y else 1
        else:
            all_y = [val for curve in curves_data for val in curve['y']]
            y_max = max(all_y) + 1 if all_y else 1
    
    # Handle axis break if enabled
    axis_break_enabled = settings.get('axis_break_enabled', False)
    # Handle both boolean and string 'true'/'false' from JSON
    if isinstance(axis_break_enabled, str):
        axis_break_enabled = axis_break_enabled.lower() in ('true', '1', 'yes')
    axis_break_type = settings.get('axis_break_type', 'x')
    axis_break_start = settings.get('axis_break_start', '')
    axis_break_end = settings.get('axis_break_end', '')
    
    print(f"CREATE_PLOT - Axis break: enabled={axis_break_enabled} (type: {type(axis_break_enabled)}), start='{axis_break_start}', end='{axis_break_end}'")
    
    # Check if axis break should be applied
    # Handle both string and numeric inputs from JavaScript
    # JavaScript sends numbers as numbers, not strings
    axis_break_start_valid = (axis_break_start is not None and 
                             axis_break_start != '' and 
                             str(axis_break_start).strip() != '')
    axis_break_end_valid = (axis_break_end is not None and 
                           axis_break_end != '' and 
                           str(axis_break_end).strip() != '')
    
    print(f"CREATE_PLOT - After validation: start_valid={axis_break_start_valid}, end_valid={axis_break_end_valid}")
    
    if axis_break_enabled and axis_break_type == 'x' and axis_break_start_valid and axis_break_end_valid:
        print(f"CREATE_PLOT - Axis break condition met, attempting to create break...")
        try:
            # Convert to float (handles both string and numeric)
            break_start = float(axis_break_start)
            break_end = float(axis_break_end)
            print(f"CREATE_PLOT - Parsed break: start={break_start}, end={break_end}, x_min={x_min}, x_max={x_max}")
            
            if break_start < break_end and x_min < break_start and break_end < x_max:
                print(f"CREATE_PLOT - Creating axis break!")
                # Create a broken x-axis using subplots
                fig.clf()
                
                # Use calculated y limits
                y_plot_min = y_min
                y_plot_max = y_max
                
                # Create two subplots side by side
                gs = gridspec.GridSpec(1, 2, width_ratios=[break_start - x_min, x_max - break_end], 
                                     wspace=0.05, left=0.1, right=0.95, top=0.9, bottom=0.1)
                ax1 = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
                
                # Hide the spines between the axes
                ax1.spines['right'].set_visible(False)
                ax2.spines['left'].set_visible(False)
                ax1.yaxis.tick_left()
                ax2.yaxis.tick_right()
                ax2.tick_params(labelleft=False)
                
                # Add diagonal lines to indicate break
                d = 0.015  # Size of diagonal lines
                kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linewidth=1)
                ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
                ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
                kwargs.update(transform=ax2.transAxes)
                ax2.plot((-d, +d), (-d, +d), **kwargs)
                ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
                
                # Plot curves on both axes
                for curve_data in curves_data:
                    config = curve_data['config']
                    x = np.array(curve_data['x'])
                    y = np.array(curve_data['y'])
                    
                    if settings['y_scale'] == 'log':
                        y_plot = y
                    else:
                        y_plot = y
                    
                    # Split data at break
                    mask_before = x <= break_start
                    mask_after = x >= break_end
                    
                    # Marker kwargs
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
                    if np.any(mask_before):
                        x_before = x[mask_before]
                        y_before = y_plot[mask_before]
                        
                        if config['show_line']:
                            ax1.plot(x_before, y_before,
                                   linestyle=config['line_style'],
                                   color=config['color'],
                                   linewidth=config['line_width'],
                                   label=config['name'])
                        ax1.plot(x_before, y_before, **marker_kwargs)
                    
                    # Plot on second axis
                    if np.any(mask_after):
                        x_after = x[mask_after]
                        y_after = y_plot[mask_after]
                        
                        if config['show_line']:
                            ax2.plot(x_after, y_after,
                                   linestyle=config['line_style'],
                                   color=config['color'],
                                   linewidth=config['line_width'],
                                   label=None)
                        marker_kwargs_copy = marker_kwargs.copy()
                        marker_kwargs_copy['label'] = None
                        ax2.plot(x_after, y_after, **marker_kwargs_copy)
                
                # Set axis limits
                ax1.set_xlim(x_min, break_start)
                ax2.set_xlim(break_end, x_max)
                ax1.set_ylim(y_plot_min, y_plot_max)
                ax2.set_ylim(y_plot_min, y_plot_max)
                
                # Set x-axis ticks
                x_tick_mode_break = settings.get('x_tick_mode', 'auto')
                x_tick_interval_break = settings.get('x_tick_interval', 0)
                
                # Convert to appropriate types if needed
                if isinstance(x_tick_interval_break, str):
                    try:
                        x_tick_interval_break = float(x_tick_interval_break)
                    except:
                        x_tick_interval_break = 0
                
                if x_tick_mode_break == 'custom' and x_tick_interval_break > 0:
                    tick_interval = float(x_tick_interval_break)
                    ticks_before = np.arange(x_min, break_start + tick_interval, tick_interval)
                    ticks_after = np.arange(break_end, x_max + tick_interval, tick_interval)
                    # Filter ticks to be within the respective axis ranges
                    ticks_before = ticks_before[(ticks_before >= x_min) & (ticks_before <= break_start)]
                    ticks_after = ticks_after[(ticks_after >= break_end) & (ticks_after <= x_max)]
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
                
                # Grid
                if settings['show_grid']:
                    ax1.grid(True, alpha=0.3, linestyle='--')
                    ax2.grid(True, alpha=0.3, linestyle='--')
                
                plt.tight_layout()
                return fig
        except Exception as e:
            print(f"Warning: Could not create axis break: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to regular plot
    
    ax.set_xlim(x_min, x_max)
    
    # Set x-axis tick marks (this is independent of data point spacing)
    x_tick_mode = settings.get('x_tick_mode', 'auto')
    x_tick_interval = settings.get('x_tick_interval', 0)
    
    print(f"CREATE_PLOT - Tick settings: mode={x_tick_mode}, interval={x_tick_interval} (type: {type(x_tick_interval)})")
    
    # Convert to appropriate types if needed
    if isinstance(x_tick_interval, str):
        try:
            x_tick_interval = float(x_tick_interval)
        except:
            x_tick_interval = 0
    elif x_tick_interval is None:
        x_tick_interval = 0
    
    # Apply custom tick marks if specified
    # Note: This controls tick marks on the axis, NOT data point spacing
    if x_tick_mode == 'custom' and x_tick_interval and x_tick_interval > 0:
        tick_interval = float(x_tick_interval)
        # Generate ticks starting from x_min
        x_ticks = np.arange(x_min, x_max + tick_interval, tick_interval)
        # Filter ticks to be within the x-axis range
        x_ticks = x_ticks[(x_ticks >= x_min) & (x_ticks <= x_max)]
        print(f"CREATE_PLOT - Setting custom ticks: {x_ticks.tolist()}")
        if len(x_ticks) > 0:
            ax.set_xticks(x_ticks)
            # Ensure tick labels are shown
            ax.tick_params(axis='x', which='major', labelsize=10)
            print(f"CREATE_PLOT - Ticks set successfully!")
        else:
            print(f"CREATE_PLOT - WARNING: No ticks generated (x_min={x_min}, x_max={x_max}, interval={tick_interval})")
    else:
        print(f"CREATE_PLOT - Using auto ticks (mode={x_tick_mode}, interval={x_tick_interval})")
    # Otherwise, use auto ticks (matplotlib default)
    
    # Make axes meet at origin
    ax.spines['left'].set_position(('data', x_min))
    
    # Set y-axis limits (already calculated above)
    ax.set_ylim(y_min, y_max)
    
    # For log scale: set integer tick marks
    if settings['y_scale'] == 'log':
        # Set integer tick marks (0, 1, 2, 3, 4, 5, etc. or 0, 2, 4, 6, 8, 10)
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
    
    # Make axes meet at origin
    ax.spines['bottom'].set_position(('data', y_min))
    
    # Title
    if settings['title']:
        ax.set_title(settings['title'], fontsize=12, fontweight='bold')
    
    # Legend
    if settings['show_legend']:
        ax.legend(loc='best', framealpha=0.9)
    
    # Grid
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

# =============================================================================
# File Output Functions
# =============================================================================

def get_next_synthetic_name():
    """
    Generate the next available name for a synthetic plot.
    Names follow pattern: AA, AB, AC, ... AZ, BA, BB, ... ZZ
    Scans existing folders in SYNTHETIC_DIR to find the next available name.
    """
    import string
    
    # Get existing folder names
    existing = set()
    if os.path.exists(SYNTHETIC_DIR):
        for item in os.listdir(SYNTHETIC_DIR):
            if os.path.isdir(os.path.join(SYNTHETIC_DIR, item)):
                # Extract the name part (could be just "AA" or with other suffixes)
                existing.add(item.upper())
    
    # Generate names: AA, AB, ... AZ, BA, BB, ... ZZ
    for first in string.ascii_uppercase:
        for second in string.ascii_uppercase:
            name = f"{first}{second}"
            if name not in existing:
                return name
    
    # If all 676 combinations used, fall back to timestamp
    from datetime import datetime
    return f"ZZ_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def save_plot_and_data(settings, curves_data, x_values):
    """
    Save the plot as PNG (and optionally SVG) and the data as CSV.
    Creates a subfolder for each plot to match the first_examples structure:
    plots/synthetic/{name}/{name}.png
    plots/synthetic/{name}/{name}-original.csv
    
    Returns:
    --------
    dict : Paths to saved files
    """
    # Generate auto-incrementing name (AA, AB, AC, etc.)
    base_filename = get_next_synthetic_name()
    
    # Create subfolder for this plot (like first_examples/A/A-1/)
    plot_folder = os.path.join(SYNTHETIC_DIR, base_filename)
    os.makedirs(plot_folder, exist_ok=True)
    
    # Create the figure
    fig = create_plot(settings, curves_data, x_values)
    
    # Save PNG inside the subfolder
    png_path = os.path.join(plot_folder, f'{base_filename}.png')
    dpi = settings.get('dpi', 150)
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
    
    # Save SVG if requested
    svg_path = None
    if settings['save_svg']:
        svg_path = os.path.join(plot_folder, f'{base_filename}.svg')
        fig.savefig(svg_path, format='svg', bbox_inches='tight')
    
    plt.close(fig)
    
    # Save CSV with -original suffix (required by app.py)
    csv_path = os.path.join(plot_folder, f'{base_filename}-original.csv')
    save_csv(curves_data, x_values, settings, csv_path)
    
    return {
        'png': png_path,
        'svg': svg_path,
        'csv': csv_path,
        'filename': base_filename,
        'folder': plot_folder
    }

def save_csv(curves_data, x_values, settings, filepath):
    """
    Save curve data to CSV file in the same format as extracted data.
    Format: x1, y1, x2, y2, ... (paired columns for each curve)
    Y-values are saved in log10 scale (e.g., 3 for 10^3 = 1000)
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        # Header row
        headers = []
        for curve_data in curves_data:
            config = curve_data['config']
            x_col = settings['x_label'] if settings['x_label'] else 'x'
            y_col = config['name']
            headers.extend([x_col, y_col])
        f.write(','.join(headers) + '\n')
        
        # Data rows - Y values saved in log10 scale
        n_points = len(x_values)
        for i in range(n_points):
            row = []
            for curve_data in curves_data:
                x_val = curve_data['x'][i]
                y_val = curve_data['y'][i]
                # Y values are already in log10 scale, keep them as-is
                # Round to reasonable precision
                row.extend([str(x_val), f'{y_val:.4f}'])
            f.write(','.join(row) + '\n')

# =============================================================================
# Flask Routes
# =============================================================================

@app.route('/')
def index():
    """Render the main page."""
    settings = load_settings()
    # Ensure we have curves configured
    if not settings['curves']:
        settings['curves'] = get_default_curves(settings['num_curves'])
    return render_template('synthetic.html', settings=settings)

@app.route('/get_settings')
def get_settings():
    """Return current settings as JSON."""
    settings = load_settings()
    if not settings['curves']:
        settings['curves'] = get_default_curves(settings['num_curves'])
    return jsonify(settings)

@app.route('/update_curves', methods=['POST'])
def update_curves():
    """Update the number of curves and return new curve configs."""
    data = request.json
    num_curves = int(data.get('num_curves', 3))
    
    settings = load_settings()
    current_curves = settings.get('curves', [])
    
    # Adjust curves list
    if num_curves > len(current_curves):
        # Add new curves
        for i in range(len(current_curves), num_curves):
            curve = DEFAULT_CURVE.copy()
            curve['name'] = f'Condition {i + 1}'
            curve['color'] = COLOR_PALETTE[i % len(COLOR_PALETTE)]
            trends = ['stable', 'down', 'up', 'kill_regrowth', 'mixed']
            curve['trend'] = trends[i % len(trends)]
            current_curves.append(curve)
    elif num_curves < len(current_curves):
        # Remove extra curves
        current_curves = current_curves[:num_curves]
    
    settings['curves'] = current_curves
    settings['num_curves'] = num_curves
    save_settings(settings)
    
    return jsonify({'curves': current_curves})

@app.route('/preview', methods=['POST'])
def preview():
    """Generate a preview of the plot."""
    settings = request.json
    
    # Ensure all new settings have default values if missing
    settings.setdefault('x_tick_mode', 'custom')
    settings.setdefault('x_tick_interval', 2)
    settings.setdefault('axis_break_enabled', False)
    settings.setdefault('axis_break_type', 'x')
    settings.setdefault('axis_break_start', '')
    settings.setdefault('axis_break_end', '')
    
    # Debug: Print received settings
    print("=" * 50)
    print("PREVIEW - Received settings:")
    print(f"  x_tick_mode: {settings.get('x_tick_mode')}")
    print(f"  x_tick_interval: {settings.get('x_tick_interval')} (type: {type(settings.get('x_tick_interval'))})")
    print(f"  axis_break_enabled: {settings.get('axis_break_enabled')} (type: {type(settings.get('axis_break_enabled'))})")
    print(f"  axis_break_start: '{settings.get('axis_break_start')}'")
    print(f"  axis_break_end: '{settings.get('axis_break_end')}'")
    print("=" * 50)
    
    # Generate data
    x_values, curves_data = generate_all_curves(settings)
    
    # Create plot and convert to base64
    fig = create_plot(settings, curves_data, x_values)
    img_base64 = fig_to_base64(fig)
    
    # Save settings for persistence
    save_settings(settings)
    
    return jsonify({
        'success': True,
        'image': img_base64,
        'x_values': x_values,
        'curves_data': curves_data
    })

@app.route('/save', methods=['POST'])
def save():
    """Save the plot and data to files."""
    settings = request.json
    
    # Ensure all new settings have default values if missing
    settings.setdefault('x_tick_mode', 'custom')
    settings.setdefault('x_tick_interval', 2)
    settings.setdefault('axis_break_enabled', False)
    settings.setdefault('axis_break_type', 'x')
    settings.setdefault('axis_break_start', '')
    settings.setdefault('axis_break_end', '')
    
    # Generate data
    x_values, curves_data = generate_all_curves(settings)
    
    # Save files
    saved_files = save_plot_and_data(settings, curves_data, x_values)
    
    # Save settings for persistence
    save_settings(settings)
    
    return jsonify({
        'success': True,
        'files': saved_files,
        'message': f"Saved to {saved_files['filename']}"
    })

@app.route('/reset', methods=['POST'])
def reset():
    """Reset all settings to defaults."""
    settings = DEFAULT_SETTINGS.copy()
    settings['curves'] = get_default_curves(settings['num_curves'])
    save_settings(settings)
    return jsonify(settings)

@app.route('/regenerate', methods=['POST'])
def regenerate():
    """Regenerate curve data with same settings (new random values)."""
    settings = request.json
    
    # Ensure all new settings have default values if missing
    settings.setdefault('x_tick_mode', 'custom')
    settings.setdefault('x_tick_interval', 2)
    settings.setdefault('axis_break_enabled', False)
    settings.setdefault('axis_break_type', 'x')
    settings.setdefault('axis_break_start', '')
    settings.setdefault('axis_break_end', '')
    
    # Generate new data
    x_values, curves_data = generate_all_curves(settings)
    
    # Create plot and convert to base64
    fig = create_plot(settings, curves_data, x_values)
    img_base64 = fig_to_base64(fig)
    
    return jsonify({
        'success': True,
        'image': img_base64,
        'x_values': x_values,
        'curves_data': curves_data
    })

# =============================================================================
# Plot Editor Routes
# =============================================================================

@app.route('/get_existing_plots')
def get_existing_plots():
    """Get list of existing synthetic plots that can be edited."""
    plots = []
    
    if os.path.exists(SYNTHETIC_DIR):
        for item in sorted(os.listdir(SYNTHETIC_DIR)):
            item_path = os.path.join(SYNTHETIC_DIR, item)
            if os.path.isdir(item_path):
                # Check for PNG and CSV files
                png_file = os.path.join(item_path, f'{item}.png')
                csv_file = os.path.join(item_path, f'{item}-original.csv')
                
                # Also check for copy files
                if not os.path.exists(png_file):
                    # Look for any PNG in the folder
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

@app.route('/load_plot_for_edit/<plot_name>')
def load_plot_for_edit(plot_name):
    """Load an existing plot's data and settings for editing."""
    plot_folder = os.path.join(SYNTHETIC_DIR, plot_name)
    
    if not os.path.exists(plot_folder):
        return jsonify({'success': False, 'error': 'Plot folder not found'})
    
    # Find the CSV file
    csv_file = os.path.join(plot_folder, f'{plot_name}-original.csv')
    if not os.path.exists(csv_file):
        # Try to find any -original.csv
        for f in os.listdir(plot_folder):
            if f.endswith('-original.csv'):
                csv_file = os.path.join(plot_folder, f)
                break
    
    if not os.path.exists(csv_file):
        return jsonify({'success': False, 'error': 'CSV file not found'})
    
    # Parse the CSV file
    curves_data = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Parse header to get curve names
        header = lines[0].strip().split(',')
        num_curves = len(header) // 2
        
        curve_names = []
        for i in range(num_curves):
            y_col_name = header[i * 2 + 1].strip()
            curve_names.append(y_col_name)
        
        # Parse data rows
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
        
        # Build curves_data with default visual settings
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
        
        # Find the PNG file for preview
        png_file = os.path.join(plot_folder, f'{plot_name}.png')
        if not os.path.exists(png_file):
            for f in os.listdir(plot_folder):
                if f.endswith('.png') and not '_copy' in f:
                    png_file = os.path.join(plot_folder, f)
                    break
        
        # Read PNG as base64
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

@app.route('/preview_edit', methods=['POST'])
def preview_edit():
    """Generate a preview of the edited plot with visual changes only."""
    try:
        data = request.json
        curves_data = data['curves_data']
        settings = data['settings']
        x_values = data['x_values']
        
        # Ensure settings has all required keys with defaults
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
        
        # Create plot with the modified visual settings
        fig = create_plot(settings, curves_data, x_values)
        img_base64 = fig_to_base64(fig)
        
        return jsonify({
            'success': True,
            'image': img_base64
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/save_edit', methods=['POST'])
def save_edit():
    """Save the edited plot as a copy."""
    try:
        data = request.json
        original_name = data['original_name']
        curves_data = data['curves_data']
        settings = data['settings']
        x_values = data['x_values']
        
        # Ensure settings has all required keys with defaults
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
        
        # Find the next copy number
        copy_num = 1
        while True:
            copy_name = f'{original_name}_copy{copy_num}'
            png_path = os.path.join(plot_folder, f'{copy_name}.png')
            if not os.path.exists(png_path):
                break
            copy_num += 1
        
        # Create the figure
        fig = create_plot(settings, curves_data, x_values)
    
        # Save PNG
        png_path = os.path.join(plot_folder, f'{copy_name}.png')
        dpi = settings.get('dpi', 150)
        fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
        
        # Save SVG if requested
        svg_path = None
        if settings.get('save_svg', False):
            svg_path = os.path.join(plot_folder, f'{copy_name}.svg')
            fig.savefig(svg_path, format='svg', bbox_inches='tight')
        
        plt.close(fig)
        
        # Save CSV
        csv_path = os.path.join(plot_folder, f'{copy_name}-original.csv')
        save_csv(curves_data, x_values, settings, csv_path)
        
        return jsonify({
            'success': True,
            'files': {
                'png': png_path,
                'svg': svg_path,
                'csv': csv_path,
                'filename': copy_name,
                'folder': plot_folder
            },
            'message': f'Saved as {copy_name}'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    print(f"Synthetic plots will be saved to: {SYNTHETIC_DIR}")
    print("Starting Synthetic Time-Kill Plot Generator...")
    print("Open http://127.0.0.1:5001 in your browser")
    app.run(debug=True, port=5001)
