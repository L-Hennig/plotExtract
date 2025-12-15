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
        
        # Plot markers
        ax.plot(x, y_plot,
               marker=config['marker'],
               linestyle='none' if config['show_line'] else config['line_style'],
               color=config['color'],
               markersize=config['marker_size'],
               label=config['name'] if not config['show_line'] else None)
    
    # Set axis labels
    x_label = settings['x_label']
    if settings['x_unit']:
        x_label += f" ({settings['x_unit']})"
    ax.set_xlabel(x_label, fontsize=11)
    
    y_label = settings['y_label']
    if settings['y_unit']:
        y_label += f" ({settings['y_unit']})"
    ax.set_ylabel(y_label, fontsize=11)
    
    # Set axis limits
    if settings['x_min'] != '' and settings['x_max'] != '':
        try:
            ax.set_xlim(float(settings['x_min']), float(settings['x_max']))
        except:
            pass
    
    # For log scale: use linear axis but with log10 values, starting at 0
    if settings['y_scale'] == 'log':
        # Set y-axis to start at 0 by default
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
                # Auto-calculate from data
                all_y = [val for curve in curves_data for val in curve['y']]
                y_max = max(all_y) + 1
        else:
            # Auto-calculate from data
            all_y = [val for curve in curves_data for val in curve['y']]
            y_max = max(all_y) + 1
        
        ax.set_ylim(y_min, y_max)
        
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
    else:
        # Linear scale
        if settings['y_min'] != '' and settings['y_max'] != '':
            try:
                ax.set_ylim(float(settings['y_min']), float(settings['y_max']))
            except:
                pass
    
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

def save_plot_and_data(settings, curves_data, x_values):
    """
    Save the plot as PNG (and optionally SVG) and the data as CSV.
    
    Returns:
    --------
    dict : Paths to saved files
    """
    # Generate timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f'timekill_{timestamp}'
    
    # Create the figure
    fig = create_plot(settings, curves_data, x_values)
    
    # Save PNG
    png_path = os.path.join(SYNTHETIC_DIR, f'{base_filename}.png')
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    
    # Save SVG if requested
    svg_path = None
    if settings['save_svg']:
        svg_path = os.path.join(SYNTHETIC_DIR, f'{base_filename}.svg')
        fig.savefig(svg_path, format='svg', bbox_inches='tight')
    
    plt.close(fig)
    
    # Save CSV
    csv_path = os.path.join(SYNTHETIC_DIR, f'{base_filename}.csv')
    save_csv(curves_data, x_values, settings, csv_path)
    
    return {
        'png': png_path,
        'svg': svg_path,
        'csv': csv_path,
        'filename': base_filename
    }

def save_csv(curves_data, x_values, settings, filepath):
    """
    Save curve data to CSV file in the same format as extracted data.
    Format: x1, y1, x2, y2, ... (paired columns for each curve)
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
        
        # Data rows
        n_points = len(x_values)
        for i in range(n_points):
            row = []
            for curve_data in curves_data:
                x_val = curve_data['x'][i]
                y_val = curve_data['y'][i]
                # For log scale, convert back to actual values
                if settings['y_scale'] == 'log':
                    y_val = 10 ** y_val
                row.extend([str(x_val), str(y_val)])
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
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    print(f"Synthetic plots will be saved to: {SYNTHETIC_DIR}")
    print("Starting Synthetic Time-Kill Plot Generator...")
    print("Open http://127.0.0.1:5001 in your browser")
    app.run(debug=True, port=5001)
