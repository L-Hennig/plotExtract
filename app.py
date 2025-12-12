import os
import subprocess
import glob
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
PROMPTS_DIR = os.path.join(BASE_DIR, 'prompts')

def get_input_images(directory):
    """
    Recursively find only original input .png files in the plots directory.
    Excludes output files like replot, comparison, interpolated, pointwise images.
    Also excludes files inside version folders (e.g., A-1.p2.v1/)
    """
    images = []
    exclude_patterns = [
        '-replot', 'comparison_', 'interpolated_', 'pointwise_',
        '.mistral.out', '.claude.out', '_VS_'
    ]
    # Pattern to detect version folders: {name}.p{n}.v{n}
    import re
    version_folder_pattern = re.compile(r'\.p\d+\.v\d+[\\/]')
    
    for ext in ['*.png', '*.PNG']:
        for img_path in glob.glob(os.path.join(directory, '**', ext), recursive=True):
            rel_path = os.path.relpath(img_path, directory)
            filename = os.path.basename(img_path)
            
            # Skip if inside a version folder
            if version_folder_pattern.search(rel_path):
                continue
            
            # Skip if filename contains any exclude pattern
            if not any(pattern in filename for pattern in exclude_patterns):
                images.append(os.path.relpath(img_path, PLOTS_DIR))
    
    return sorted(images)

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
    Now searches inside version folders and returns the latest version."""
    import re
    
    image_dir = os.path.dirname(os.path.join(PLOTS_DIR, image_path))
    image_name = os.path.basename(image_path)
    base_name = os.path.splitext(image_name)[0]
    
    # Get prompt short name (e.g., prompt_1.py -> p1)
    prompt_name = os.path.splitext(prompt_file)[0].replace('prompt_', 'p')
    
    # Look for version folders matching this image+prompt
    version_pattern = re.compile(rf'^{re.escape(base_name)}\.{re.escape(prompt_name)}\.v(\d+)$')
    
    latest_version = 0
    latest_file = None
    
    if os.path.exists(image_dir):
        for item in os.listdir(image_dir):
            match = version_pattern.match(item)
            if match:
                version_num = int(match.group(1))
                version_dir = os.path.join(image_dir, item)
                extracted_file = os.path.join(version_dir, f"{image_name}.{prompt_name}.mistral.out_data")
                
                if os.path.exists(extracted_file) and version_num > latest_version:
                    latest_version = version_num
                    latest_file = extracted_file
    
    if latest_file:
        return os.path.relpath(latest_file, PLOTS_DIR)
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
        'other': []
    }
    
    if not os.path.exists(image_dir):
        return outputs
    
    # Add the original image
    original_path = os.path.join(image_dir, image_name)
    if os.path.exists(original_path):
        outputs['images'].append({
            'path': os.path.relpath(original_path, PLOTS_DIR),
            'label': 'Original Input',
            'filename': image_name
        })
    
    # Add original CSV if exists
    original_csv = os.path.join(image_dir, f"{base_name}-original.csv")
    if os.path.exists(original_csv):
        outputs['data'].append({
            'path': os.path.relpath(original_csv, PLOTS_DIR),
            'label': 'Original Data',
            'filename': f"{base_name}-original.csv"
        })
    
    # If version_dir is specified, only scan that folder
    if version_dir and os.path.exists(version_dir):
        version_label = os.path.basename(version_dir)
        _scan_version_folder(version_dir, version_label, outputs, PLOTS_DIR)
    else:
        # Scan for all version folders matching pattern: {base_name}.p*.v*
        import re
        version_pattern = re.compile(rf'^{re.escape(base_name)}\.p\d+\.v\d+$')
        
        for item in os.listdir(image_dir):
            item_path = os.path.join(image_dir, item)
            if os.path.isdir(item_path) and version_pattern.match(item):
                version_label = item
                _scan_version_folder(item_path, version_label, outputs, PLOTS_DIR)
    
    return outputs

def _scan_version_folder(folder_path, version_label, outputs, plots_dir):
    """Helper to scan a version folder and add files to outputs."""
    for f in os.listdir(folder_path):
        full_path = os.path.join(folder_path, f)
        rel_path = os.path.relpath(full_path, plots_dir)
        
        label = None
        
        if f.endswith('.png') or f.endswith('.jpg'):
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
    images = get_input_images(PLOTS_DIR)
    prompts = get_prompts()
    return render_template('index.html', images=images, prompts=prompts)

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
        prompt_name = os.path.splitext(prompt_file)[0].replace('prompt_', 'p')
        
        # Find latest version folder
        version_pattern = re.compile(rf'^{re.escape(base_name)}\.{re.escape(prompt_name)}\.v(\d+)$')
        latest_version = 0
        latest_dir = None
        
        if os.path.exists(image_dir):
            for item in os.listdir(image_dir):
                match = version_pattern.match(item)
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
    Run extraction and optionally interpolation/pointwise comparison.
    Returns results after all tasks complete.
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
    
    full_image_path = os.path.join(PLOTS_DIR, image_path)
    full_prompt_path = os.path.join(PROMPTS_DIR, prompt_file)
    
    console_output = []
    success = True
    version_dir = None  # Will be set after extraction
    
    # Get CSV info
    csv_info = check_csv_exists(image_path, prompt_file)
    original_csv = os.path.join(PLOTS_DIR, csv_info['original']['path'])
    
    # Get prompt short name for extracted file
    prompt_name = os.path.splitext(prompt_file)[0].replace('prompt_', 'p')
    image_dir = os.path.dirname(full_image_path)
    image_name = os.path.basename(image_path)
    
    # Step 1: Run extraction
    console_output.append("=" * 60)
    console_output.append("STEP 1: Running Plot Extraction")
    console_output.append("=" * 60)
    console_output.append(f"Image: {image_path}")
    console_output.append(f"Prompt: {prompt_file}")
    console_output.append("")
    
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
            console_output.append(f"\n[ERROR] Extraction failed with exit code {result.returncode}")
        else:
            console_output.append("\n[SUCCESS] Extraction completed!")
            
    except subprocess.TimeoutExpired:
        success = False
        console_output.append("[ERROR] Extraction timed out after 5 minutes")
    except Exception as e:
        success = False
        console_output.append(f"[ERROR] {str(e)}")
    
    # Determine extracted CSV path (inside version folder)
    if version_dir:
        extracted_csv = os.path.join(version_dir, f"{image_name}.{prompt_name}.mistral.out_data")
    else:
        # Fallback if VERSION_DIR not found
        extracted_csv = os.path.join(image_dir, f"{image_name}.{prompt_name}.mistral.out_data")
    
    # Step 2: Run interpolation if requested
    if run_interpolation and success:
        console_output.append("")
        console_output.append("=" * 60)
        console_output.append("STEP 2: Running Interpolation")
        console_output.append("=" * 60)
        
        if not os.path.exists(original_csv):
            console_output.append(f"[WARNING] Original CSV not found: {original_csv}")
            console_output.append("[SKIPPED] Interpolation skipped - missing original CSV")
        elif not os.path.exists(extracted_csv):
            console_output.append(f"[WARNING] Extracted data not found: {extracted_csv}")
            console_output.append("[SKIPPED] Interpolation skipped - missing extracted data")
        else:
            console_output.append(f"Original: {original_csv}")
            console_output.append(f"Extracted: {extracted_csv}")
            console_output.append(f"Output dir: {version_dir}")
            console_output.append(f"Axis range: X=[{left_x}, {right_x}], Y=[{bottom_y}, {top_y}]")
            console_output.append("")
            
            try:
                # Pass version_dir as output directory
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
    
    # Step 3: Run pointwise if requested
    if run_pointwise and success:
        console_output.append("")
        console_output.append("=" * 60)
        console_output.append("STEP 3: Running Pointwise Comparison")
        console_output.append("=" * 60)
        
        if not os.path.exists(original_csv):
            console_output.append(f"[WARNING] Original CSV not found: {original_csv}")
            console_output.append("[SKIPPED] Pointwise skipped - missing original CSV")
        elif not os.path.exists(extracted_csv):
            console_output.append(f"[WARNING] Extracted data not found: {extracted_csv}")
            console_output.append("[SKIPPED] Pointwise skipped - missing extracted data")
        else:
            console_output.append(f"Extracted: {extracted_csv}")
            console_output.append(f"Original: {original_csv}")
            console_output.append(f"Output dir: {version_dir}")
            console_output.append(f"Axis range: X=[{left_x}, {right_x}], Y=[{bottom_y}, {top_y}]")
            console_output.append("")
            
            try:
                # Pass version_dir as output directory
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
    
    console_output.append("")
    console_output.append("=" * 60)
    console_output.append("ALL TASKS COMPLETED")
    console_output.append("=" * 60)
    
    # Get updated outputs - only for the current version
    outputs = get_output_files(image_path, prompt_file, version_dir)
    csv_status = check_csv_exists(image_path, prompt_file)
    
    return jsonify({
        'success': success,
        'console': '\n'.join(console_output),
        'outputs': outputs,
        'csv_status': csv_status,
        'version_dir': version_dir
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
