import os
import subprocess
import glob
import json
import copy
import math
import sys
import numpy as np

# Set matplotlib backend to Agg (non-interactive) BEFORE importing pyplot
# This prevents threading issues with Tkinter when using Flask
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import io
import base64
import shutil
import threading
import time
import uuid
from collections import Counter
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect
from jinja2 import TemplateNotFound
from werkzeug.utils import secure_filename

UI_DIR = os.path.dirname(os.path.abspath(__file__))

# Repo root: Plot_Extract/
REPO_ROOT = os.path.abspath(os.path.join(UI_DIR, '..'))

# Serve templates/static from the WebExtract folder, but run the backend
# pipelines using the Prototype folder (scripts, plots, prompts).
app = Flask(
    __name__,
    template_folder=os.path.join(UI_DIR, 'templates'),
    static_folder=os.path.join(UI_DIR, 'static'),
)

# Limit upload size (25 MB)
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024

# Base directory (backend/scripts root)
BASE_DIR = REPO_ROOT
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
PROMPTS_DIR = os.path.join(BASE_DIR, 'prompts')
PROMPTS_V2_DIR = os.path.join(BASE_DIR, 'plot_extract_v2', 'prompts')
PROMPTS_V2_CHAINS_DIR = os.path.join(PROMPTS_V2_DIR, 'chains')
SYNTHETIC_DIR = os.path.join(PLOTS_DIR, 'synthetic')

# UI folders
UI_EXAMPLES_DIR = os.path.join(UI_DIR, 'examples')

ALLOWED_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}

def _pick_python_exe() -> str:
    # Prefer repo-root venv so subprocesses see the same installed packages
    # (e.g. google-generativeai) even if the UI server was started with a
    # different global Python.
    venv_py = os.path.join(REPO_ROOT, '.venv', 'Scripts', 'python.exe')
    if os.name == 'nt' and os.path.exists(venv_py):
        return venv_py
    return sys.executable or 'python'


PYTHON_EXE = _pick_python_exe()


def _safe_abs_under_plots(rel_path: str) -> str | None:
    """Return absolute path under PLOTS_DIR for a plots-relative path, else None."""
    if not rel_path:
        return None
    try:
        p = str(rel_path).replace('\\', '/').lstrip('/')
    except Exception:
        return None
    # Prevent traversal
    if any(part == '..' for part in p.split('/')):
        return None
    abs_p = os.path.abspath(os.path.join(PLOTS_DIR, p.replace('/', os.sep)))
    if not abs_p.startswith(os.path.abspath(PLOTS_DIR)):
        return None
    return abs_p


def _safe_abs_under_ui_examples(rel_path: str) -> str | None:
    """Return absolute path under UI_EXAMPLES_DIR for a examples-relative path, else None."""
    if not rel_path:
        return None
    try:
        p = str(rel_path).replace('\\', '/').lstrip('/')
    except Exception:
        return None
    if any(part == '..' for part in p.split('/')):
        return None
    abs_p = os.path.abspath(os.path.join(UI_EXAMPLES_DIR, p.replace('/', os.sep)))
    if not abs_p.startswith(os.path.abspath(UI_EXAMPLES_DIR)):
        return None
    return abs_p


def _resolve_ui_example_to_plots_rel(example_rel_path: str) -> str | None:
    """Map a UI example (WebExtract/examples) to its canonical plots-relative image path."""
    example_name = os.path.basename(example_rel_path or '').strip()
    if not example_name:
        return None

    example_name_lower = example_name.lower()
    example_stem_lower = os.path.splitext(example_name_lower)[0]
    candidates = []

    for root, _dirs, files in os.walk(PLOTS_DIR):
        for fname in files:
            if fname.lower() != example_name_lower:
                continue
            if os.path.splitext(fname)[1].lower() not in ALLOWED_IMAGE_EXTS:
                continue
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, PLOTS_DIR).replace('\\', '/')
            candidates.append(rel_path)

    if not candidates:
        return None

    def _rank(rel_path: str):
        parts = rel_path.split('/')
        top = (parts[0].lower() if parts else '')
        parent = (parts[-2].lower() if len(parts) >= 2 else '')
        top_rank = 0
        if top == 'synthetic':
            top_rank = 0
        elif top == 'first_examples':
            top_rank = 1
        elif top == 'quick_test':
            top_rank = 2
        else:
            top_rank = 3
        parent_rank = 0 if parent == example_stem_lower else 1
        return (top_rank, parent_rank, len(parts), rel_path.lower())

    candidates.sort(key=_rank)
    return candidates[0]


def _list_v2_version_dirs(image_path: str, prompt_name: str) -> list[tuple[int, str]]:
    """Return list of (version_num, version_dir_abs) for v2 runs of image+prompt."""
    import re

    if not image_path or not prompt_name:
        return []

    image_abs = _safe_abs_under_plots(image_path)
    if not image_abs:
        return []

    image_dir = os.path.dirname(image_abs)
    image_name = os.path.basename(image_abs)
    base_name = os.path.splitext(image_name)[0]
    name_for_folder = image_name.replace('.', '_')
    full_prompt_name = f"pv2_{prompt_name}"

    pat_new = re.compile(rf'^{re.escape(name_for_folder)}\.{re.escape(full_prompt_name)}\.v(\d+)(?:\.key\d+)?(?:\.web)?$')
    pat_old = re.compile(rf'^{re.escape(base_name)}\.{re.escape(full_prompt_name)}\.v(\d+)(?:\.key\d+)?(?:\.web)?$')

    out: list[tuple[int, str]] = []
    if not os.path.isdir(image_dir):
        return out

    for item in os.listdir(image_dir):
        item_path = os.path.join(image_dir, item)
        if not os.path.isdir(item_path):
            continue
        m = pat_new.match(item) or pat_old.match(item)
        if not m:
            continue
        try:
            v = int(m.group(1))
        except Exception:
            continue
        out.append((v, item_path))

    out.sort(key=lambda t: t[0], reverse=True)
    return out


def _read_v2_progress_snapshot(version_dir: str) -> dict:
    """Return parsed _extraction_progress.json for a v2 run folder, else {}."""
    version_dir = _resolve_path_under_plots(version_dir)
    progress_path = os.path.join(version_dir, '_extraction_progress.json')
    if not os.path.isfile(progress_path):
        return {}
    try:
        with open(progress_path, 'r', encoding='utf-8') as f:
            pj = json.load(f)
        return pj if isinstance(pj, dict) else {}
    except Exception:
        return {}


def _read_v2_progress_console(version_dir: str) -> tuple[str, bool]:
    """Return (console_text, success) from a v2 version folder if possible."""
    version_dir = _resolve_path_under_plots(version_dir)
    pj = _read_v2_progress_snapshot(version_dir)
    if pj:
        console_text = str(pj.get('console_output') or '')
        status = str(pj.get('status') or '').strip().lower()
        success = status in {'completed', 'complete', 'success', 'succeeded'}
        if not status:
            cur_stage = pj.get('current_stage')
            tot = pj.get('total_stages')
            success = (cur_stage == tot) if (cur_stage is not None and tot is not None) else bool(console_text)
        return console_text, bool(success)

    # Fallback: try to find any tracking/log file.
    try:
        cand = None
        for f in os.listdir(version_dir):
            if f.endswith('.mistral.out_tracking') or f.endswith('.out_tracking'):
                cand = os.path.join(version_dir, f)
                break
        if cand and os.path.isfile(cand):
            with open(cand, 'r', encoding='utf-8', errors='replace') as fp:
                return fp.read(), True
    except Exception:
        pass

    return '', False


def _resolve_path_under_plots(path_value):
    """Resolve absolute/relative paths that may have been persisted before moving the repo.

    If a stored path contains a ".../plots/..." segment, rebuild it under the current
    `PLOTS_DIR`. This keeps old batch/extraction state usable after renames/moves.
    """
    if not path_value:
        return path_value

    try:
        raw = str(path_value)
    except Exception:
        return path_value

    # Already valid on disk.
    try:
        if os.path.exists(raw):
            return raw
    except Exception:
        pass

    norm = raw.replace('\\', '/')
    marker = '/plots/'
    if marker in norm:
        suffix = norm.split(marker, 1)[1]
        candidate = os.path.join(PLOTS_DIR, suffix.replace('/', os.sep))
        return candidate

    # If it looks relative, assume it's relative to plots.
    try:
        if not os.path.isabs(raw):
            return os.path.join(PLOTS_DIR, raw.replace('/', os.sep))
    except Exception:
        pass

    return raw

# Create synthetic folder if it doesn't exist
os.makedirs(SYNTHETIC_DIR, exist_ok=True)

STATE_DIR = UI_DIR

# Settings/state files (kept under WebExtract so this UI stays isolated)
SETTINGS_FILE = os.path.join(STATE_DIR, 'synthetic_settings.json')
SETTINGS_FILE_V2 = os.path.join(STATE_DIR, 'synthetic_settings_v2.json')
EXTRACTION_STATE_FILE = os.path.join(STATE_DIR, 'extraction_state.json')
BATCH_RUNS_FILE = os.path.join(STATE_DIR, 'batch_runs.json')
batch_runs_lock = threading.Lock()


@app.errorhandler(TemplateNotFound)
def _handle_template_not_found(_e):
    # This app intentionally only ships the minimal UI templates.
    return (
        "<h1>Page not available</h1><p>This UI only provides the main extraction page at <a href='/'>/</a>.</p>",
        404,
    )

# =============================================================================
# Background Task Management
# =============================================================================

# In-memory task storage (for running tasks)
extraction_tasks = {}
extraction_tasks_lock = threading.Lock()


class TaskCancelledError(Exception):
    pass


def _is_cancel_requested(task_id: str) -> bool:
    with extraction_tasks_lock:
        task = extraction_tasks.get(task_id)
        return bool(task and task.get('cancel_requested'))


def _set_active_pid(task_id: str, pid):
    with extraction_tasks_lock:
        if task_id in extraction_tasks:
            extraction_tasks[task_id]['active_pid'] = pid


def _run_subprocess_with_cancel(task_id: str, args, *, cwd=None, timeout_s: float | None = None, env_overrides: dict | None = None):
    """Run a subprocess while allowing best-effort cancellation via cancel_requested.

    Returns a dict with keys: returncode, stdout, stderr.
    Raises:
      - TaskCancelledError if cancellation was requested.
      - subprocess.TimeoutExpired if timeout elapsed.
    """
    env = None
    if env_overrides:
        env = os.environ.copy()
        for k, v in env_overrides.items():
            env[str(k)] = str(v)

    proc = subprocess.Popen(
        args,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    _set_active_pid(task_id, proc.pid)

    stdout_lines = []
    stderr_lines = []

    def _reader(pipe, sink):
        try:
            for line in iter(pipe.readline, ''):
                sink.append(line)
        except Exception:
            pass
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    t_out = threading.Thread(target=_reader, args=(proc.stdout, stdout_lines))
    t_err = threading.Thread(target=_reader, args=(proc.stderr, stderr_lines))
    t_out.daemon = True
    t_err.daemon = True
    t_out.start()
    t_err.start()

    start = time.time()
    try:
        while proc.poll() is None:
            if _is_cancel_requested(task_id):
                try:
                    proc.terminate()
                except Exception:
                    pass
                # Give it a moment then force kill
                for _ in range(10):
                    if proc.poll() is not None:
                        break
                    time.sleep(0.1)
                if proc.poll() is None:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                raise TaskCancelledError('Cancellation requested')

            if timeout_s is not None and (time.time() - start) > float(timeout_s):
                try:
                    proc.terminate()
                except Exception:
                    pass
                for _ in range(10):
                    if proc.poll() is not None:
                        break
                    time.sleep(0.1)
                if proc.poll() is None:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                raise subprocess.TimeoutExpired(cmd=args, timeout=timeout_s)

            time.sleep(0.2)

        # Ensure pipes drained
        try:
            t_out.join(timeout=2)
            t_err.join(timeout=2)
        except Exception:
            pass

        return {
            'returncode': proc.returncode,
            'stdout': ''.join(stdout_lines),
            'stderr': ''.join(stderr_lines),
        }
    finally:
        _set_active_pid(task_id, None)

def load_extraction_state():
    """Load the last extraction result from file."""
    if os.path.exists(EXTRACTION_STATE_FILE):
        try:
            with open(EXTRACTION_STATE_FILE, 'r') as f:
                state = json.load(f)
            if isinstance(state, dict) and state.get('version_dir'):
                state['version_dir'] = _resolve_path_under_plots(state.get('version_dir'))
            return state
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


def _load_batch_runs_state():
    """Load batch run registry from file."""
    if os.path.exists(BATCH_RUNS_FILE):
        try:
            with open(BATCH_RUNS_FILE, 'r', encoding='utf-8') as f:
                state = json.load(f)
            if isinstance(state, dict):
                state.setdefault('next_batch_number', 1)
                state.setdefault('batches', {})

                # Normalize persisted version_dir values so batch pages keep working
                # after the repo is moved/renamed.
                for rec in (state.get('batches') or {}).values():
                    if not isinstance(rec, dict):
                        continue
                    for item in rec.get('items') or []:
                        if isinstance(item, dict) and item.get('version_dir'):
                            item['version_dir'] = _resolve_path_under_plots(item.get('version_dir'))

                return state
        except Exception as e:
            print(f"Error loading batch runs: {e}")
    return {'next_batch_number': 1, 'batches': {}}


def _save_batch_runs_state(state):
    """Save batch run registry to file."""
    try:
        with open(BATCH_RUNS_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"Error saving batch runs: {e}")


def _allocate_batch_number():
    with batch_runs_lock:
        state = _load_batch_runs_state()
        batch_number = int(state.get('next_batch_number', 1))
        state['next_batch_number'] = batch_number + 1
        _save_batch_runs_state(state)
        return batch_number


def _normalize_batch_name(name):
    if name is None:
        return None
    normalized = str(name).strip()
    return normalized if normalized else None


def _is_duplicate_batch_name(state, batch_name):
    """Check for duplicate batch names in persistent registry (case-insensitive)."""
    if not batch_name:
        return False
    name_key = batch_name.strip().lower()
    batches = state.get('batches') or {}
    for rec in batches.values():
        existing = rec.get('batch_name')
        if existing and str(existing).strip().lower() == name_key:
            return True
    return False


def _basename(p):
    try:
        return os.path.basename(p.replace('\\', '/'))
    except Exception:
        return str(p)


def _parse_percent_to_float(val):
    """Parse values like '54.3%' -> 54.3. Returns None if not parseable."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    try:
        s = str(val).strip()
        if s.endswith('%'):
            s = s[:-1].strip()
        return float(s)
    except Exception:
        return None


def _split_validation_reasons(reason_text):
    if not reason_text:
        return []
    reason_text = str(reason_text).strip()
    if not reason_text or reason_text.lower() in ('n/a', 'na', 'none'):
        return []
    # Allow multiple reasons, e.g. "X-axis, Y-axis" or "X-axis; Trends"
    parts = []
    for chunk in reason_text.replace(';', ',').replace('\n', ',').split(','):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    return parts


def create_batch_run_record(extraction_version, prompt, images, *, pipeline=None, task_id=None, batch_name=None):
    """Create a new batch run record and return the assigned batch_number."""
    normalized_name = _normalize_batch_name(batch_name)
    now = time.time()
    with batch_runs_lock:
        state = _load_batch_runs_state()
        if _is_duplicate_batch_name(state, normalized_name):
            raise ValueError("duplicate batch_name")
        batch_number = int(state.get('next_batch_number', 1))
        state['next_batch_number'] = batch_number + 1
        record = {
            'batch_number': batch_number,
            'batch_name': normalized_name,
            'status': 'running',
            'created_at': now,
            'started_at': now,
            'completed_at': None,
            'extraction_version': extraction_version,
            'pipeline': pipeline,
            'task_id': task_id,
            'prompt': prompt,
            'images': [{'image_path': p, 'name': _basename(p)} for p in (images or [])],
            'items': [
                {
                    'image_path': p,
                    'name': _basename(p),
                    'status': 'pending',
                    'time_s': None,
                    'summary': {},
                    'error': None,
                    'task_id': None,
                    'version_dir': None,
                    'console': None,
                }
                for p in (images or [])
            ],
        }
        state.setdefault('batches', {})
        state['batches'][str(batch_number)] = record
        _save_batch_runs_state(state)

    return batch_number


def update_batch_run_item(batch_number, image_path, *, status=None, time_s=None, summary=None, error=None,
                          task_id=None, version_dir=None, console=None):
    with batch_runs_lock:
        state = _load_batch_runs_state()
        record = (state.get('batches') or {}).get(str(batch_number))
        if not record:
            return False

        for item in record.get('items', []):
            if item.get('image_path') == image_path:
                if status is not None:
                    item['status'] = status
                if time_s is not None:
                    item['time_s'] = time_s
                if summary is not None:
                    item['summary'] = summary
                if error is not None:
                    item['error'] = error
                if task_id is not None:
                    item['task_id'] = task_id
                if version_dir is not None:
                    item['version_dir'] = version_dir
                if console is not None:
                    item['console'] = console
                break

        state['batches'][str(batch_number)] = record
        _save_batch_runs_state(state)
        return True


def complete_batch_run(batch_number, *, status='completed'):
    with batch_runs_lock:
        state = _load_batch_runs_state()
        record = (state.get('batches') or {}).get(str(batch_number))
        if not record:
            return False
        record['status'] = status
        record['completed_at'] = time.time()
        state['batches'][str(batch_number)] = record
        _save_batch_runs_state(state)
        return True


def compute_batch_aggregates(record):
    """Compute mean metrics, validation fraction, and validation reason counts."""
    items = record.get('items', []) if isinstance(record, dict) else []

    precision_vals = []
    recall_vals = []
    interp_mae_vals = []
    pointwise_x_vals = []
    pointwise_y_vals = []

    validation_yes = 0
    validation_total = 0
    failed = 0

    reasons = []
    for it in items:
        if it.get('status') == 'failed':
            failed += 1
        summary = it.get('summary') or {}

        p = _parse_percent_to_float(summary.get('precision'))
        r = _parse_percent_to_float(summary.get('recall'))
        if p is not None:
            precision_vals.append(p)
        if r is not None:
            recall_vals.append(r)

        if isinstance(summary.get('interpolation_mae'), (int, float)):
            interp_mae_vals.append(float(summary['interpolation_mae']))

        px = _parse_percent_to_float(summary.get('pointwise_mae_x'))
        py = _parse_percent_to_float(summary.get('pointwise_mae_y'))
        if px is not None:
            pointwise_x_vals.append(px)
        if py is not None:
            pointwise_y_vals.append(py)

        vr = summary.get('validation_result')
        if vr in ('Yes', 'No'):
            validation_total += 1
            if vr == 'Yes':
                validation_yes += 1

        reasons.extend(_split_validation_reasons(summary.get('validation_reason')))

    reason_counts = Counter(reasons)

    def _mean(vals):
        return (sum(vals) / len(vals)) if vals else None

    return {
        'count_total': len(items),
        'count_failed': failed,
        'validation_fraction': {
            'successful': validation_yes,
            'total': validation_total,
        },
        'mean': {
            'precision_percent': _mean(precision_vals),
            'recall_percent': _mean(recall_vals),
            'interpolation_mae': _mean(interp_mae_vals),
            'pointwise_mae_x_percent': _mean(pointwise_x_vals),
            'pointwise_mae_y_percent': _mean(pointwise_y_vals),
        },
        'validation_reason_counts': dict(reason_counts.most_common()),
    }


# =============================================================================
# Batch Results Page + APIs
# =============================================================================


@app.route('/batch_results')
def batch_results_page():
    return render_template('batch_results.html')


@app.route('/api/batch_runs')
def api_batch_runs_list():
    with batch_runs_lock:
        state = _load_batch_runs_state()
        batches = state.get('batches') or {}

    rows = []
    for k, rec in batches.items():
        if not isinstance(rec, dict):
            continue
        try:
            bn = int(rec.get('batch_number') or k)
        except Exception:
            continue

        rows.append({
            'batch_number': bn,
            'batch_name': rec.get('batch_name'),
            'status': rec.get('status'),
            'created_at': rec.get('created_at'),
            'completed_at': rec.get('completed_at'),
            'extraction_version': rec.get('extraction_version'),
            'pipeline': rec.get('pipeline'),
            'prompt': rec.get('prompt'),
            'num_images': len(rec.get('items') or []),
            'task_id': rec.get('task_id'),
        })

    rows.sort(key=lambda r: r.get('batch_number', 0), reverse=True)
    resp = jsonify({'batches': rows})
    resp.headers['Cache-Control'] = 'no-store'
    return resp


@app.route('/api/batch_runs/<int:batch_number>')
def api_batch_run_detail(batch_number: int):
    with batch_runs_lock:
        state = _load_batch_runs_state()
        record = (state.get('batches') or {}).get(str(batch_number))

    if not record:
        return jsonify({'error': 'not_found'}), 404

    # Enrich with derived outputs (images, parsed summary) without mutating persisted state.
    # This powers the Batch Results "Details" view, matching the extraction pages.
    record_view = copy.deepcopy(record)
    extraction_version = (record_view.get('extraction_version') or '').lower()
    prompt = record_view.get('prompt')
    for item in record_view.get('items') or []:
        try:
            image_path = item.get('image_path')
            version_dir = item.get('version_dir')
            if not image_path or not version_dir:
                item['outputs'] = {'images': [], 'stats': [], 'data': [], 'summary': item.get('summary') or {}}
                continue

            if extraction_version == 'v2':
                outputs = get_output_files_v2(image_path, prompt, version_dir)
            else:
                outputs = get_output_files(image_path, prompt, version_dir)
            item['outputs'] = outputs
        except Exception as e:
            # Keep the endpoint resilient; still return other items.
            item['outputs'] = {'images': [], 'stats': [], 'data': [], 'summary': item.get('summary') or {}, 'error': str(e)}

    aggregates = compute_batch_aggregates(record)
    resp = jsonify({'record': record_view, 'aggregates': aggregates})
    resp.headers['Cache-Control'] = 'no-store'
    return resp


@app.route('/batch/create', methods=['POST'])
def create_batch_route():
    data = request.get_json(force=True, silent=True) or {}
    extraction_version = data.get('extraction_version') or 'v1'
    prompt = data.get('prompt')
    images = data.get('images') or []
    pipeline = data.get('pipeline')
    batch_name = data.get('batch_name')

    if not prompt or not isinstance(images, list) or not images:
        return jsonify({'error': 'prompt and images are required'}), 400

    try:
        batch_number = create_batch_run_record(
            extraction_version=extraction_version,
            prompt=prompt,
            images=images,
            pipeline=pipeline,
            task_id=None,
            batch_name=batch_name,
        )
    except ValueError:
        return jsonify({'error': 'duplicate batch_name'}), 400
    return jsonify({'batch_number': batch_number})


@app.route('/batch/complete', methods=['POST'])
def complete_batch_route():
    data = request.get_json(force=True, silent=True) or {}
    batch_number = data.get('batch_number')
    status = data.get('status') or 'completed'
    try:
        batch_number = int(batch_number)
    except Exception:
        return jsonify({'error': 'invalid batch_number'}), 400

    ok = complete_batch_run(batch_number, status=status)
    return jsonify({'success': bool(ok)})

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
    'y_min': '0.1',
    'y_max': '6.9',
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


def load_synthetic_settings_v2():
    """Load settings for synthetic generator v2 from file, or return defaults."""
    if os.path.exists(SETTINGS_FILE_V2):
        try:
            with open(SETTINGS_FILE_V2, 'r') as f:
                saved = json.load(f)
                settings = DEFAULT_SETTINGS.copy()
                settings.update(saved)
                return settings
        except Exception as e:
            print(f"Error loading synthetic v2 settings: {e}")
    return DEFAULT_SETTINGS.copy()


def save_synthetic_settings_v2(settings):
    """Save synthetic generator v2 settings to file for persistence."""
    try:
        with open(SETTINGS_FILE_V2, 'w') as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"Error saving synthetic v2 settings: {e}")


# =============================================================================
# Synthetic Generator v2 - Template Curve Library + Generator
# =============================================================================

def categorize_curve(time_points, log10cfu_values):
    """Categorize a curve based on shape characteristics."""
    if not time_points or not log10cfu_values:
        return "UNKNOWN"
    initial = log10cfu_values[0]
    final = log10cfu_values[-1]
    minimum = min(log10cfu_values)
    min_idx = log10cfu_values.index(minimum)

    initial_drop = initial - minimum
    net_change = final - initial
    regrowth = final - minimum if min_idx < len(log10cfu_values) - 2 else 0

    if initial_drop >= 3.0 and regrowth >= 2.0:
        return "KILL_WITH_REGROWTH"
    elif minimum <= initial - 3.0 and final <= initial - 2.5:
        return "KILL"
    elif net_change > 1.5:
        return "GROWTH"
    elif -3.0 < net_change < -1.0:
        return "PARTIAL_KILL"
    elif abs(net_change) <= 1.5:
        return "STABLE"
    else:
        return "BIPHASIC"


# NOTE: These are real-curve templates provided by the user. Values are log10(CFU/mL).
GROWTH_CURVES = [
    {
        "name": "09-95_Control_Rep1",
        "time": [0, 2, 4, 6, 8, 10, 12, 24],
        "log10cfu": [6.023787256, 7.848107281, 9.093292064, 9.693208664, 9.465851167, 10.31393214, 9.772175273, 8.966575533],
    },
    {
        "name": "09-95_Meropenem_4",
        "time": [0, 2, 4, 6, 8, 10, 12, 24],
        "log10cfu": [6.005194894, 7.54886367, 9.458278503, 9.259634554, 9.506539211, 10.30540343, 9.109249252, 10.46024201],
    },
    {
        "name": "09-95_Minocycline_1",
        "time": [0, 2, 4, 6, 8, 10, 12, 24],
        "log10cfu": [6.005194894, 6.78411105, 8.095022219, 8.26878055, 8.07013454, 8.696095638, 8.464197364, 8.644780899],
    },
    {
        "name": "09-95_Rifampicin_0.06",
        "time": [0, 2, 4, 6, 8, 10, 12, 24],
        "log10cfu": [6.005194894, 7.994414337, 8.513972723, 9.033537292, 9.267141867, 9.021943511, 9.328700595, 8.791083855],
    },
    {
        "name": "09-95_Minocycline_2",
        "time": [0, 2, 4, 6, 8, 10, 12, 24],
        "log10cfu": [5.973394308, 5.659583642, 6.408081251, 6.698188753, 6.828223519, 7.031013692, 7.97596174, 8.251322523],
    },
    {
        "name": "09-95_Meropenem_16",
        "time": [0, 2, 4, 6, 8, 10, 12, 24],
        "log10cfu": [5.973394308, 7.25303743, 8.532683016, 8.793688848, 9.505806289, 9.286588987, 9.969597367, 9.997577941],
    },
    {
        "name": "09-2092_Control_Rep1",
        "time": [0, 2, 4, 6, 8, 10, 12, 24],
        "log10cfu": [5.734517673, 7.069078675, 7.973913099, 7.589491325, 8.279475203, 8.341308933, 8.667656448, 9.075731026],
    },
    {
        "name": "09-2092_Colistin_0.125",
        "time": [0, 2, 4, 6, 8, 10, 12, 24],
        "log10cfu": [5.7566548, 5.475408314, 6.46537647, 7.874637538, 8.105870253, 8.29717365, 8.528398113, 8.546305857],
    },
    {
        "name": "09-2092_Meropenem_4",
        "time": [0, 2, 4, 6, 8, 10, 12, 24],
        "log10cfu": [5.7566548, 7.185882591, 7.976191778, 8.027715741, 8.664934347, 8.463554749, 8.501779943, 8.552953179],
    },
    {
        "name": "09-2092_Minocycline_1",
        "time": [0, 2, 4, 6, 8, 10, 12, 24],
        "log10cfu": [5.7566548, 6.473732042, 6.658379865, 7.468643396, 7.706535811, 8.124116347, 7.922745001, 8.093714444],
    },
    {
        "name": "09-2092_Rifampicin_0.06",
        "time": [0, 2, 4, 6, 8, 10, 12, 24],
        "log10cfu": [5.7566548, 7.671743622, 8.11595058, 8.486946224, 8.232317908, 8.350405864, 8.561675983, 7.993880833],
    },
    {
        "name": "09-2092_Control_Rep2",
        "time": [0, 2, 4, 6, 8, 10, 12, 24],
        "log10cfu": [5.775212876, 6.994760539, 7.716654195, 8.061545273, 8.768380036, 8.841829258, 8.900219587, 9.063757048],
    },
    {
        "name": "09-2092_Meropenem_16",
        "time": [0, 2, 4, 6, 8, 10, 12, 24],
        "log10cfu": [5.772520473, 6.849863512, 8.26933576, 8.611464968, 8.342129208, 8.79344859, 8.7788899, 8.844404004],
    },
    {
        "name": "50111_Control",
        "time": [0, 1, 2, 4, 6, 8, 24],
        "log10cfu": [6.457603026, 7.566125085, 8.34258076, 8.991401431, 9.169839167, 9.089868581, 9.308212125],
    },
    {
        "name": "50111_Meropenem_4",
        "time": [0, 1, 2, 4, 6, 8, 24],
        "log10cfu": [6.457603026, 7.427746999, 8.333355555, 8.954480096, 9.040706797, 9.052988269, 9.262086096],
    },
    {
        "name": "AB1845_Control",
        "time": [0, 1, 2, 4, 6, 8, 24],
        "log10cfu": [7.031535253, 7.811866051, 8.332534267, 8.705903665, 8.78247344, 8.831167006, 8.840505926],
    },
    {
        "name": "AB1845_Meropenem_2",
        "time": [0, 1, 2, 4, 6, 8, 24],
        "log10cfu": [7.031535253, 7.774744624, 8.360363624, 8.724519038, 8.819579249, 8.821890553, 8.849751144],
    },
    {
        "name": "AB2092_Control",
        "time": [0, 1, 2, 4, 6, 8, 24],
        "log10cfu": [6.634906393, 7.780039393, 8.429508315, 8.77502328, 8.701121468, 8.856090968, 8.836965788],
    },
    {
        "name": "AB2092_Meropenem_8",
        "time": [0, 1, 2, 4, 6, 8, 24],
        "log10cfu": [6.634906393, 7.169931571, 8.429475964, 8.813155019, 9.091955616, 8.856139496, 8.732119681],
    },
]


KILL_WITH_REGROWTH_CURVES = [
    {
        "name": "09-1769_Colistin_0.25",
        "time": [0, 2, 4, 6, 8, 10, 12, 24],
        "log10cfu": [5.949387868, 4.055988942, 3.789700669, 3.810551141, 4.557224548, 4.80939902, 4.654766919, 8.240274107],
    },
    {
        "name": "09-1769_Minocycline_4",
        "time": [0, 2, 4, 6, 8, 10, 12, 24],
        "log10cfu": [5.854319699, 4.415876293, 4.528668682, 4.175351212, 4.572137665, 5.806459412, 7.652514577, 13.62703761],
    },
    {
        "name": "10-548_Colistin_0.25",
        "time": [0, 2, 4, 6, 8, 10, 12, 24],
        "log10cfu": [5.987236765, 2.982160457, 3.705624987, 4.612731057, 4.881106021, 7.137521645, 8.547603632, 13.85324079],
    },
    {
        "name": "10-548_Minocycline_2",
        "time": [0, 2, 4, 6, 8, 10, 12, 24],
        "log10cfu": [5.959769457, 3.512564503, 3.566053371, 2.979833015, 5.206854731, 7.920952413, 7.8581182, 10.7021051],
    },
    {
        "name": "50111_Colistin_0.25",
        "time": [0, 1, 2, 4, 6, 8, 24],
        "log10cfu": [6.457603026, 3.654678856, 1.958799893, 2.072558628, 1.162360545, 3.62848522, 7.65690029],
    },
    {
        "name": "AB1845_Colistin_0.25",
        "time": [0, 1, 2, 4, 6, 8, 24],
        "log10cfu": [6.37289152, 3.257127954, 2.098789532, 3.297716337, 4.366882128, 5.380295503, 8.608563384],
    },
    {
        "name": "AB2092_Colistin_0.25",
        "time": [0, 1, 2, 4, 6, 8, 24],
        "log10cfu": [6.634906393, 3.509284636, 2.252231316, 3.322119913, 4.354135584, 6.596785438, 8.341350237],
    },
]


KILL_CURVES = [
    {
        "name": "09-95_Col+Mino_0.125+1",
        "time": [0, 2, 4, 6, 8, 10, 12, 24],
        "log10cfu": [6.005194894, 1.0, 1.0, 1.0, 2.191498556, 1.667000551, 1.0, 2.008042335],
    },
    {
        "name": "09-95_Rif+Col_0.06+0.125",
        "time": [0, 2, 4, 6, 8, 10, 12, 24],
        "log10cfu": [6.005194894, 1.789926358, 1.0, 1.532285193, 2.730149792, 1.0, 2.40600822, 1.0],
    },
    {
        "name": "09-95_Mino+Mero_2+16",
        "time": [0, 2, 4, 6, 8, 10, 12, 24],
        "log10cfu": [5.973394308, 4.786469485, 3.112045494, 2.114295077, 1.604036437, 2.010560474, 1.398437365, 1.0],
    },
    {
        "name": "09-1769_Col+Mino_0.25+1",
        "time": [0, 2, 4, 6, 8, 10, 12, 24],
        "log10cfu": [5.949387868, 1.415898818, 1.333060299, 1.545327704, 1.414641622, 1.961913124, 1.599920832, 2.521418563],
    },
    {
        "name": "09-1769_Rif+Col_0.06+0.25",
        "time": [0, 2, 4, 6, 8, 10, 12, 24],
        "log10cfu": [5.949387868, 1.0, 1.205451974, 1.0, 1.0, 1.690726533, 1.0, 3.271169728],
    },
    {
        "name": "09-2092_Mero+Col_4+0.125",
        "time": [0, 2, 4, 6, 8, 10, 12, 24],
        "log10cfu": [5.7566548, 1.18255893, 1.0, 1.877961199, 2.481910187, 1.0, 1.0, 2.948959834],
    },
    {
        "name": "09-2092_Rif+Col_0.06+0.125",
        "time": [0, 2, 4, 6, 8, 10, 12, 24],
        "log10cfu": [5.7566548, 3.671743622, 2.259045422, 1.45200446, 1.0, 1.608313485, 1.513423071, 1.0],
    },
    {
        "name": "09-2092_Minocycline_4",
        "time": [0, 2, 4, 6, 8, 10, 12, 24],
        "log10cfu": [5.772520473, 3.457688808, 3.035486806, 3.326660601, 3.181073703, 3.552320291, 4.192902639, 6.755232029],
    },
    {
        "name": "09-2092_Mino+Mero_4+16",
        "time": [0, 2, 4, 6, 8, 10, 12, 24],
        "log10cfu": [5.772520473, 2.074613285, 1.674249318, 1.950864422, 1.404913558, 1.179253867, 1.0, 1.0],
    },
    {
        "name": "50111_Mero+Col_4+0.25",
        "time": [0, 1, 2, 4, 6, 8, 24],
        "log10cfu": [6.457603026, 2.759731336, 2.512332751, 1.445224123, 1.0, 1.0, 1.12541357],
    },
    {
        "name": "AB1845_Mero+Col_2+0.25",
        "time": [0, 1, 2, 4, 6, 8, 24],
        "log10cfu": [6.623340116, 3.155071362, 1.46795954, 2.843217024, 1.0, 1.0, 1.0],
    },
    {
        "name": "AB2092_Mero+Col_8+0.25",
        "time": [0, 1, 2, 4, 6, 8, 24],
        "log10cfu": [6.634906393, 2.441595947, 1.832733661, 1.939941433, 1.0, 1.649208491, 1.0],
    },
]


PARTIAL_KILL_CURVES = [
    {
        "name": "09-1769_Mino+Mero_4+16",
        "time": [0, 2, 4, 6, 8, 10, 12, 24],
        "log10cfu": [5.854319699, 3.177863877, 3.683900071, 3.002862983, 2.940861707, 3.177439549, 3.698050927, 4.807778037],
    },
    {
        "name": "10-548_Mero+Col_4+0.25",
        "time": [0, 2, 4, 6, 8, 10, 12, 24],
        "log10cfu": [5.987236765, 2.886361034, 2.955108588, 2.816315964, 2.629610268, 2.203365933, 2.200301776, 1.270422374],
    },
    {
        "name": "10-548_Mino+Mero_2+16",
        "time": [0, 2, 4, 6, 8, 10, 12, 24],
        "log10cfu": [5.959769457, 3.243645103, 2.533819932, 2.514589944, 2.488139784, 2.178120564, 2.260699436, 1.266406376],
    },
]


STABLE_CURVES = [
    {
        "name": "A10_Control_Rep1",
        "time": [0, 1, 2, 4, 6, 8, 24],
        "log10cfu": [7.841221887, 8.009950607, 7.900853594, 7.81743634, 7.607058936, 7.80952381, 7.920634921],
    },
    {
        "name": "A10_Meropenem_512_Rep1",
        "time": [0, 1, 2, 4, 6, 8, 24],
        "log10cfu": [7.841221887, 7.509902652, 7.615019422, 7.618999664, 7.480074809, 7.571380617, 7.015849039],
    },
    {
        "name": "A10_Control_Rep2",
        "time": [0, 1, 2, 4, 6, 8, 24],
        "log10cfu": [7.844036697, 8.004587156, 7.900229358, 7.811926606, 7.587155963, 7.79587156, 7.555045872],
    },
    {
        "name": "A10_Meropenem_256",
        "time": [0, 1, 2, 4, 6, 8, 24],
        "log10cfu": [7.651376147, 7.587155963, 7.458715596, 7.153669725, 7.009174312, 6.904816514, 7.779816514],
    },
    {
        "name": "A13_Control_Rep1",
        "time": [0, 2, 4, 6, 8, 24],
        "log10cfu": [7.376, 7.28, 7.424, 7.488, 7.552, 7.856],
    },
    {
        "name": "A13_Meropenem_512_Rep1",
        "time": [0, 2, 4, 6, 8, 24],
        "log10cfu": [7.376, 6.64, 6.456, 6.464, 6.656, 6.984],
    },
    {
        "name": "A13_Control_Rep2",
        "time": [0, 2, 4, 6, 8, 24],
        "log10cfu": [7.143979819, 7.182606899, 7.205283046, 7.307950835, 7.338618792, 7.399938363],
    },
    {
        "name": "A13_Meropenem_256",
        "time": [0, 2, 4, 6, 8, 24],
        "log10cfu": [7.143979819, 6.270674401, 6.405333943, 6.4360019, 6.474653496, 6.863964129],
    },
]


CURVE_TEMPLATES = {
    "GROWTH": GROWTH_CURVES,
    "KILL_WITH_REGROWTH": KILL_WITH_REGROWTH_CURVES,
    "KILL": KILL_CURVES,
    "PARTIAL_KILL": PARTIAL_KILL_CURVES,
    "STABLE": STABLE_CURVES,
}


SYNTHETIC_V2_MAX_LOG10CFU = 10.0


def _filter_templates_by_max_y(templates, max_y=SYNTHETIC_V2_MAX_LOG10CFU):
    filtered = []
    for template in templates or []:
        ys = template.get('log10cfu') if isinstance(template, dict) else None
        if not ys:
            continue
        try:
            y_max = max(float(v) for v in ys)
        except Exception:
            continue
        if y_max <= float(max_y):
            filtered.append(template)
    return filtered


# Remove any real-template curves that exceed the allowed log10(CFU/mL) maximum.
RAW_CURVE_TEMPLATES = CURVE_TEMPLATES
CURVE_TEMPLATES = {
    category: _filter_templates_by_max_y(templates)
    for category, templates in (RAW_CURVE_TEMPLATES or {}).items()
}


def _trend_to_template_category(trend_key):
    trend_key = (trend_key or '').strip().lower()
    if trend_key == 'up':
        return 'GROWTH'
    if trend_key == 'down':
        return 'KILL'
    if trend_key == 'kill_regrowth':
        return 'KILL_WITH_REGROWTH'
    if trend_key == 'mixed':
        # Mixed isn't explicitly provided; pick a diverse but reasonable default.
        return 'PARTIAL_KILL'
    return 'STABLE'


def generate_curve_data_v2(x_values, curve_config, y_scale='log'):
    """Generate Y-values using real-template curves + proportional noise.

    - Picks a template category based on the curve's `trend`
    - Interpolates template to requested `x_values`
    - Aligns starting value to `initial_y`
    - Scales deviations by `trend_magnitude`
    - Adds proportional noise (per user's spec) and clamps to >= 1.0
    """
    import random

    x_arr = np.asarray(x_values, dtype=float)

    category = _trend_to_template_category(curve_config.get('trend'))
    templates = CURVE_TEMPLATES.get(category) or CURVE_TEMPLATES.get('STABLE') or []
    if not templates:
        # If template library was fully filtered out, fall back to the original generator.
        y_fallback = generate_curve_data(x_arr, curve_config, y_scale=y_scale)
        if str(y_scale).lower() == 'log':
            return np.minimum(np.asarray(y_fallback, dtype=float), SYNTHETIC_V2_MAX_LOG10CFU)
        return y_fallback
    template = random.choice(templates)

    t = np.asarray(template['time'], dtype=float)
    y_t = np.asarray(template['log10cfu'], dtype=float)
    if len(t) == 0 or len(y_t) == 0:
        # Fallback to original generator behavior if template is malformed
        return generate_curve_data(x_arr, curve_config, y_scale=y_scale)

    # Interpolate template onto desired x grid
    y_interp = np.interp(x_arr, t, y_t, left=float(y_t[0]), right=float(y_t[-1]))

    initial_y = float(curve_config.get('initial_y', 6.0))
    magnitude = float(curve_config.get('trend_magnitude', 1.0) or 1.0)

    # Align template start to requested initial_y, then scale deviations
    y0 = float(y_interp[0])
    y_base = initial_y + (y_interp - y0) * magnitude

    noise_level = float(curve_config.get('noise_level', 0.1) or 0.0)
    if noise_level < 0:
        noise_level = 0.0

    # Proportional noise per user's method
    noise_magnitude = np.where(
        y_base <= 2.0,
        0.3 * noise_level,
        np.abs(y_base) * 0.1 * noise_level,
    )
    noise = np.random.normal(0.0, noise_magnitude)
    y_noisy = y_base + noise
    # Detection limit clamp + enforce max range for v2 examples
    y_noisy = np.maximum(y_noisy, 1.0)
    y_noisy = np.minimum(y_noisy, SYNTHETIC_V2_MAX_LOG10CFU)

    if str(y_scale).lower() == 'log':
        return y_noisy
    # Convert log10(CFU/mL) -> CFU/mL
    return np.power(10.0, y_noisy)


def generate_all_curves_v2(settings):
    """Generate data for all curves using the v2 generator."""
    x_values = generate_x_values(settings)
    curves_data = []
    for curve_config in settings['curves']:
        y_values = generate_curve_data_v2(x_values, curve_config, settings.get('y_scale', 'log'))
        curves_data.append({
            'x': x_values.tolist(),
            'y': y_values.tolist(),
            'config': curve_config
        })
    return x_values.tolist(), curves_data

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
    """Generate Y-values for a single curve based on its configuration.
    
    Implements realistic time-kill curve dynamics:
    - Stable: Control growth (0.05-0.25 log10 CFU/mL/hr)
    - Down: Kill curves with realistic slopes (-0.5 to -2 log10 CFU/mL/hr)
    - Up: Growth without killing
    - Mixed: Rise and fall within realistic CFU ranges
    - Kill_regrowth: Kill phase followed by bacterial regrowth
    """
    n_points = len(x_values)
    initial_y = curve_config['initial_y']
    trend = curve_config['trend']
    magnitude = curve_config['trend_magnitude']
    noise = curve_config['noise_level']
    
    # Normalize x to [0, 1] for relative positioning
    x_norm = (x_values - x_values.min()) / (x_values.max() - x_values.min() + 1e-10)
    # Calculate actual time span in hours (for realistic slope calculation)
    time_span = x_values.max() - x_values.min()
    
    if trend == 'stable':
        # Control/stable growth: realistic linear growth phase
        # Slope: 0.05-0.25 log10 CFU/mL per hour (varies with magnitude)
        # magnitude acts as a multiplier: 0.5->0.075/hr, 1.0->0.15/hr, 2.0->0.30/hr (capped)
        slope_per_hour = 0.05 + (magnitude * 0.1)
        slope_per_hour = min(slope_per_hour, 0.25)  # Cap at realistic max
        base = slope_per_hour * (x_values - x_values.min())
        # Add small random walk for realism (minor fluctuations)
        random_walk = np.zeros(n_points)
        for i in range(1, n_points):
            random_walk[i] = random_walk[i-1] + np.random.normal(0, 0.02)
        base = base + random_walk
    
    elif trend == 'down':
        # Kill curve: realistic decline phase
        # Slope: -0.5 to -2 log10 CFU/mL per hour
        # magnitude: 0.5 -> -0.75/hr, 1.0 -> -1.25/hr, 2.0 -> -2.0/hr
        kill_slope = -(0.5 + magnitude * 0.75)
        kill_slope = max(kill_slope, -2.0)  # Cap at realistic max killing rate
        base = kill_slope * (x_values - x_values.min())
        # Add slight curvature to slow down at very low CFU (realistic antibiotic dynamics)
        base = base * (1 + 0.15 * np.sin(x_norm * np.pi * 0.5))
    
    elif trend == 'up':
        # Growth without killing: exponential-like approach to upper limit
        # Maximum plausible ~8-10 log10 CFU/mL
        # Use exponential saturation curve
        saturation_value = 2.0 + magnitude * 1.0  # Total growth potential
        saturation_value = min(saturation_value, 3.5)  # Realistic max gain
        growth_rate = 0.3 + magnitude * 0.3
        base = saturation_value * (1 - np.exp(-growth_rate * x_norm * 4))
    
    elif trend == 'mixed':
        # Rise and fall within realistic CFU ranges
        # Peak typically at mid-timeline, max ~12-13 log10 CFU/mL
        peak_pos = np.random.uniform(0.3, 0.7)
        peak_height = 1.5 + magnitude * 0.8  # Peak relative to initial
        peak_height = min(peak_height, 4.0)  # Realistic max
        base = np.zeros(n_points)
        for i, xn in enumerate(x_norm):
            if xn < peak_pos:
                # Rise to peak
                base[i] = peak_height * (xn / peak_pos) ** 0.9
            else:
                # Fall from peak (steeper than rise for realism)
                base[i] = peak_height * (1 - (xn - peak_pos) / (1 - peak_pos)) ** 0.8
    
    elif trend == 'kill_regrowth':
        # Kill phase followed by regrowth
        # Nadir: typically 0-2 log10 CFU/mL (avoid negative)
        # Regrowth slope: 0.1-0.5 log10 CFU/mL per hour
        nadir_pos = np.random.uniform(0.25, 0.45)
        nadir_depth = 2.5 + magnitude * 1.0  # How far down from initial
        nadir_depth = min(nadir_depth, 4.5)  # Realistic max killing
        
        regrowth_rate = 0.15 + magnitude * 0.2  # Regrowth slope multiplier
        regrowth_rate = min(regrowth_rate, 0.5)
        
        base = np.zeros(n_points)
        for i, xn in enumerate(x_norm):
            if xn < nadir_pos:
                # Kill phase: curved decline (slower at low CFU)
                base[i] = -nadir_depth * (xn / nadir_pos) ** 0.7
            else:
                # Regrowth phase: exponential regrowth from nadir
                regrowth_x = (xn - nadir_pos) / (1 - nadir_pos)
                # Realistic regrowth with diminishing slope as CFU increases
                regrowth_amount = regrowth_rate * nadir_depth * (1 - np.exp(-2.0 * regrowth_x))
                base[i] = -nadir_depth + regrowth_amount
    else:
        base = np.zeros(n_points)
    
    # Generate noise: Gaussian, realistic for microbiology assays
    noise_vals = np.random.normal(0, noise, n_points)
    
    if y_scale == 'log':
        y_values = initial_y + base + noise_vals
        # Enforce realistic CFU range: 0.1 to ~12-13 log10 CFU/mL
        y_values = np.maximum(y_values, 0.1)  # Avoid negative/unrealistic low values
        y_values = np.minimum(y_values, 13.0)  # Cap at extreme but plausible max
    else:
        y_values = (10 ** initial_y) * (10 ** (base + noise_vals))
        y_values = np.maximum(y_values, 0.1)
        y_values = np.minimum(y_values, 10 ** 13)
    
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
    # Add log10 scale indication if y_scale is log
    if settings.get('y_scale', '').lower() == 'log':
        y_label += " (log10 scale)"
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
        all_y = [val for curve in curves_data for val in curve.get('y', [])]
        if settings['y_min'] != '':
            try:
                y_min = float(settings['y_min'])
            except:
                y_min = 0.1
        else:
            # Auto y-min (log10-space) based on data
            if all_y:
                y_min = max(0.1, float(min(all_y)) - 0.5)
            else:
                y_min = 0.1
        
        if settings['y_max'] != '':
            try:
                y_max = float(settings['y_max'])
            except:
                y_max = 6.9
        else:
            # Auto y-max (log10-space) based on data
            if all_y:
                y_max = float(math.ceil(max(all_y)) + 1)
            else:
                y_max = 6.9

        # Keep sane bounds for log10(CFU/mL) synthetic plots
        y_min = max(0.1, float(y_min))
        y_max = min(13.0, float(y_max))
        if y_max <= y_min:
            y_max = min(13.0, y_min + 1.0)
        
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
        # Add minor ticks at every 1 unit
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.tick_params(axis='y', which='minor', left=True, right=False)
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
                # Create broken x-axis using two subplots side by side
                fig.clf()

                # Create two subplots with width proportional to data range
                fig, (ax1, ax2) = plt.subplots(
                    1, 2,
                    sharey=True,
                    figsize=(settings['figure_width'], settings['figure_height']),
                    gridspec_kw={
                        'width_ratios': [break_start - x_min, x_max - break_end],
                        'wspace': 0.05
                    }
                )

                # Plot all curves on both axes
                for curve_data in curves_data:
                    config = curve_data['config']
                    x = np.array(curve_data['x'])
                    y = np.array(curve_data['y'])

                    # Prepare plot kwargs
                    plot_kwargs = {
                        'color': config['color'],
                        'linewidth': config['line_width'],
                        'markersize': config['marker_size'],
                        'label': config['name']
                    }

                    # Handle marker edge color for black markers
                    if config['color'].lower() in ['#000000', '#000', 'black']:
                        plot_kwargs['markeredgecolor'] = 'none'

                    if config['show_line']:
                        plot_kwargs['linestyle'] = config['line_style']
                        plot_kwargs['marker'] = config['marker']
                    else:
                        plot_kwargs['linestyle'] = 'none'
                        plot_kwargs['marker'] = config['marker']

                    # Plot on left axis (data before break)
                    mask_left = x <= break_start
                    if mask_left.any():
                        ax1.plot(x[mask_left], y[mask_left], **plot_kwargs)

                    # Plot on right axis (data after break) - no label to avoid duplicate
                    plot_kwargs_right = plot_kwargs.copy()
                    plot_kwargs_right['label'] = None
                    mask_right = x >= break_end
                    if mask_right.any():
                        ax2.plot(x[mask_right], y[mask_right], **plot_kwargs_right)

                    # Draw connecting line across the break if show_line is True
                    if config['show_line'] and mask_left.any() and mask_right.any():
                        x_before = x[mask_left]
                        y_before = y[mask_left]
                        x_after = x[mask_right]
                        y_after = y[mask_right]

                        # Last point before gap, first point after gap
                        x0, y0 = x_before[-1], y_before[-1]
                        x1, y1 = x_after[0], y_after[0]

                        actual_gap = x1 - x0
                        if actual_gap != 0:
                            # Visual gap is compressed, so slopes should reflect that.
                            visual_gap = 1.5
                            slope = (y1 - y0) / visual_gap

                            left_segment_x = break_start - x0
                            left_fraction = left_segment_x / actual_gap
                            left_segment_y = slope * (visual_gap * left_fraction)

                            right_segment_x = x1 - break_end
                            right_fraction = right_segment_x / actual_gap
                            right_segment_y = slope * (visual_gap * right_fraction)

                            y_at_break_left = y0 + left_segment_y
                            ax1.plot(
                                [x0, break_start], [y0, y_at_break_left],
                                color=config['color'],
                                linestyle=config['line_style'],
                                linewidth=config['line_width'],
                                alpha=0.7,
                            )

                            y_at_break_right = y1 - right_segment_y
                            ax2.plot(
                                [break_end, x1], [y_at_break_right, y1],
                                color=config['color'],
                                linestyle=config['line_style'],
                                linewidth=config['line_width'],
                                alpha=0.7,
                            )

                # Set axis limits
                ax1.set_xlim(x_min, break_start)
                ax2.set_xlim(break_end, x_max)
                ax1.set_ylim(y_min, y_max)
                ax2.set_ylim(y_min, y_max)

                # Hide the spines between the axes and on top
                ax1.spines['right'].set_visible(False)
                ax2.spines['left'].set_visible(False)
                ax1.spines['top'].set_visible(False)
                ax2.spines['top'].set_visible(False)

                # Only show y-axis on the left
                ax2.yaxis.set_visible(False)
                ax1.yaxis.tick_left()

                # Only show x-axis ticks on bottom
                ax1.xaxis.set_ticks_position('bottom')
                ax2.xaxis.set_ticks_position('bottom')
                ax1.tick_params(top=False)
                ax2.tick_params(top=False)

                # Add diagonal lines to indicate break (ONLY on bottom)
                d = 0.015
                kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linewidth=1)
                ax1.plot((1-d, 1+d), (-d, +d), **kwargs)

                kwargs2 = dict(transform=ax2.transAxes, color='k', clip_on=False, linewidth=1)
                ax2.plot((-d, +d), (-d, +d), **kwargs2)

                # Set x-axis ticks
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

                    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))
                    ax1.tick_params(axis='x', which='minor', bottom=True, top=False)
                    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(1))
                    ax2.tick_params(axis='x', which='minor', bottom=True, top=False)

                # Add minor ticks for y-axis
                if settings.get('y_scale', '').lower() == 'log':
                    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1))
                    ax1.tick_params(axis='y', which='minor', left=True, right=False)

                # Set labels
                x_label = settings['x_label']
                if settings['x_unit']:
                    x_label += f" ({settings['x_unit']})"
                fig.text(0.5, 0.02, x_label, ha='center', fontsize=11)

                y_label = settings['y_label']
                if settings['y_unit']:
                    y_label += f" ({settings['y_unit']})"
                if settings.get('y_scale', '').lower() == 'log':
                    y_label += " (log10 scale)"
                ax1.set_ylabel(y_label, fontsize=11)

                # Title
                if settings['title']:
                    fig.suptitle(settings['title'], fontsize=12, fontweight='bold')

                # Legend (only on left axis to avoid duplicates)
                if settings['show_legend']:
                    ax1.legend(loc='best', framealpha=0.9)

                # Grid (on both axes)
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
            # Add minor ticks at every 1 unit
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            ax.tick_params(axis='x', which='minor', bottom=True, top=False)
    # Otherwise, use auto ticks (matplotlib default)
    
    if settings['title']:
        ax.set_title(settings['title'], fontsize=12, fontweight='bold')
    
    if settings['show_legend']:
        from matplotlib.lines import Line2D
        legend_handles = []
        for curve_data in curves_data:
            config = curve_data.get('config') or {}
            color = config.get('color', '#000000')
            marker = config.get('marker', 'o')
            marker_size = config.get('marker_size', 6)
            line_style = config.get('line_style', '-')
            line_width = config.get('line_width', 1.5)
            show_line = bool(config.get('show_line', True))
            name = config.get('name', 'Condition')

            handle = Line2D(
                [0], [0],
                color=color,
                linestyle=line_style if show_line else 'none',
                linewidth=line_width,
                marker=marker,
                markersize=marker_size,
                markerfacecolor=color,
                markeredgecolor='none' if str(color).lower() in ['#000000', '#000', 'black'] else color,
                label=name,
            )
            legend_handles.append(handle)
        ax.legend(handles=legend_handles, loc='best', framealpha=0.9)
    
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


def build_synthetic_context(settings, curves_data, x_values, base_name):
    """Build structured context metadata for a synthetic time-kill plot."""
    def map_line_style(style):
        return {
            '-': 'solid',
            '--': 'dashed',
            '-.': 'dash-dot',
            ':': 'dotted'
        }.get(style, 'custom')

    def map_marker(marker):
        return {
            'o': 'circle',
            's': 'square',
            '^': 'triangle_up',
            'v': 'triangle_down',
            'D': 'diamond',
            'x': 'x',
            '+': 'plus',
            None: 'none',
            '': 'none',
            'none': 'none',
            'None': 'none'
        }.get(marker, marker or 'none')

    def map_trend(trend_key):
        trend_map = {
            'stable': 'stable (flat)',
            'up': 'growth (increasing)',
            'down': 'kill (decreasing)',
            'kill_regrowth': 'kill + regrowth',
            'mixed': 'mixed'
        }
        return trend_map.get(trend_key, trend_key or 'unspecified')

    # Axis ranges derived from generator settings/data (not from images)
    try:
        x_min_val = max(0, float(settings.get('x_min', 0))) if str(settings.get('x_min', '')).strip() != '' else 0.0
    except Exception:
        x_min_val = 0.0
    try:
        if str(settings.get('x_max', '')).strip() != '':
            x_max_val = float(settings['x_max'])
        else:
            x_max_val = float(max(x_values)) if len(x_values) > 0 else 24.0
    except Exception:
        x_max_val = float(max(x_values)) if len(x_values) > 0 else 24.0

    all_y = [val for curve in curves_data for val in curve.get('y', [])]
    if str(settings.get('y_scale', '')).lower() == 'log':
        try:
            if str(settings.get('y_min', '')).strip() != '':
                y_min_val = float(settings['y_min'])
            else:
                y_min_val = max(0.1, float(min(all_y)) - 0.5) if all_y else 0.1
        except Exception:
            y_min_val = 0.1
        try:
            if str(settings.get('y_max', '')).strip() != '':
                y_max_val = float(settings['y_max'])
            else:
                y_max_val = float(math.ceil(max(all_y)) + 1) if all_y else 6.9
        except Exception:
            y_max_val = 6.9

        y_min_val = max(0.1, float(y_min_val))
        y_max_val = min(13.0, float(y_max_val))
        if y_max_val <= y_min_val:
            y_max_val = min(13.0, y_min_val + 1.0)
    else:
        try:
            if str(settings.get('y_min', '')).strip() != '':
                y_min_val = float(settings['y_min'])
            else:
                y_min_val = float(min(all_y)) if all_y else 0.0
        except Exception:
            y_min_val = float(min(all_y)) if all_y else 0.0
        try:
            if str(settings.get('y_max', '')).strip() != '':
                y_max_val = float(settings['y_max'])
            else:
                y_max_val = float(max(all_y)) if all_y else 10.0
        except Exception:
            y_max_val = float(max(all_y)) if all_y else 10.0

    legend_curves = []
    trend_entries = []
    for idx, curve in enumerate(curves_data):
        config = curve.get('config', {})
        name = config.get('name') or f"Curve {idx + 1}"
        color = config.get('color', '')
        marker = map_marker(config.get('marker'))
        line_type = map_line_style(config.get('line_style', '-'))
        trend_key = config.get('trend') or curve.get('trend')
        legend_curves.append({
            'id': name,
            'color': color,
            'marker': marker,
            'line_type': line_type
        })
        trend_entries.append({
            'id': name,
            'trend': trend_key or 'unspecified',
            'description': map_trend(trend_key)
        })

    return {
        'plot_type': 'time-kill plot',
        'source': 'synthetic_generator',
        'name': base_name,
        'legend': {
            'total_curves': len(curves_data),
            'curves': legend_curves
        },
        'axes': {
            'x': {
                'label': settings.get('x_label', ''),
                'units': settings.get('x_unit', ''),
                'range': {'min': x_min_val, 'max': x_max_val}
            },
            'y': {
                'label': settings.get('y_label', ''),
                'units': settings.get('y_unit', ''),
                'range': {'min': y_min_val, 'max': y_max_val}
            }
        },
        'curves_trends': trend_entries
    }

def find_synthetic_plot_folder(plot_name):
    """Find a synthetic plot folder by name, supporting nested letter folders.
    
    Looks for plot_name in SYNTHETIC_DIR/[A-Z]/plot_name/ or SYNTHETIC_DIR/plot_name/"""
    # Try nested structure first
    if os.path.exists(SYNTHETIC_DIR):
        first_letter = plot_name[0].upper()
        letter_folder = os.path.join(SYNTHETIC_DIR, first_letter)
        nested_path = os.path.join(letter_folder, plot_name)
        if os.path.isdir(nested_path):
            return nested_path
    
    # Fall back to flat structure for backwards compatibility
    flat_path = os.path.join(SYNTHETIC_DIR, plot_name)
    if os.path.isdir(flat_path):
        return flat_path
    
    return None

def get_next_synthetic_name():
    """Generate the next available name for a synthetic plot (AA, AB, AC, ...).
    
    Looks in nested letter folders (A/, B/, C/, etc.) for existing plots."""
    import string
    existing = set()
    if os.path.exists(SYNTHETIC_DIR):
        # Look in nested letter folders (A/, B/, C/, D/, ...)
        for letter_folder in os.listdir(SYNTHETIC_DIR):
            letter_path = os.path.join(SYNTHETIC_DIR, letter_folder)
            if os.path.isdir(letter_path) and len(letter_folder) == 1 and letter_folder.isalpha():
                # This is a letter folder, scan inside it
                for plot_folder in os.listdir(letter_path):
                    if os.path.isdir(os.path.join(letter_path, plot_folder)):
                        existing.add(plot_folder.upper())
    
    for first in string.ascii_uppercase:
        for second in string.ascii_uppercase:
            name = f"{first}{second}"
            if name not in existing:
                return name
    
    from datetime import datetime
    return f"ZZ_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def save_synthetic_plot_and_data(settings, curves_data, x_values):
    """Save the plot as PNG and the data as CSV into the appropriate letter subfolder."""
    base_filename = get_next_synthetic_name()
    first_letter = base_filename[0].upper()
    letter_folder = os.path.join(SYNTHETIC_DIR, first_letter)
    plot_folder = os.path.join(letter_folder, base_filename)
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

    # Save context metadata alongside the image
    context_path = os.path.join(plot_folder, f'{base_filename}.context.json')
    context_payload = build_synthetic_context(settings, curves_data, x_values, base_filename)
    with open(context_path, 'w', encoding='utf-8') as f:
        json.dump(context_payload, f, indent=2)
    
    return {
        'png': png_path,
        'svg': svg_path,
        'csv': csv_path,
        'context': context_path,
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

@app.route('/v2/get_prompts')
def get_prompts_route_v2():
    return jsonify({'prompts': get_prompts_v2()})

def get_prompts_v2():
    """Get all v2 prompt sets (prompt_1, prompt_2, etc.) from the v2 prompts directory."""
    prompts = []
    prompts_dir = os.path.join(BASE_DIR, 'plot_extract_v2', 'prompts')
    if os.path.exists(prompts_dir):
        for item in os.listdir(prompts_dir):
            item_path = os.path.join(prompts_dir, item)
            # Check if it's a directory and has a prompts.py file
            if os.path.isdir(item_path) and not item.startswith('__'):
                prompts_file = os.path.join(item_path, 'prompts.py')
                if os.path.exists(prompts_file):
                    prompts.append(item)
    # Sort numerically (prompt_1, prompt_2, etc.)
    prompts.sort(key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0)
    return prompts

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
    version_pattern_new = re.compile(rf'^{re.escape(name_for_folder)}\.{re.escape(prompt_name)}\.v(\d+)(?:\.(web))?$')
    version_pattern_old = re.compile(rf'^{re.escape(base_name)}\.{re.escape(prompt_name)}\.v(\d+)(?:\.(web))?$')
    
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
                tag_suffix = f".{match.group(2)}" if match.lastindex and match.lastindex >= 2 and match.group(2) else ""
                extracted_file = os.path.join(version_dir, f"{image_name}.{prompt_name}.v{version_num}{tag_suffix}.mistral.out_data")
                
                if os.path.exists(extracted_file) and version_num > latest_version:
                    latest_version = version_num
                    latest_file = extracted_file
    
    if latest_file:
        return os.path.relpath(latest_file, PLOTS_DIR).replace('\\', '/')
    return None

def _get_prompt_name_v2(chain_file):
    """Convert chain filename to v2 prompt identifier."""
    chain_short = os.path.splitext(chain_file)[0]
    return f"pv2_{chain_short}"

def find_extracted_csv_v2(image_path, prompt_name):
    """Find extracted data file for v2 prompt set outputs."""
    import re

    image_dir = os.path.dirname(os.path.join(PLOTS_DIR, image_path))
    image_name = os.path.basename(image_path)
    base_name = os.path.splitext(image_name)[0]
    name_for_folder = image_name.replace('.', '_')
    full_prompt_name = f"pv2_{prompt_name}"

    version_pattern_new = re.compile(rf'^{re.escape(name_for_folder)}\.{re.escape(full_prompt_name)}\.v(\d+)(?:\.key(\d+))?(?:\.(web))?$')
    version_pattern_old = re.compile(rf'^{re.escape(base_name)}\.{re.escape(full_prompt_name)}\.v(\d+)(?:\.key(\d+))?(?:\.(web))?$')

    latest_version = 0
    latest_file = None

    if os.path.exists(image_dir):
        for item in os.listdir(image_dir):
            match = version_pattern_new.match(item) or version_pattern_old.match(item)
            if match:
                version_num = int(match.group(1))
                key_idx = match.group(2) if match.lastindex and match.lastindex >= 2 else None
                web_tag = match.group(3) if match.lastindex and match.lastindex >= 3 else None
                version_dir = os.path.join(image_dir, item)
                key_suffix = f".key{key_idx}" if key_idx else ""
                web_suffix = f".{web_tag}" if web_tag else ""
                extracted_file = os.path.join(version_dir, f"{image_name}.{full_prompt_name}.v{version_num}{key_suffix}{web_suffix}.mistral.out_data")
                if os.path.exists(extracted_file) and version_num > latest_version:
                    latest_version = version_num
                    latest_file = extracted_file

    if latest_file:
        return os.path.relpath(latest_file, PLOTS_DIR).replace('\\', '/')
    return None

def get_output_files_v2(image_path, prompt_name=None, version_dir=None):
    """Get output files for PlotExtractV2 runs."""
    if version_dir:
        version_dir = _resolve_path_under_plots(version_dir)
    image_dir = os.path.dirname(os.path.join(PLOTS_DIR, image_path))
    image_name = os.path.basename(image_path)
    base_name = os.path.splitext(image_name)[0]
    full_prompt_name = f"pv2_{prompt_name}" if prompt_name else None

    outputs = {
        'images': [],
        'stats': [],
        'data': [],
        'other': [],
        'summary': {}
    }

    if not os.path.exists(image_dir):
        return outputs

    original_path = os.path.join(image_dir, image_name)
    if os.path.exists(original_path):
        outputs['images'].append({
            'path': os.path.relpath(original_path, PLOTS_DIR).replace('\\', '/'),
            'label': 'Original Input',
            'filename': image_name
        })

    original_csv = os.path.join(image_dir, f"{base_name}-original.csv")
    if os.path.exists(original_csv):
        outputs['data'].append({
            'path': os.path.relpath(original_csv, PLOTS_DIR).replace('\\', '/'),
            'label': 'Original Data',
            'filename': f"{base_name}-original.csv"
        })

    if version_dir and os.path.exists(version_dir):
        version_label = os.path.basename(version_dir)
        _scan_version_folder(version_dir, version_label, outputs, PLOTS_DIR)
        outputs['summary'] = _parse_summary_stats(version_dir)
    else:
        import re
        name_for_folder = image_name.replace('.', '_')
        if full_prompt_name:
            version_pattern_new = re.compile(rf'^{re.escape(name_for_folder)}\.{re.escape(full_prompt_name)}\.v\d+(?:\.key\d+)?(?:\.web)?$')
            version_pattern_old = re.compile(rf'^{re.escape(base_name)}\.{re.escape(full_prompt_name)}\.v\d+(?:\.key\d+)?(?:\.web)?$')
        else:
            version_pattern_new = re.compile(r'^$a')
            version_pattern_old = re.compile(r'^$a')

        for item in os.listdir(image_dir):
            item_path = os.path.join(image_dir, item)
            if os.path.isdir(item_path) and (version_pattern_new.match(item) or version_pattern_old.match(item)):
                version_label = os.path.basename(item_path)
                _scan_version_folder(item_path, version_label, outputs, PLOTS_DIR)

    return outputs

def get_output_files(image_path, prompt_file=None, version_dir=None):
    """Get output files related to an image. If version_dir is provided, only show that version."""
    if version_dir:
        version_dir = _resolve_path_under_plots(version_dir)
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
        version_pattern_new = re.compile(rf'^{re.escape(name_for_folder)}\.p\d+\.v\d+(?:\.web)?$')
        version_pattern_old = re.compile(rf'^{re.escape(base_name)}\.p\d+\.v\d+(?:\.web)?$')
        
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
            # Keep overlay variants distinct so the UI can reliably pick the normal replot.
            if '-replot_overlay_minmax' in f:
                label = f'Replot Overlay (min/max axes) ({version_label})'
            elif '-replot_overlay_full' in f:
                label = f'Replot Overlay (full axes) ({version_label})'
            elif '-replot_overlay' in f:
                label = f'Replot Overlay ({version_label})'
            elif '-replot' in f:
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

        elif f.endswith('.csv'):
            # Many pipelines persist extracted results as CSV in the version folder.
            label = f'Extracted CSV ({version_label})'
            if f.endswith('-original.csv'):
                label = f'Original Data ({version_label})'
            outputs['data'].append({'path': rel_path, 'label': label, 'filename': f})
            
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

def check_csv_exists_v2(image_path, prompt_name=None):
    """Check if original and v2 extracted CSVs exist."""
    csv_info = get_csv_paths(image_path)

    original_full = os.path.join(PLOTS_DIR, csv_info['original'])
    original_exists = os.path.exists(original_full)

    extracted_exists = False
    extracted_path = None

    if prompt_name:
        extracted_rel = find_extracted_csv_v2(image_path, prompt_name)
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
    ex1_prompts = get_prompts()
    ex2_prompts = get_prompts_v2()
    return render_template('index.html', ex1_prompts=ex1_prompts, ex2_prompts=ex2_prompts)

@app.route('/v2')
def index_v2():
    return redirect('/')

@app.route('/plots/<path:filename>')
def serve_plot(filename):
    """Serve files from the plots directory."""
    return send_from_directory(PLOTS_DIR, filename)


# =============================================================================
# Minimal User-Facing UI Endpoints
# =============================================================================


@app.route('/ui/prompts')
def ui_prompts():
    """Return available prompts for the unified UI."""
    return jsonify({
        'ex1': get_prompts(),
        'ex2': get_prompts_v2(),
    })


def _iter_example_files():
    if not os.path.isdir(UI_EXAMPLES_DIR):
        return
    for root, _dirs, files in os.walk(UI_EXAMPLES_DIR):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in ALLOWED_IMAGE_EXTS:
                continue
            full_path = os.path.join(root, name)
            rel_path = os.path.relpath(full_path, UI_EXAMPLES_DIR).replace('\\', '/')
            yield rel_path


@app.route('/ui/examples')
def ui_examples():
    """List example images from WebExtract/examples."""
    return jsonify({'examples': sorted(set(_iter_example_files()))})


@app.route('/ui/examples/<path:filename>')
def ui_examples_file(filename):
    """Serve a UI example image for preview."""
    return send_from_directory(UI_EXAMPLES_DIR, filename)


def _is_allowed_image(filename: str) -> bool:
    ext = os.path.splitext(filename or '')[1].lower()
    return ext in ALLOWED_IMAGE_EXTS


@app.route('/ui/upload', methods=['POST'])
def ui_upload():
    """Upload an image and store it under Prototype plots/batch_uploads for extraction."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'missing_file'}), 400

    file = request.files['file']
    if not file or not file.filename:
        return jsonify({'success': False, 'error': 'empty_filename'}), 400

    filename = secure_filename(file.filename)
    if not _is_allowed_image(filename):
        return jsonify({'success': False, 'error': 'unsupported_file_type'}), 400

    token = str(uuid.uuid4())[:8]
    final_name = f"ui_{token}_{filename}"

    os.makedirs(BATCH_DIR, exist_ok=True)
    dest_full = os.path.join(BATCH_DIR, final_name)
    file.save(dest_full)

    rel_image_path = os.path.relpath(dest_full, PLOTS_DIR).replace('\\', '/')
    return jsonify({
        'success': True,
        'image_path': rel_image_path,
        'preview_url': f"/plots/{rel_image_path}",
        'filename': final_name,
    })


@app.route('/ui/prepare_example', methods=['POST'])
def ui_prepare_example():
    """Validate a WebExtract/examples image and resolve it to canonical plots path."""
    data = request.json or {}
    example_path = (data.get('example_path') or '').strip().replace('\\', '/')
    if not example_path:
        return jsonify({'success': False, 'error': 'missing_example_path'}), 400

    src_full = _safe_abs_under_ui_examples(example_path)
    if not src_full:
        return jsonify({'success': False, 'error': 'invalid_example_path'}), 400

    if not os.path.isfile(src_full):
        return jsonify({'success': False, 'error': 'example_not_found'}), 404

    if not _is_allowed_image(src_full):
        return jsonify({'success': False, 'error': 'unsupported_file_type'}), 400

    rel_image_path = _resolve_ui_example_to_plots_rel(example_path)
    if not rel_image_path:
        return jsonify({'success': False, 'error': 'example_not_mapped_in_plots'}), 400

    return jsonify({
        'success': True,
        'image_path': rel_image_path,
        'preview_url': f"/plots/{rel_image_path}",
        'filename': os.path.basename(src_full),
    })

@app.route('/check_csv', methods=['POST'])
def check_csv():
    """Check if CSV files exist for the selected image."""
    image_path = request.json.get('image_path')
    prompt_file = request.json.get('prompt_file')
    
    result = check_csv_exists(image_path, prompt_file)
    return jsonify(result)

@app.route('/v2/check_csv', methods=['POST'])
def check_csv_v2():
    """Check CSV existence for PlotExtractV2 outputs."""
    image_path = request.json.get('image_path')
    prompt_name = request.json.get('prompt_name') or request.json.get('prompt_file')

    result = check_csv_exists_v2(image_path, prompt_name)
    return jsonify(result)


@app.route('/v2/get_context', methods=['POST'])
def get_context_v2():
    """Return synthetic context metadata if available (only for synthetic plots)."""
    image_path = request.json.get('image_path', '') or ''

    # Only synthetic plots have generator-derived context files
    if not image_path.lower().startswith('synthetic/'):
        return jsonify({'found': False, 'reason': 'non_synthetic'})

    full_image_path = os.path.join(PLOTS_DIR, image_path)
    if not os.path.exists(full_image_path):
        return jsonify({'found': False, 'reason': 'image_missing'})

    base_name = os.path.splitext(os.path.basename(full_image_path))[0]
    context_path = os.path.join(os.path.dirname(full_image_path), f"{base_name}.context.json")

    if not os.path.exists(context_path):
        return jsonify({'found': False, 'reason': 'context_missing', 'context_path': os.path.relpath(context_path, PLOTS_DIR)})

    try:
        with open(context_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return jsonify({
            'found': True,
            'context_path': os.path.relpath(context_path, PLOTS_DIR),
            'content': content
        })
    except Exception as e:
        return jsonify({'found': False, 'reason': 'read_error', 'error': str(e)})

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

@app.route('/v2/get_outputs', methods=['POST'])
def get_outputs_v2():
    """Get output files for PlotExtractV2 using prompt sets."""
    import re

    image_path = request.json.get('image_path')
    prompt_name = request.json.get('prompt_name') or request.json.get('prompt_file')
    version_dir_param = request.json.get('version_dir')

    if not version_dir_param and prompt_name:
        image_dir = os.path.dirname(os.path.join(PLOTS_DIR, image_path))
        image_name = os.path.basename(image_path)
        base_name = os.path.splitext(image_name)[0]
        name_for_folder = image_name.replace('.', '_')
        full_prompt_name = f"pv2_{prompt_name}"

        version_pattern_new = re.compile(rf'^{re.escape(name_for_folder)}\.{re.escape(full_prompt_name)}\.v(\d+)(?:\.key\d+)?(?:\.web)?$')
        version_pattern_old = re.compile(rf'^{re.escape(base_name)}\.{re.escape(full_prompt_name)}\.v(\d+)(?:\.key\d+)?(?:\.web)?$')
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

    outputs = get_output_files_v2(image_path, prompt_name, version_dir_param)
    return jsonify({'outputs': outputs, 'version_dir': version_dir_param})


@app.route('/v2/load_saved_runs', methods=['POST', 'GET'])
def v2_load_saved_runs():
    """Load precomputed v2 runs from disk without executing extraction.

    Returns runs as objects compatible with the UI polling result:
      { success, outputs, console }
    """
    if request.method == 'GET':
        return jsonify({
            'success': False,
            'error': 'use_post',
            'hint': 'POST JSON: {"image_path": "first_examples/A/A-1/A-1.png", "prompt_name": "prompt_13", "limit": 3}',
        })

    data = request.json or {}
    image_path = (data.get('image_path') or data.get('image') or '').strip().replace('\\', '/')
    prompt_name = (data.get('prompt_name') or data.get('prompt') or data.get('prompt_file') or '').strip()
    version_dir_req = (data.get('version_dir') or data.get('version') or '').strip().replace('\\', '/')
    limit = data.get('limit')

    try:
        n = int(limit) if limit is not None else 3
    except Exception:
        n = 3
    n = max(1, min(10, n))

    image_abs = _safe_abs_under_plots(image_path)
    if not image_abs or not os.path.isfile(image_abs):
        return jsonify({'success': False, 'error': 'image_not_found', 'image_path': image_path}), 404

    if not prompt_name:
        return jsonify({'success': False, 'error': 'missing_prompt_name'}), 400

    # Optional: load one specific saved run directory.
    if version_dir_req:
        version_abs = _safe_abs_under_plots(version_dir_req)
        if (not version_abs) or (not os.path.isdir(version_abs)):
            return jsonify({
                'success': False,
                'error': 'version_dir_not_found',
                'version_dir': version_dir_req,
                'image_path': image_path,
                'prompt_name': prompt_name,
            }), 404

        progress_snapshot = _read_v2_progress_snapshot(version_abs)
        console_text, ok = _read_v2_progress_console(version_abs)
        outputs = get_output_files_v2(image_path, prompt_name, version_abs)
        run = {
            'version': 0,
            'version_dir': os.path.relpath(version_abs, PLOTS_DIR).replace('\\', '/'),
            'result': {
                'success': bool(ok),
                'outputs': outputs,
                'console': console_text,
                'accumulated_facts': progress_snapshot.get('accumulated_facts') if isinstance(progress_snapshot, dict) else None,
            }
        }
        return jsonify({
            'success': True,
            'image_path': image_path,
            'prompt_name': prompt_name,
            'requested': 1,
            'found': 1,
            'partial': False,
            'runs': [run],
            'selected_version_dir': run['version_dir'],
        })

    versions = _list_v2_version_dirs(image_path, prompt_name)
    if len(versions) < n and len(versions) == 0:
        # Return 200 with success=false so the UI can show a helpful message
        # without confusing this with a missing route.
        return jsonify({
            'success': False,
            'error': 'not_enough_saved_runs',
            'found': 0,
            'requested': n,
            'image_path': image_path,
            'prompt_name': prompt_name,
        })

    take_n = min(n, len(versions))
    runs = []
    for v, vdir in versions[:take_n]:
        progress_snapshot = _read_v2_progress_snapshot(vdir)
        console_text, ok = _read_v2_progress_console(vdir)
        outputs = get_output_files_v2(image_path, prompt_name, vdir)
        runs.append({
            'version': v,
            'version_dir': os.path.relpath(vdir, PLOTS_DIR).replace('\\', '/'),
            'result': {
                'success': bool(ok),
                'outputs': outputs,
                'console': console_text,
                'accumulated_facts': progress_snapshot.get('accumulated_facts') if isinstance(progress_snapshot, dict) else None,
            }
        })

    return jsonify({
        'success': True,
        'image_path': image_path,
        'prompt_name': prompt_name,
        'requested': n,
        'found': len(runs),
        'partial': len(runs) < n,
        'runs': runs,
    })

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

    # Optional: EX1 model selection (passed to plotExtract.py via env)
    llm_model = (data.get('llm_model') or data.get('llmModel') or '').strip() or None

    # Optional: rate-limit backoff mode (slower, disables server-side extraction timeout)
    rate_limit_backoff = bool(data.get('rate_limit_backoff') or data.get('rateLimitBackoff') or False)

    # Optional: batch metadata (for persistent batch results page)
    batch_number = data.get('batch_number')
    
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
            'prompt_file': prompt_file,
            'batch_number': batch_number
            ,
            'llm_model': llm_model,
            'rate_limit_backoff': rate_limit_backoff,
            'cancel_requested': False,
            'active_pid': None,
        }
    
    # Start background thread
    thread = threading.Thread(
        target=run_extraction_task,
        args=(task_id, image_path, prompt_file, run_interpolation, run_pointwise, 
              left_x, right_x, bottom_y, top_y, llm_model, rate_limit_backoff)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({'task_id': task_id, 'status': 'started'})

@app.route('/v2/run_all', methods=['POST'])
def run_all_v2():
    """Start v2 extraction pipeline in background."""
    import re

    data = request.json
    image_path = data.get('image')
    prompt_name = data.get('prompt') or data.get('prompt_name')
    article_info = data.get('articleInfo', '').strip()
    llm_key = str(data.get('llmKey') or data.get('llm_key') or '').strip()
    if not llm_key:
        llm_key = '4'
    llm_provider = (data.get('llm_provider') or data.get('llmProvider') or '').strip() or None
    llm_model = (data.get('llm_model') or data.get('llmModel') or '').strip() or None
    if (not llm_provider) and llm_key in ('1', '2', '4'):
        if llm_key == '2':
            llm_provider = 'google'
            llm_model = llm_model or 'gemma-3-27b-it'
        else:
            llm_provider = 'mistral'
            llm_model = llm_model or 'mistral-large-2512'
    debug_mode = bool(data.get('debug', False))
    rate_limit_backoff = bool(data.get('rate_limit_backoff') or data.get('rateLimitBackoff') or False)
    run_interpolation = data.get('runInterpolation', False)
    run_pointwise = data.get('runPointwise', False)
    left_x = str(data.get('leftX', 0))
    right_x = str(data.get('rightX', 100))
    bottom_y = str(data.get('bottomY', 0))
    top_y = str(data.get('topY', 100))

    task_id = str(uuid.uuid4())[:8]
    with extraction_tasks_lock:
        extraction_tasks[task_id] = {
            'status': 'running',
            'progress': 'Starting...',
            'console': [],
            'started_at': time.time(),
            'image_path': image_path,
            'prompt_name': prompt_name,
            'pipeline': 'v2',
            'debug': debug_mode,
            'rate_limit_backoff': rate_limit_backoff,
            'cancel_requested': False,
            'active_pid': None,
        }

    thread = threading.Thread(
        target=run_extraction_task_v2,
          args=(task_id, image_path, prompt_name, article_info, debug_mode, run_interpolation, run_pointwise,
                            left_x, right_x, bottom_y, top_y, llm_provider, llm_model, rate_limit_backoff)
    )
    thread.daemon = True
    thread.start()

    return jsonify({'task_id': task_id, 'status': 'started'})


# =============================================================================
# V2 Extraction Progress Tracking Routes
# =============================================================================

@app.route('/v2/extraction_progress/<task_id>')
def get_extraction_progress(task_id):
    """Get real-time progress of a running V2 extraction.
    
    Returns: current stage, percentage, accumulated facts, and stage timing."""
    try:
        progress_files = []

        # Prefer scoping the search to this task's image directory (more accurate, much faster)
        with extraction_tasks_lock:
            task = extraction_tasks.get(task_id)

        if task and task.get('pipeline') == 'v2':
            image_path = task.get('image_path')
            prompt_name = task.get('prompt_name')
            if image_path and prompt_name:
                full_image_path = os.path.join(PLOTS_DIR, image_path)
                image_dir = os.path.dirname(full_image_path)

                if os.path.isdir(image_dir):
                    for root, dirs, files in os.walk(image_dir):
                        # Only consider v2 output dirs for this prompt
                        if f'pv2_{prompt_name}' not in root:
                            continue
                        if '_extraction_progress.json' in files:
                            progress_path = os.path.join(root, '_extraction_progress.json')
                            mtime = os.path.getmtime(progress_path)
                            progress_files.append((mtime, progress_path))

        # Fallback: global search (e.g., if server restarted and task_id not in memory)
        if not progress_files:
            plots_dir = os.path.join(BASE_DIR, 'plots')
            for root, dirs, files in os.walk(plots_dir):
                if '_extraction_progress.json' in files:
                    progress_path = os.path.join(root, '_extraction_progress.json')
                    mtime = os.path.getmtime(progress_path)
                    progress_files.append((mtime, progress_path))

        if not progress_files:
            resp = jsonify({'status': 'not_found'})
            resp.headers['Cache-Control'] = 'no-store'
            return resp

        progress_files.sort(reverse=True)
        latest_progress_path = progress_files[0][1]
        with open(latest_progress_path, 'r', encoding='utf-8') as f:
            progress_data = json.load(f)

        # Add server-derived elapsed time so the UI timer doesn't reset on reload.
        with extraction_tasks_lock:
            task = extraction_tasks.get(task_id)
            started_at = (task or {}).get('started_at')
        if started_at:
            progress_data.setdefault('started_at', started_at)
            progress_data['elapsed'] = time.time() - float(started_at)

        resp = jsonify(progress_data)
        resp.headers['Cache-Control'] = 'no-store'
        return resp

    except Exception as e:
        resp = jsonify({'error': str(e), 'status': 'error'})
        resp.headers['Cache-Control'] = 'no-store'
        return resp, 500


# =============================================================================
# V2 Batch Extraction (server-side orchestration)
# =============================================================================

@app.route('/v2/run_batch', methods=['POST'])
def run_batch_v2():
    """Start a V2 batch extraction as a background task.

    This enables batch processing to continue even if the user navigates away.
    Payload:
      images: list[str] (plot paths relative to PLOTS_DIR)
      prompt: string (e.g., "prompt_1.py")
      articleInfo: string
      runInterpolation/runPointwise: bool
      axis ranges: leftX/rightX/bottomY/topY
    """
    data = request.get_json(force=True, silent=True) or {}
    images = data.get('images') or []
    prompt_file = data.get('prompt', 'prompt_1.py')
    batch_name = data.get('batch_name')
    article_info = data.get('articleInfo', '')
    llm_key = str(data.get('llmKey') or data.get('llm_key') or '').strip()
    if not llm_key:
        llm_key = '4'
    llm_provider = (data.get('llm_provider') or data.get('llmProvider') or '').strip() or None
    llm_model = (data.get('llm_model') or data.get('llmModel') or '').strip() or None
    if (not llm_provider) and llm_key in ('1', '2', '4'):
        if llm_key == '2':
            llm_provider = 'google'
            llm_model = llm_model or 'gemma-3-27b-it'
        else:
            llm_provider = 'mistral'
            llm_model = llm_model or 'mistral-large-2512'
    debug_mode = bool(data.get('debug', False))
    run_interpolation = bool(data.get('runInterpolation', False))
    run_pointwise = bool(data.get('runPointwise', False))
    left_x = str(data.get('leftX', '0'))
    right_x = str(data.get('rightX', '100'))
    bottom_y = str(data.get('bottomY', '0'))
    top_y = str(data.get('topY', '100'))

    if not isinstance(images, list) or not images:
        return jsonify({'error': 'No images provided'}), 400

    # Extract prompt name for v2 runner (e.g. "prompt_1.py" -> "prompt_1")
    prompt_name = os.path.splitext(prompt_file)[0]

    task_id = str(uuid.uuid4())

    # Create a persistent batch record (used by /batch_results)
    try:
        batch_number = create_batch_run_record(
            extraction_version='v2',
            prompt=prompt_file,
            images=images,
            pipeline='v2_batch',
            task_id=task_id,
            batch_name=batch_name,
        )
    except ValueError:
        return jsonify({'error': 'duplicate batch_name'}), 400
    except Exception as e:
        print(f"Failed to create batch record for v2 batch: {e}")
        batch_number = None
    with extraction_tasks_lock:
        extraction_tasks[task_id] = {
            'status': 'running',
            'progress': 'Starting batch... ',
            'console': [],
            'started_at': time.time(),
            'pipeline': 'v2_batch',
            'prompt_name': prompt_name,
            'batch_number': batch_number,
            'batch': {
                'total': len(images),
                'completed': 0,
                'failed': 0,
                'current_index': 0,
                'current_image': None,
                'items': [
                    {
                        'image_path': img,
                        'status': 'pending',
                        'task_id': None,
                        'time_s': None,
                        'error': None,
                        'result': None,
                    }
                    for img in images
                ],
            },
            'cancel_requested': False,
        }

    thread = threading.Thread(
        target=run_batch_task_v2,
        args=(task_id, images, prompt_name, article_info, debug_mode, run_interpolation, run_pointwise,
              left_x, right_x, bottom_y, top_y, llm_provider, llm_model),
    )
    thread.daemon = True
    thread.start()
    return jsonify({'task_id': task_id, 'status': 'started', 'batch_number': batch_number})


@app.route('/v2/batch_progress/<task_id>')
def get_batch_progress(task_id):
    """Get real-time progress for a V2 batch task."""
    with extraction_tasks_lock:
        task = extraction_tasks.get(task_id)
        if not task or task.get('pipeline') != 'v2_batch':
            resp = jsonify({'status': 'not_found'})
            resp.headers['Cache-Control'] = 'no-store'
            return resp

        payload = {
            'status': task.get('status', 'running'),
            'progress': task.get('progress', ''),
            'elapsed': time.time() - task.get('started_at', time.time()),
            'batch_number': task.get('batch_number'),
            'batch': task.get('batch', {}),
            'result': task.get('result'),
        }

    resp = jsonify(payload)
    resp.headers['Cache-Control'] = 'no-store'
    return resp


@app.route('/v2/cancel_task/<task_id>', methods=['POST'])
def cancel_task(task_id):
    """Request cancellation for a running task (best-effort).

    For batch tasks, this stops after the current image finishes.
    """
    with extraction_tasks_lock:
        if task_id not in extraction_tasks:
            return jsonify({'status': 'not_found'}), 404
        extraction_tasks[task_id]['cancel_requested'] = True
        return jsonify({'status': 'cancellation_requested'})


@app.route('/cancel_task/<task_id>', methods=['POST'])
def cancel_task_v1(task_id):
    """Request cancellation for a running task (v1 + generic).

    This is best-effort: the server will attempt to terminate the currently
    running subprocess and mark the task as cancelled.
    """
    with extraction_tasks_lock:
        if task_id not in extraction_tasks:
            return jsonify({'status': 'not_found'}), 404
        extraction_tasks[task_id]['cancel_requested'] = True
        extraction_tasks[task_id]['progress'] = 'Cancellation requested.'
        return jsonify({'status': 'cancellation_requested'})


def run_batch_task_v2(task_id, images, prompt_name, article_info, debug_mode, run_interpolation, run_pointwise,
                      left_x, right_x, bottom_y, top_y, llm_provider=None, llm_model=None):
    """Background batch runner.

    This orchestrates multiple single-image v2 tasks sequentially on the server,
    so batch continues even if the client disconnects.
    """
    batch_results = []
    batch_start = time.time()

    with extraction_tasks_lock:
        batch_number = (extraction_tasks.get(task_id) or {}).get('batch_number')

    def update_batch(**kwargs):
        with extraction_tasks_lock:
            if task_id not in extraction_tasks:
                return
            batch = extraction_tasks[task_id].setdefault('batch', {})
            batch.update(kwargs)

    for idx, image_path in enumerate(images):
        with extraction_tasks_lock:
            task = extraction_tasks.get(task_id)
            if not task:
                return
            if task.get('cancel_requested'):
                task['progress'] = 'Cancelled.'
                task['status'] = 'completed'
                task['result'] = {
                    'success': False,
                    'cancelled': True,
                    'batch_results': batch_results,
                    'completed_at': time.time(),
                }

                if batch_number:
                    try:
                        complete_batch_run(batch_number, status='cancelled')
                    except Exception as e:
                        print(f"Failed to mark batch {batch_number} cancelled: {e}")
                return

            task['progress'] = f'Processing {idx + 1}/{len(images)}: {image_path}'
            task['batch']['current_index'] = idx
            task['batch']['current_image'] = image_path
            task['batch']['items'][idx]['status'] = 'processing'

        child_task_id = str(uuid.uuid4())
        with extraction_tasks_lock:
            extraction_tasks[child_task_id] = {
                'status': 'running',
                'progress': 'Starting... ',
                'console': [],
                'started_at': time.time(),
                'image_path': image_path,
                'prompt_name': prompt_name,
                'pipeline': 'v2',
                'cancel_requested': False,
                'active_pid': None,
            }
            extraction_tasks[task_id]['batch']['items'][idx]['task_id'] = child_task_id

        start_time = time.time()
        try:
            run_extraction_task_v2(
                child_task_id,
                image_path,
                prompt_name,
                article_info,
                debug_mode,
                run_interpolation,
                run_pointwise,
                left_x,
                right_x,
                bottom_y,
                top_y,
                llm_provider,
                llm_model,
            )
            with extraction_tasks_lock:
                child = extraction_tasks.get(child_task_id, {})
                child_result = child.get('result')

            elapsed_s = round(time.time() - start_time, 2)
            ok = bool(child_result and child_result.get('success'))

            with extraction_tasks_lock:
                parent = extraction_tasks.get(task_id)
                if parent:
                    parent_item = parent['batch']['items'][idx]
                    parent_item['status'] = 'completed' if ok else 'failed'
                    parent_item['time_s'] = elapsed_s
                    parent_item['result'] = child_result
                    if not ok:
                        parent_item['error'] = (child_result or {}).get('console', '')[:5000] or 'Extraction failed'

                    parent['batch']['completed'] += 1
                    if not ok:
                        parent['batch']['failed'] += 1

            # Persist this item to the batch registry
            if batch_number:
                try:
                    summary = ((child_result or {}).get('outputs') or {}).get('summary') or {}
                    status = 'success' if ok else 'failed'
                    console_text = (child_result or {}).get('console')
                    if console_text and len(console_text) > 8000:
                        console_text = console_text[:8000] + "\n... (truncated)"
                    update_batch_run_item(
                        batch_number,
                        image_path,
                        status=status,
                        time_s=elapsed_s,
                        summary=summary,
                        error=None if ok else 'Extraction failed',
                        task_id=child_task_id,
                        version_dir=(child_result or {}).get('version_dir'),
                        console=console_text,
                    )
                except Exception as e:
                    print(f"Failed to persist batch item (batch {batch_number}, {image_path}): {e}")
        except Exception as e:
            elapsed_s = round(time.time() - start_time, 2)
            with extraction_tasks_lock:
                parent = extraction_tasks.get(task_id)
                if parent:
                    parent_item = parent['batch']['items'][idx]
                    parent_item['status'] = 'failed'
                    parent_item['time_s'] = elapsed_s
                    parent_item['error'] = str(e)
                    parent['batch']['completed'] += 1
                    parent['batch']['failed'] += 1

            if batch_number:
                try:
                    update_batch_run_item(
                        batch_number,
                        image_path,
                        status='failed',
                        time_s=elapsed_s,
                        summary={},
                        error=str(e),
                        task_id=child_task_id,
                    )
                except Exception:
                    pass

        # Keep parent results lightweight and deterministic for the UI
        batch_results.append({'image_path': image_path, 'task_id': child_task_id})

    with extraction_tasks_lock:
        if task_id in extraction_tasks:
            extraction_tasks[task_id]['status'] = 'completed'
            extraction_tasks[task_id]['progress'] = 'Batch complete.'
            extraction_tasks[task_id]['result'] = {
                'success': True,
                'batch_results': batch_results,
                'completed_at': time.time(),
                'total_time_s': round(time.time() - batch_start, 2),
            }

    if batch_number:
        try:
            complete_batch_run(batch_number, status='completed')
        except Exception as e:
            print(f"Failed to mark batch {batch_number} completed: {e}")


@app.route('/v2/extraction_console/<image_name>/<prompt_name>')
def get_extraction_console(image_name, prompt_name):
    """Display a standalone console for a completed V2 extraction.
    
    This page can be opened in a new tab to show full extraction details."""
    try:
        # Parse image_name and find the corresponding version directory
        # image_name format: "plotname" or with path elements
        base_name = image_name.replace('.png', '')
        
        # Search for the version directory
        version_dir = None
        plots_base = os.path.join(BASE_DIR, 'plots')
        
        for root, dirs, files in os.walk(plots_base):
            # Look for directories matching the prompt_name pattern
            for d in dirs:
                if f'pv2_{prompt_name}' in d and base_name in root:
                    version_dir = os.path.join(root, d)
                    break
            if version_dir:
                break
        
        if not version_dir or not os.path.isdir(version_dir):
            return f"<h1>Extraction console not found</h1><p>Image: {image_name}, Prompt: {prompt_name}</p>", 404
        
        # Read the progress file and all output files
        progress_file = os.path.join(version_dir, '_extraction_progress.json')
        progress_data = {}
        if os.path.exists(progress_file):
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
        
        # Read other output files
        tracking_file = os.path.join(version_dir, f"{image_name}.pv2_{prompt_name}.*.out_tracking")
        tracking_content = ""
        for f in os.listdir(version_dir):
            if f.endswith('.out_tracking'):
                with open(os.path.join(version_dir, f), 'r', encoding='utf-8') as tf:
                    tracking_content = tf.read()
                break
        
        # Build HTML page
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Extraction Console - {image_name}</title>
            <style>
                body {{ font-family: monospace; background: #1e1e1e; color: #d4d4d4; padding: 20px; }}
                .header {{ background: #333; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .header h1 {{ margin: 0; color: #4ec9b0; }}
                .metadata {{ margin: 10px 0; font-size: 0.9em; color: #858585; }}
                .console {{ background: #1e1e1e; border: 1px solid #444; padding: 15px; border-radius: 5px; max-height: 70vh; overflow-y: auto; }}
                .section {{ margin-bottom: 20px; }}
                .section-title {{ color: #4ec9b0; font-weight: bold; margin-bottom: 10px; }}
                .progress {{ background: #252526; padding: 10px; border-left: 3px solid #4ec9b0; margin-bottom: 15px; }}
                code {{ background: #252526; padding: 2px 5px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Extraction Console</h1>
                <div class="metadata">
                    <strong>Image:</strong> {image_name}<br>
                    <strong>Prompt:</strong> {prompt_name}
                </div>
            </div>
            
            <div class="console">
                <div class="section">
                    <div class="section-title">📊 Current Progress</div>
                    <div class="progress">
                        Stage: <code>{progress_data.get('stage', 'N/A')}</code><br>
                        Progress: <code>{progress_data.get('percentage', 0)}%</code> ({progress_data.get('stage_index', 0)}/{progress_data.get('total_stages', 5)})<br>
                        Stage Duration: <code>{progress_data.get('stage_duration_ms', 0):.0f}ms</code>
                    </div>
                </div>
                
                <div class="section">
                    <div class="section-title">📋 Accumulated Facts</div>
                    <pre style="background: #252526; padding: 10px; border-radius: 5px; overflow-x: auto; max-height: 300px;">{json.dumps(progress_data.get('accumulated_facts', {{}}), indent=2)}</pre>
                </div>
                
                <div class="section">
                    <div class="section-title">📝 Extraction Report</div>
                    <pre style="background: #252526; padding: 10px; border-radius: 5px; overflow-x: auto;">{tracking_content or 'No tracking data available yet'}</pre>
                </div>
            </div>
        </body>
        </html>
        """
        return html
    
    except Exception as e:
        return f"<h1>Error loading console</h1><p>{str(e)}</p>", 500


def run_extraction_task(task_id, image_path, prompt_file, run_interpolation, run_pointwise,
                        left_x, right_x, bottom_y, top_y, llm_model=None, rate_limit_backoff: bool = False):
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
    cancelled = False
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
        env_overrides = {'PLOTEXTRACT_OUTPUT_TAG': 'web'}
        if rate_limit_backoff:
            env_overrides.update({
                'PLOTEXTRACT_RATE_LIMIT_BACKOFF_MODE': '1',
                # Also disable the per-request LLM timeout in v2 (no effect on v1).
                'PLOTEXTRACT_MISTRAL_TIMEOUT_MS': '0',
            })
        if llm_model:
            env_overrides.update({
                'PLOTEXTRACT_LLM_PROVIDER': 'mistral',
                'PLOTEXTRACT_LLM_MODEL': str(llm_model),
                'PLOTEXTRACT_MISTRAL_MODEL': str(llm_model),
                'PLOTEXTRACT_LLM_KEY': '4',
            })
        extraction_timeout_s = None
        if not rate_limit_backoff:
            try:
                extraction_timeout_s = int(os.getenv('PLOTEXTRACT_EXTRACTION_TIMEOUT_S', '500'))
            except Exception:
                extraction_timeout_s = 500
        result = _run_subprocess_with_cancel(
            task_id,
            [PYTHON_EXE, 'plotExtract.py', full_image_path, full_prompt_path],
            cwd=BASE_DIR,
            timeout_s=extraction_timeout_s,
            env_overrides=env_overrides,
        )
        
        if result.get('stdout'):
            console_output.append(result['stdout'])
            # Parse VERSION_DIR from output
            for line in result['stdout'].split('\n'):
                if line.startswith('VERSION_DIR:'):
                    version_dir = line.replace('VERSION_DIR:', '').strip()
                    break
        if result.get('stderr'):
            console_output.append(f"[STDERR] {result['stderr']}")
        
        if result.get('returncode', 0) != 0:
            success = False
            step_status['extraction'] = f"failed (exit code {result.get('returncode')})"
            console_output.append(f"\n[ERROR] Extraction failed with exit code {result.get('returncode')}")
        else:
            step_status['extraction'] = 'success'
            console_output.append("\n[SUCCESS] Extraction completed.")
            
    except TaskCancelledError:
        success = False
        cancelled = True
        step_status['extraction'] = 'cancelled'
        console_output.append("[CANCELLED] Extraction cancelled by user.")
    except subprocess.TimeoutExpired:
        success = False
        step_status['extraction'] = 'failed (timeout)'
        console_output.append(f"[ERROR] Extraction timed out after {extraction_timeout_s} seconds")
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
        base_vdir = os.path.basename(version_dir)
        version_match = re.search(r'\.v(\d+)(?:\.key\d+)?(?:\.web)?$', base_vdir)
        if version_match:
            version_num = int(version_match.group(1))
        tag_suffix = '.web' if base_vdir.endswith('.web') else ''
        extracted_csv = os.path.join(version_dir, f"{image_name}.{prompt_name}.v{version_num}{tag_suffix}.mistral.out_data")
    else:
        name_for_folder = image_name.replace('.', '_')
        fallback_dir = os.path.join(image_dir, f"{name_for_folder}.{prompt_name}.v{version_num}.web")
        extracted_csv = os.path.join(fallback_dir, f"{image_name}.{prompt_name}.v{version_num}.web.mistral.out_data")
    
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
                cmd = [PYTHON_EXE, 'interpolation.py', original_csv, extracted_csv,
                       left_x, right_x, bottom_y, top_y]
                if version_dir:
                    cmd.append(version_dir)
                
                result = _run_subprocess_with_cancel(
                    task_id,
                    cmd,
                    cwd=BASE_DIR,
                    timeout_s=300,
                )
                
                if result.get('stdout'):
                    console_output.append(result['stdout'])
                if result.get('stderr'):
                    console_output.append(f"[STDERR] {result['stderr']}")
                
                if result.get('returncode', 0) != 0:
                    success = False
                    step_status['interpolation'] = f"failed (exit code {result.get('returncode')})"
                    console_output.append(f"\n[ERROR] Interpolation failed with exit code {result.get('returncode')}")
                else:
                    step_status['interpolation'] = 'success'
                    console_output.append("\n[SUCCESS] Interpolation completed.")
                    
            except TaskCancelledError:
                success = False
                cancelled = True
                step_status['interpolation'] = 'cancelled'
                console_output.append("[CANCELLED] Interpolation cancelled by user.")
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
                cmd = [PYTHON_EXE, 'pointwise.py', extracted_csv, original_csv,
                       left_x, right_x, bottom_y, top_y]
                if version_dir:
                    cmd.append(version_dir)

                try:
                    pointwise_timeout_s = int(os.getenv('PLOTEXTRACT_POINTWISE_TIMEOUT_S', '200'))
                except Exception:
                    pointwise_timeout_s = 200
                
                result = _run_subprocess_with_cancel(
                    task_id,
                    cmd,
                    cwd=BASE_DIR,
                    timeout_s=pointwise_timeout_s,
                )
                
                if result.get('stdout'):
                    console_output.append(result['stdout'])
                if result.get('stderr'):
                    console_output.append(f"[STDERR] {result['stderr']}")
                
                if result.get('returncode', 0) != 0:
                    success = False
                    step_status['pointwise'] = f"failed (exit code {result.get('returncode')})"
                    console_output.append(f"\n[ERROR] Pointwise comparison failed with exit code {result.get('returncode')}")
                else:
                    step_status['pointwise'] = 'success'
                    console_output.append("\n[SUCCESS] Pointwise comparison completed.")
                    
            except TaskCancelledError:
                success = False
                cancelled = True
                step_status['pointwise'] = 'cancelled'
                console_output.append("[CANCELLED] Pointwise comparison cancelled by user.")
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
        'cancelled': cancelled,
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
    batch_number = None
    with extraction_tasks_lock:
        if task_id in extraction_tasks:
            extraction_tasks[task_id]['status'] = 'completed'
            extraction_tasks[task_id]['result'] = final_result

            # If this was part of a batch, update persistent batch registry
            batch_number = extraction_tasks[task_id].get('batch_number')

    if batch_number:
        try:
            summary = (final_result.get('outputs') or {}).get('summary') or {}
            status = 'success' if final_result.get('success') else 'failed'
            time_s = (final_result.get('timings') or {}).get('total')
            console_text = final_result.get('console')
            if console_text and len(console_text) > 8000:
                console_text = console_text[:8000] + "\n... (truncated)"

            update_batch_run_item(
                batch_number,
                image_path,
                status=status,
                time_s=time_s,
                summary=summary,
                error=None if status == 'success' else 'Extraction failed',
                task_id=task_id,
                version_dir=final_result.get('version_dir'),
                console=console_text,
            )
        except Exception as e:
            print(f"Batch registry update failed (batch {batch_number}): {e}")
    
    # Save to file for persistence
    save_extraction_state(final_result)

def run_extraction_task_v2(task_id, image_path, prompt_name, article_info, debug_mode, run_interpolation, run_pointwise,
                           left_x, right_x, bottom_y, top_y, llm_provider=None, llm_model=None, rate_limit_backoff: bool = False):
    """Background task for PlotExtractV2 pipeline."""
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

    console_output = []
    success = True
    cancelled = False
    version_dir = None
    timings = {}
    step_status = {
        'extraction': 'pending',
        'interpolation': 'skipped',
        'pointwise': 'skipped'
    }

    csv_info = check_csv_exists_v2(image_path, prompt_name)
    original_csv = os.path.join(PLOTS_DIR, csv_info['original']['path'])

    image_dir = os.path.dirname(full_image_path)
    image_name = os.path.basename(image_path)
    base_name = os.path.splitext(image_name)[0]

    update_task(progress='Running extraction (v2 pipeline)...')
    console_output.append("=" * 60)
    console_output.append("STEP 1: Running PlotExtractV2")
    console_output.append("=" * 60)
    console_output.append(f"Image: {image_path}")
    console_output.append(f"Prompt set: {prompt_name}")
    console_output.append("")

    step1_start = time.time()
    try:
        env_overrides = {}
        env_overrides['PLOTEXTRACT_OUTPUT_TAG'] = 'web'
        if debug_mode:
            env_overrides['PLOTEXTRACT_DEBUG'] = '1'
        if rate_limit_backoff:
            env_overrides['PLOTEXTRACT_RATE_LIMIT_BACKOFF_MODE'] = '1'
            # Apply gentle pacing defaults; override via env if you want.
            env_overrides.setdefault('PLOTEXTRACT_LLM_MIN_INTERVAL_S', '5.0')
            env_overrides.setdefault('PLOTEXTRACT_BACKOFF_STAGE_SLEEP_S', '0.6')
            # Disable per-request timeout in the Mistral client (0 => no timeout)
            env_overrides.setdefault('PLOTEXTRACT_MISTRAL_TIMEOUT_MS', '0')
        provider_norm = (str(llm_provider) if llm_provider is not None else '').strip().lower()
        if llm_provider:
            env_overrides['PLOTEXTRACT_LLM_PROVIDER'] = str(llm_provider)
        # Record which API key slot is being used (requested: key1/key2)
        if provider_norm == 'google':
            env_overrides['PLOTEXTRACT_LLM_KEY'] = '2'
        elif provider_norm == 'mistral':
            env_overrides['PLOTEXTRACT_LLM_KEY'] = '4'
        if llm_model:
            env_overrides['PLOTEXTRACT_LLM_MODEL'] = str(llm_model)
            # Back-compat: existing Mistral path reads PLOTEXTRACT_MISTRAL_MODEL
            if provider_norm == 'mistral':
                env_overrides['PLOTEXTRACT_MISTRAL_MODEL'] = str(llm_model)
            if provider_norm == 'google':
                env_overrides['PLOTEXTRACT_GOOGLE_MODEL'] = str(llm_model)

        extraction_timeout_s = None
        if not rate_limit_backoff:
            try:
                extraction_timeout_s = int(os.getenv('PLOTEXTRACT_EXTRACTION_TIMEOUT_S', '500'))
            except Exception:
                extraction_timeout_s = 500

        if not env_overrides:
            env_overrides = None
        result = _run_subprocess_with_cancel(
            task_id,
            [PYTHON_EXE, os.path.join('plot_extract_v2', 'runner.py'), full_image_path, prompt_name, article_info],
            cwd=BASE_DIR,
            timeout_s=extraction_timeout_s,
            env_overrides=env_overrides,
        )

        if result.get('stdout'):
            console_output.append(result['stdout'])
            for line in result['stdout'].split('\n'):
                if line.startswith('VERSION_DIR:'):
                    version_dir = line.replace('VERSION_DIR:', '').strip()
                    break
        if result.get('stderr'):
            console_output.append(f"[STDERR] {result['stderr']}")

        if result.get('returncode', 0) != 0:
            success = False
            step_status['extraction'] = f"failed (exit code {result.get('returncode')})"
            console_output.append(f"\n[ERROR] Extraction failed with exit code {result.get('returncode')}")
        else:
            step_status['extraction'] = 'success'
            console_output.append("\n[SUCCESS] Extraction completed.")

    except TaskCancelledError:
        success = False
        cancelled = True
        step_status['extraction'] = 'cancelled'
        console_output.append("[CANCELLED] Extraction cancelled by user.")
    except subprocess.TimeoutExpired:
        success = False
        step_status['extraction'] = 'failed (timeout)'
        console_output.append(f"[ERROR] Extraction timed out after {extraction_timeout_s} seconds")
    except Exception as e:
        success = False
        step_status['extraction'] = 'failed (exception)'
        console_output.append(f"[ERROR] {str(e)}")

    step1_time = time.time() - step1_start
    timings['extraction'] = step1_time
    console_output.append(f"[TIME] Extraction took {step1_time:.2f} seconds")

    # Build the expected extracted CSV path using the v2 naming (pv2_<prompt_name>)
    full_prompt_name = f"pv2_{prompt_name}"
    version_num = 1
    if version_dir:
        version_match = re.search(r'\.v(\d+)(?:\.key\d+)?(?:\.web)?$', os.path.basename(version_dir))
        if version_match:
            version_num = int(version_match.group(1))
        # V2 runner saves the clean CSV as {base_name}_extracted.csv (Stage 4 backup)
        extracted_csv = os.path.join(version_dir, f"{base_name}_extracted.csv")
    else:
        name_for_folder = image_name.replace('.', '_')
        fallback_dir = os.path.join(image_dir, f"{name_for_folder}.{full_prompt_name}.v{version_num}.web")
        extracted_csv = os.path.join(fallback_dir, f"{base_name}_extracted.csv")

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
                cmd = [PYTHON_EXE, 'interpolation.py', original_csv, extracted_csv, left_x, right_x, bottom_y, top_y]
                if version_dir:
                    cmd.append(version_dir)

                result = _run_subprocess_with_cancel(
                    task_id,
                    cmd,
                    cwd=BASE_DIR,
                    timeout_s=300,
                )

                if result.get('stdout'):
                    console_output.append(result['stdout'])
                if result.get('stderr'):
                    console_output.append(f"[STDERR] {result['stderr']}")

                if result.get('returncode', 0) != 0:
                    success = False
                    step_status['interpolation'] = f"failed (exit code {result.get('returncode')})"
                    console_output.append(f"\n[ERROR] Interpolation failed with exit code {result.get('returncode')}")
                else:
                    step_status['interpolation'] = 'success'
                    console_output.append("\n[SUCCESS] Interpolation completed.")

            except TaskCancelledError:
                success = False
                cancelled = True
                step_status['interpolation'] = 'cancelled'
                console_output.append("[CANCELLED] Interpolation cancelled by user.")
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
                cmd = [PYTHON_EXE, 'pointwise.py', extracted_csv, original_csv, left_x, right_x, bottom_y, top_y]
                if version_dir:
                    cmd.append(version_dir)

                result = _run_subprocess_with_cancel(
                    task_id,
                    cmd,
                    cwd=BASE_DIR,
                    timeout_s=300,
                )

                if result.get('stdout'):
                    console_output.append(result['stdout'])
                if result.get('stderr'):
                    console_output.append(f"[STDERR] {result['stderr']}")

                if result.get('returncode', 0) != 0:
                    success = False
                    step_status['pointwise'] = f"failed (exit code {result.get('returncode')})"
                    console_output.append(f"\n[ERROR] Pointwise comparison failed with exit code {result.get('returncode')}")
                else:
                    step_status['pointwise'] = 'success'
                    console_output.append("\n[SUCCESS] Pointwise comparison completed.")

            except TaskCancelledError:
                success = False
                cancelled = True
                step_status['pointwise'] = 'cancelled'
                console_output.append("[CANCELLED] Pointwise comparison cancelled by user.")
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
    console_output.append("PIPELINE FINISHED SUCCESSFULLY" if success else "PIPELINE FINISHED WITH ERRORS")
    console_output.append(f"Extraction: {step_status['extraction']}")
    console_output.append(f"Interpolation: {step_status['interpolation']}")
    console_output.append(f"Pointwise: {step_status['pointwise']}")
    console_output.append(f"Total time: {total_time:.2f} seconds")
    console_output.append("=" * 60)

    outputs = get_output_files_v2(image_path, prompt_name, version_dir)
    csv_status = check_csv_exists_v2(image_path, prompt_name)

    final_result = {
        'success': success,
        'cancelled': cancelled,
        'console': '\n'.join(console_output),
        'outputs': outputs,
        'csv_status': csv_status,
        'version_dir': version_dir,
        'timings': timings,
        'completed_at': time.time(),
        'image_path': image_path,
        'prompt_name': prompt_name,
        'pipeline': 'v2'
    }

    with extraction_tasks_lock:
        if task_id in extraction_tasks:
            extraction_tasks[task_id]['status'] = 'completed'
            extraction_tasks[task_id]['result'] = final_result

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
        use_v2 = request.form.get('useV2', 'false') == 'true'
        article_info = request.form.get('articleInfo', '')
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
        base_name = os.path.splitext(original_filename)[0]
        image_path = os.path.join(batch_image_dir, original_filename)
        file.save(image_path)
        
        # Get relative path for the extraction pipeline
        rel_image_path = os.path.relpath(image_path, PLOTS_DIR).replace('\\', '/')
        
        # Determine which extraction system to use
        if use_v2:
            # Extract prompt name from file (e.g., 'prompt_1.py' -> 'prompt_1')
            prompt_name_v2 = os.path.splitext(prompt_file)[0]
            extraction_cmd = [PYTHON_EXE, 'plot_extract_v2/runner.py', image_path, prompt_name_v2]
            if article_info:
                extraction_cmd.append(article_info)
            prompt_short = prompt_name_v2.replace('prompt_', 'p')
            output_pattern = f".pv2_{prompt_name_v2}.v"
        else:
            # Use v1 extraction
            full_prompt_path = os.path.join(PROMPTS_DIR, prompt_file)
            extraction_cmd = [PYTHON_EXE, 'plotExtract.py', image_path, full_prompt_path]
            prompt_short = os.path.splitext(prompt_file)[0].replace('prompt_', 'p')
            output_pattern = f".{prompt_short}.v"
        
        console_output = []
        success = True
        version_dir = None
        timings = {}
        total_start_time = time.time()
        
        # Get prompt short name
        prompt_name = prompt_short
        
        # Step 1: Run extraction
        console_output.append("=" * 60)
        console_output.append("STEP 1: Running Plot Extraction")
        console_output.append("=" * 60)
        console_output.append(f"Image: {original_filename}")
        console_output.append(f"Prompt: {prompt_file}")
        console_output.append(f"Extraction: {'V2' if use_v2 else 'V1'}")
        if use_v2 and article_info:
            console_output.append(f"Article Info: {article_info[:100]}..." if len(article_info) > 100 else f"Article Info: {article_info}")
        console_output.append("")
        
        step1_start = time.time()
        try:
            try:
                extraction_timeout_s = int(os.getenv('PLOTEXTRACT_EXTRACTION_TIMEOUT_S', '500'))
            except Exception:
                extraction_timeout_s = 500
            result = subprocess.run(
                extraction_cmd,
                cwd=BASE_DIR,
                capture_output=True,
                text=True,
                timeout=extraction_timeout_s
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
            console_output.append(f"[ERROR] Extraction timed out after {extraction_timeout_s} seconds")
        except Exception as e:
            success = False
            console_output.append(f"[ERROR] {str(e)}")
        
        step1_time = time.time() - step1_start
        timings['extraction'] = step1_time
        console_output.append(f"[TIME] Extraction took {step1_time:.2f} seconds")
        
        # Determine extracted CSV path
        version_num = 1
        if version_dir:
            
            # Determine CSV filename based on extraction version
            if use_v2:
                # V2 saves as {image}_extracted.csv
                extracted_csv = os.path.join(version_dir, f"{base_name}_extracted.csv")
            else:
                # V1 saves as {filename}.{prompt}.v{N}.mistral.out_data
                extracted_csv = os.path.join(version_dir, f"{original_filename}.{prompt_name}.v{version_num}.mistral.out_data")
        else:
            name_for_folder = original_filename.replace('.', '_')
            fallback_dir = os.path.join(batch_image_dir, f"{name_for_folder}.{prompt_name}.v{version_num}")
            if use_v2:
                extracted_csv = os.path.join(fallback_dir, f"{base_name}_extracted.csv")
            else:
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
                    cmd = [PYTHON_EXE, 'interpolation.py', original_csv, extracted_csv,
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
                    cmd = [PYTHON_EXE, 'pointwise.py', extracted_csv, original_csv,
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
        
        # Always return outputs and results, regardless of validation status
        # This ensures plots with validation="no" are still displayed
        return jsonify({
            'success': success,
            'console': '\n'.join(console_output),
            'outputs': outputs,
            'version_dir': version_dir,
            'timings': timings,
            'filename': original_filename,
            'validation_status': outputs['summary'].get('validation_result', 'Unknown'),
            'show_outputs': True  # Always show outputs in batch mode
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
    return render_template(
        'synthetic.html',
        settings=settings,
        active_tab='synthetic',
        api_prefix='/synthetic',
        page_title='🧫 Synthetic Time-Kill Plot Generator',
        save_history_key='syntheticGeneratorSaveHistory',
    )


@app.route('/synthetic_v2')
def synthetic_v2():
    """Render the synthetic generator v2 page (template-curve generator)."""
    settings = load_synthetic_settings_v2()
    if not settings['curves']:
        settings['curves'] = get_default_curves(settings['num_curves'])
    return render_template(
        'synthetic.html',
        settings=settings,
        active_tab='synthetic_v2',
        api_prefix='/synthetic_v2',
        page_title='🧫 Synthetic Time-Kill Plot Generator v2',
        save_history_key='syntheticGeneratorV2SaveHistory',
    )

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


@app.route('/synthetic_v2/get_settings')
def synthetic_v2_get_settings():
    """Return current synthetic v2 settings as JSON."""
    settings = load_synthetic_settings_v2()
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


@app.route('/synthetic_v2/update_curves', methods=['POST'])
def synthetic_v2_update_curves():
    """Update the number of curves (v2) and return new curve configs."""
    data = request.json
    num_curves = int(data.get('num_curves', 3))

    settings = load_synthetic_settings_v2()
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
    save_synthetic_settings_v2(settings)
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


@app.route('/synthetic_v2/preview', methods=['POST'])
def synthetic_v2_preview():
    """Generate a preview of the synthetic plot (v2 template-curve generator)."""
    start_time = time.time()
    settings = request.json

    settings.setdefault('x_tick_mode', 'custom')
    settings.setdefault('x_tick_interval', 2)
    settings.setdefault('axis_break_enabled', False)
    settings.setdefault('axis_break_type', 'x')
    settings.setdefault('axis_break_start', '')
    settings.setdefault('axis_break_end', '')

    x_values, curves_data = generate_all_curves_v2(settings)
    fig = create_synthetic_plot(settings, curves_data, x_values)
    img_base64 = fig_to_base64(fig)
    save_synthetic_settings_v2(settings)
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


@app.route('/synthetic_v2/save', methods=['POST'])
def synthetic_v2_save():
    """Save the synthetic plot and data to files (v2 generator)."""
    start_time = time.time()
    settings = request.json

    settings.setdefault('x_tick_mode', 'custom')
    settings.setdefault('x_tick_interval', 2)
    settings.setdefault('axis_break_enabled', False)
    settings.setdefault('axis_break_type', 'x')
    settings.setdefault('axis_break_start', '')
    settings.setdefault('axis_break_end', '')

    x_values, curves_data = generate_all_curves_v2(settings)
    saved_files = save_synthetic_plot_and_data(settings, curves_data, x_values)
    save_synthetic_settings_v2(settings)
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


@app.route('/synthetic_v2/reset', methods=['POST'])
def synthetic_v2_reset():
    """Reset all synthetic v2 settings to defaults."""
    settings = DEFAULT_SETTINGS.copy()
    settings['curves'] = get_default_curves(settings['num_curves'])
    save_synthetic_settings_v2(settings)
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


@app.route('/synthetic_v2/regenerate', methods=['POST'])
def synthetic_v2_regenerate():
    """Regenerate curve data with same settings (v2, new random values)."""
    start_time = time.time()
    settings = request.json

    settings.setdefault('x_tick_mode', 'custom')
    settings.setdefault('x_tick_interval', 2)
    settings.setdefault('axis_break_enabled', False)
    settings.setdefault('axis_break_type', 'x')
    settings.setdefault('axis_break_start', '')
    settings.setdefault('axis_break_end', '')

    x_values, curves_data = generate_all_curves_v2(settings)
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
    """Get list of existing synthetic plots that can be edited.
    
    Looks in nested letter folders (A/, B/, C/, etc.)."""
    plots = []
    
    if os.path.exists(SYNTHETIC_DIR):
        # Look in nested letter folders (A/, B/, C/, D/, ...)
        for letter_folder in sorted(os.listdir(SYNTHETIC_DIR)):
            letter_path = os.path.join(SYNTHETIC_DIR, letter_folder)
            if os.path.isdir(letter_path) and len(letter_folder) == 1 and letter_folder.isalpha():
                # This is a letter folder, scan inside it
                for item in sorted(os.listdir(letter_path)):
                    item_path = os.path.join(letter_path, item)
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
    """Load an existing plot's data and settings for editing.
    
    Supports nested letter folder structure."""
    plot_folder = find_synthetic_plot_folder(plot_name)
    
    if not plot_folder:
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
        
        original_folder = find_synthetic_plot_folder(original_name)
        if not original_folder:
            return jsonify({'success': False, 'error': 'Plot folder not found'})

        # Create a new sibling folder for the copy so it shows up in the editor list
        letter_folder = os.path.dirname(original_folder)
        copy_num = 1
        while True:
            copy_name = f'{original_name}_copy{copy_num}'
            copy_folder = os.path.join(letter_folder, copy_name)
            if not os.path.exists(copy_folder):
                break
            copy_num += 1

        os.makedirs(copy_folder, exist_ok=True)
        
        fig = create_synthetic_plot(settings, curves_data, x_values)
        
        png_path = os.path.join(copy_folder, f'{copy_name}.png')
        fig.savefig(png_path, dpi=int(settings.get('dpi', 150)), bbox_inches='tight')
        
        svg_path = None
        if settings.get('save_svg', False):
            svg_path = os.path.join(copy_folder, f'{copy_name}.svg')
            fig.savefig(svg_path, format='svg', bbox_inches='tight')
        
        plt.close(fig)
        
        csv_path = os.path.join(copy_folder, f'{copy_name}-original.csv')
        save_synthetic_csv(curves_data, x_values, settings, csv_path)

        # Save context metadata for the edited copy
        context_path = os.path.join(copy_folder, f'{copy_name}.context.json')
        context_payload = build_synthetic_context(settings, curves_data, x_values, copy_name)
        with open(context_path, 'w', encoding='utf-8') as f:
            json.dump(context_payload, f, indent=2)
        elapsed = time.time() - start_time
        
        return jsonify({
            'success': True,
            'files': {
                'png': png_path,
                'svg': svg_path,
                'csv': csv_path,
                'context': context_path,
                'filename': copy_name,
                'folder': copy_folder
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
                if '-replot' in f and '-replot_overlay' not in f and f.endswith('.png'):
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
