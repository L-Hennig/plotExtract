import base64
import copy
import json
import os
import re
import sys
import traceback
import importlib
import time
import cv2
import numpy as np
from dotenv import load_dotenv
from mistralai import Mistral
import tempfile

# Import article info schema
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'prompts'))
from complete_extraction_schema import ARTICLE_INFO_SCHEMA, SCHEMA_CONSTRAINTS

# Import extraction tracker
from extraction_tracker import ExtractionTracker

# Load env for API key
load_dotenv(override=True)
API_KEY = os.getenv("API_KEY_1")

if len(sys.argv) < 3:
    print("Usage: python plot_extract_v2/runner.py <path_to_plot_image> <prompt_name> [article_info]\nError: Missing required argument. Please provide the path to the plot image and the prompt name (e.g., prompt_1).")
    sys.exit(1)

input_plot = sys.argv[1]
prompt_name = sys.argv[2]
article_info_text = sys.argv[3] if len(sys.argv) > 3 else ""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
PROMPTS_DIR = os.path.join(BASE_DIR, 'prompts')

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_module(module_path):
    spec = importlib.util.spec_from_file_location("dynamic_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def save_stage_update(output_dir, stage_name, stage_index, total_stages, accumulated_facts, stage_time_ms, console_output=""):
    """Save real-time stage update for the web UI to poll.
    
    Creates a JSON file with current extraction progress, accumulated facts, and timing."""
    try:
        # Determine if this is a completion update.
        # IMPORTANT: Do not treat the last stage as "complete" because the pipeline still
        # performs post-processing after the stage loop (CSV finalization, replot, comparisons).
        is_complete = stage_name == "COMPLETE"
        
        update_data = {
            "status": "complete" if is_complete else "running",
            "stage": stage_name,
            "stage_index": stage_index,
            "total_stages": total_stages,
            "percentage": round((stage_index / total_stages) * 100, 1) if total_stages > 0 else 100,
            "accumulated_facts": accumulated_facts,
            "stage_duration_ms": stage_time_ms,
            "timestamp": time.time(),
            "console_output": console_output
        }
        
        update_file = os.path.join(output_dir, "_extraction_progress.json")
        with open(update_file, "w", encoding="utf-8") as f:
            json.dump(update_data, f, indent=2)
    except Exception as e:
        print(f"[WARNING] Could not save stage update: {e}")


def _snapshot_jsonable(obj):
    """Best-effort immutable snapshot for console logging.

    Prefer a JSON round-trip (guarantees no shared references); fall back to deepcopy.
    """
    try:
        return json.loads(json.dumps(obj, ensure_ascii=False))
    except Exception:
        try:
            return copy.deepcopy(obj)
        except Exception:
            return obj


def _format_stage_facts_dump(stage_number: int, accumulated_facts: dict) -> str:
    header = f"===== STAGE {stage_number} COMPLETE — ACCUMULATED FACTS ====="
    snapshot = _snapshot_jsonable(accumulated_facts)
    payload = json.dumps(snapshot, indent=2, ensure_ascii=False, default=str)
    return f"{header}\n{payload}\n"


def _extract_json_object_from_text(text: str):
    """Best-effort JSON extraction.

    The model sometimes wraps JSON in ```json fences, or includes extra text.
    This function tries (in order): direct parse, fenced-block parse, and
    a substring from first '{' to last '}'. Returns the parsed object or None.
    """
    if not text:
        return None

    raw = text.strip()

    # 1) Direct parse (fast path)
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 2) If the whole content is fenced, strip the outer fence
    if raw.startswith("```"):
        unfenced = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        unfenced = re.sub(r"\s*```$", "", unfenced)
        try:
            return json.loads(unfenced.strip())
        except Exception:
            pass

    # 3) Parse first fenced JSON block
    for match in re.finditer(r"```json\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE):
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            continue

    # 4) Parse best-effort substring from first '{'..last '}'
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None

    return None

def load_prompt_set(prompt_name):
    """Load prompts from prompt_name/prompts.py and chain metadata."""
    prompt_set_dir = os.path.join(PROMPTS_DIR, prompt_name)
    
    if not os.path.exists(prompt_set_dir):
        raise FileNotFoundError(f"Prompt set '{prompt_name}' not found at {prompt_set_dir}")
    
    # Load prompts from prompt_name/prompts.py
    prompts_file = os.path.join(prompt_set_dir, 'prompts.py')
    if not os.path.exists(prompts_file):
        raise FileNotFoundError(f"Prompts file not found at {prompts_file}")
    
    prompts_module = load_module(prompts_file)
    
    # Load chain metadata from chains/chain_<prompt_name>.py
    chains_dir = os.path.join(PROMPTS_DIR, 'chains')
    chain_file = os.path.join(chains_dir, f'chain_{prompt_name}.py')
    
    if not os.path.exists(chain_file):
        raise FileNotFoundError(f"Chain metadata file not found at {chain_file}")
    
    chain_module = load_module(chain_file)
    
    # Validate chain has required attributes
    if not hasattr(chain_module, 'EXTRACT_STAGES'):
        raise ValueError(f"Chain file {chain_file} missing 'EXTRACT_STAGES' list")
    
    return prompts_module, chain_module


def load_extraction_schema():
    """Load the complete extraction schema."""
    schema_path = os.path.join(PROMPTS_DIR, 'extraction_schema.json')
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Extraction schema not found at {schema_path}")
    
    with open(schema_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def detect_abort_signal(result_text):
    """Detect whether a stage signaled an abort.

    Accepts multiple formats for backward compatibility:
    - Exact string "None" (legacy early-stop)
    - A response starting with "ABORT" (free-form abort message)
    - JSON object containing {"abort": true, "reason": "..."}
      or nested abort flags under known sections (e.g., marker_facts, axis_facts).
    Returns (abort: bool, reason: Optional[str], confidence: Optional[float]).
    """
    text = str(result_text).strip()

    # Legacy: exact None
    if text == "None":
        return True, "Legacy abort: 'None'", None

    # Free-form: starts with ABORT
    if text.upper().startswith("ABORT"):
        return True, text, None

    # Structured: JSON with abort flag
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            # Top-level abort
            if obj.get("abort") is True:
                reason = obj.get("reason") or obj.get("abort_reason") or "Abort signaled by prompt"
                conf = obj.get("confidence")
                try:
                    conf = float(conf) if conf is not None else None
                except (TypeError, ValueError):
                    conf = None
                return True, reason, conf

            # Nested abort under known sections
            for section in ("marker_facts", "axis_facts"):
                sec = obj.get(section)
                if isinstance(sec, dict) and sec.get("abort") is True:
                    reason = sec.get("reason") or "Abort signaled by prompt"
                    conf = sec.get("confidence")
                    try:
                        conf = float(conf) if conf is not None else None
                    except (TypeError, ValueError):
                        conf = None
                    return True, reason, conf
    except (json.JSONDecodeError, TypeError):
        pass

    return False, None, None


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def create_Q_1p(convo):
    Q = []
    for ic, c in enumerate(convo):
        role = "user" if ic % 2 == 0 else "assistant"
        if isinstance(c, list):
            Q.append({
                "role": role,
                "content": [
                    {"type": "text", "text": c[1]},
                    {"type": "image_url", "image_url": {"url": f"data:image/{PNGJPG};base64,{c[0]}"}},
                ],
            })
        else:
            Q.append({"role": role, "content": c})
    return Q


def prompt_mistral(client, messages):
    response = client.chat.complete(
        model="mistral-large-2512",
        messages=messages,
        max_tokens=4096,
        temperature=0,
    )
    return messages, response.choices[0].message.content


def clean_code_response(code_text):
    """Remove markdown code fences and other formatting from LLM response."""
    code_text = code_text.strip()
    
    # Remove markdown code fences
    if code_text.startswith("```python"):
        code_text = code_text[len("```python"):].strip()
    elif code_text.startswith("```"):
        code_text = code_text[3:].strip()
    
    if code_text.endswith("```"):
        code_text = code_text[:-3].strip()
    
    return code_text


def extract_csv_from_text(text):
    """Extract pure CSV content from mixed LLM output (csv fenced block or JSON marker_facts)."""
    text = text.strip()

    # Try direct JSON parsing first (for complete JSON responses)
    try:
        obj = json.loads(text)
        csv_val = obj.get("marker_facts", {}).get("csv_output")
        if csv_val:
            # Unescape if needed
            if isinstance(csv_val, str):
                csv_val = csv_val.replace("\\n", "\n").replace("\\\\", "\\")
            return csv_val.strip()
    except (json.JSONDecodeError, TypeError):
        pass

    # Try JSON block with marker_facts.csv_output (for mixed content)
    json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.S)
    if json_match:
        try:
            obj = json.loads(json_match.group(1))
            csv_val = obj.get("marker_facts", {}).get("csv_output")
            if csv_val:
                if isinstance(csv_val, str):
                    csv_val = csv_val.replace("\\n", "\n").replace("\\\\", "\\")
                return csv_val.strip()
        except Exception:
            pass

    # Try fenced CSV block
    csv_match = re.search(r"```csv\s*(.*?)\s*```", text, re.S)
    if csv_match:
        return csv_match.group(1).strip()

    # Try to find CSV-like content by looking for lines with numbers and commas
    # This is a fallback for when the LLM output is not properly formatted
    lines = text.split('\n')
    csv_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip empty lines and non-CSV-like content
        if not line or line.startswith("Here is") or line.startswith("Step "):
            continue
        # Look for lines that look like CSV (contain commas and numbers/text)
        if ',' in line:
            csv_lines.append(line)
    
    # If we found CSV-like lines, return them
    if csv_lines and len(csv_lines) > 1:  # At least header + 1 data row
        return "\n".join(csv_lines)

    # Fallback: return raw text
    return text


def rebuild_csv_from_json_curves(accumulated_facts):
    """Rebuild clean CSV from marker_facts.curves JSON, ignoring the LLM's csv_output string.
    
    This ensures we use the actual structured points rather than the LLM's often-malformed CSV."""
    try:
        marker_facts = accumulated_facts.get("marker_facts", {})
        curves = marker_facts.get("curves", [])
        if not curves:
            return None
        
        # Collect all unique x values and curve labels
        x_set = {}  # {x_value: index} to preserve order
        curve_labels = []
        seen_curves = set()
        
        for curve_obj in curves:
            label = curve_obj.get("curve_label", "Unknown")
            if label not in seen_curves:
                curve_labels.append(label)
                seen_curves.add(label)
            
            points = curve_obj.get("points", [])
            for pt in points:
                x = pt.get("x")
                if x is not None and x not in x_set:
                    x_set[x] = len(x_set)
        
        if not x_set or not curve_labels:
            return None
        
        x_values = sorted(x_set.keys(), key=lambda v: x_set[v])
        
        # Build mapping (x, curve_label) -> y
        y_map = {}
        for curve_obj in curves:
            label = curve_obj.get("curve_label")
            points = curve_obj.get("points", [])
            for pt in points:
                x = pt.get("x")
                y = pt.get("y")
                if x is not None and y is not None:
                    y_map[(x, label)] = y
        
        # Build header: Time (hours), log10 CFU/mL (Curve1), Time (hours), log10 CFU/mL (Curve2), ...
        header = []
        for label in curve_labels:
            header.append("Time (hours)")
            header.append(f"log10 CFU/mL ({label})")
        
        lines = [",".join(header)]
        
        # Build data rows
        for x in x_values:
            row = []
            for label in curve_labels:
                row.append(str(x))
                y = y_map.get((x, label), "")
                row.append(str(y) if y != "" else "")
            lines.append(",".join(row))
        
        return "\n".join(lines)
    except Exception as e:
        print(f"Error rebuilding CSV from JSON: {e}")
        return None


def normalize_csv_to_wide(csv_text: str) -> str:
    """Convert long-form CSV (x, curve_label, y) into wide format expected by interpolation.

    If the CSV is already wide (multiple x,y column pairs), it is returned unchanged.
    """
    lines = [ln.strip() for ln in csv_text.splitlines() if ln.strip()]
    if not lines:
        return csv_text

    header = [h.strip() for h in lines[0].split(",")]

    # Check if already in wide format (even number of columns with alternating x/y pairs)
    # Wide format: "Time (hours)", "CFU/mL (Condition 1)" [2 cols for 1 curve]
    #             OR "Time (hours)", "CFU/mL (Condition 1)", "Time (hours)", "CFU/mL (Condition 2)" [4+ cols for 2+ curves]
    if len(header) >= 2 and len(header) % 2 == 0:
        # Check if columns alternate between x and y descriptors
        is_wide = True
        for i in range(0, len(header), 2):
            x_col = header[i].lower()
            # X column should contain "time" or "hour" or similar
            if not any(kw in x_col for kw in ["time", "hour", "x"]):
                is_wide = False
                break
        if is_wide:
            return csv_text  # Already wide format, return unchanged

    # Detect long format variants
    # Case A: strict 3-column long format: x, curve_label, y
    # Case B: 4+ columns where last column is condition/curve label and one numeric column is y
    rows = []
    if len(header) == 3:
        x_col, label_col, y_col = header
        for ln in lines[1:]:
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) != 3:
                continue
            x_val, curve_label, y_val = parts
            rows.append((x_val, curve_label, y_val))
    else:
        # Try heuristic: last column is curve label (condition), first column is x, one numeric column is y
        # Adjust for cases where header has more columns than data rows
        label_idx = len(header) - 1
        x_idx = 0

        # Determine the max actual column count across data rows
        max_cols = 0
        for ln in lines[1:]:
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) > max_cols:
                max_cols = len(parts)
        if max_cols == 0:
            return csv_text

        # Clamp label_idx to existing columns
        if label_idx >= max_cols:
            label_idx = max_cols - 1

        # pick y_idx as last numeric-ish column before label
        y_idx = None
        for idx in range(min(label_idx - 1, max_cols - 1), -1, -1):
            sample_val = None
            for ln in lines[1:]:
                parts = [p.strip() for p in ln.split(",") if p.strip()]
                if len(parts) > idx:
                    sample_val = parts[idx]
                    break
            if sample_val is not None:
                try:
                    float(sample_val)
                    y_idx = idx
                    break
                except ValueError:
                    continue
        if y_idx is None:
            return csv_text  # cannot detect

        x_col = header[x_idx]
        y_col = header[y_idx]
        label_col = header[label_idx]

        for ln in lines[1:]:
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) <= max(x_idx, y_idx, label_idx):
                continue
            x_val = parts[x_idx]
            y_val = parts[y_idx]
            curve_label = parts[label_idx]
            rows.append((x_val, curve_label, y_val))

    if not rows:
        return csv_text

    # Collect unique x values in original order and curve labels
    x_values = []
    curves = []
    seen_x = set()
    seen_curve = set()
    for x_val, curve_label, _ in rows:
        if x_val not in seen_x:
            seen_x.add(x_val)
            x_values.append(x_val)
        if curve_label not in seen_curve:
            seen_curve.add(curve_label)
            curves.append(curve_label)

    # Build mapping: (x, curve) -> y
    y_map = {}
    for x_val, curve_label, y_val in rows:
        y_map[(x_val, curve_label)] = y_val

    # Construct wide header: Time, Y(curve1), Time, Y(curve2), ...
    wide_header = []
    for curve in curves:
        wide_header.append(x_col)
        wide_header.append(f"{y_col} ({curve})")

    wide_lines = [",".join(wide_header)]

    for x_val in x_values:
        row_vals = []
        for curve in curves:
            row_vals.append(x_val)
            row_vals.append(y_map.get((x_val, curve), ""))
        wide_lines.append(",".join(row_vals))

    return "\n".join(wide_lines)


def get_next_version(parent_dir, name_for_folder, prompt_name):
    version = 1
    while True:
        folder_name = f"{name_for_folder}.{prompt_name}.v{version}"
        folder_path = os.path.join(parent_dir, folder_name)
        if not os.path.exists(folder_path):
            return version, folder_path
        version += 1


def stack_images_vertically(image1_path, image2_path, border_color, output_dir, prompt_name, version_num, border_size=30):
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    if img1 is None or img2 is None:
        print(f"Error: Cannot read images. img1={image1_path}, img2={image2_path}")
        return None

    width = max(img1.shape[1], img2.shape[1])
    img1_resized = cv2.resize(img1, (width, int(img1.shape[0] * width / img1.shape[1])))
    img2_resized = cv2.resize(img2, (width, int(img2.shape[0] * width / img2.shape[1])))

    label_height = 60
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.8
    font_thickness = 3
    label1 = np.ones((label_height, width, 3), dtype=np.uint8) * 255
    label2 = np.ones((label_height, width, 3), dtype=np.uint8) * 255
    cv2.putText(label1, "Original", (15, 45), font, font_scale, (0, 0, 0), font_thickness)
    cv2.putText(label2, "Extracted (Re-plotted)", (15, 45), font, font_scale, (0, 0, 0), font_thickness)
    combined_image = np.vstack((label1, img1_resized, label2, img2_resized))

    if "yes" in border_color.lower():
        color = (0, 255, 0)
    elif "no" in border_color.lower():
        color = (0, 0, 255)
    else:
        print("Invalid border color input. Use 'yes' for green or 'no' for red.")
        return None

    combined_image_with_border = cv2.copyMakeBorder(
        combined_image,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )

    original_filename = os.path.basename(image1_path)
    base_name = original_filename.rsplit(".", 1)[0] if "." in original_filename else original_filename
    output_filename = os.path.join(output_dir, f"comparison_{base_name}.{prompt_name}.v{version_num}.png")
    cv2.imwrite(output_filename, combined_image_with_border)
    return output_filename


# NOTE: Prompts and stages are loaded via load_prompt_set() function after image encoding below

# Load prompts and chain metadata from the new modular structure
try:
    prompts_module, chain_module = load_prompt_set(prompt_name)
except (FileNotFoundError, ValueError) as e:
    print(f"Error loading prompt set: {e}")
    sys.exit(1)

CHAIN_NAME = getattr(chain_module, "CHAIN_NAME", prompt_name)
EXTRACT_STAGES = chain_module.EXTRACT_STAGES

# Verify all stages exist in prompts module
for stage_name in EXTRACT_STAGES:
    if not hasattr(prompts_module, stage_name):
        print(f"Error: Stage '{stage_name}' not found in prompts module")
        sys.exit(1)

# Also load shared prompts (CODE_PLOT, CODE_FIX, COMPARE_*)
for prompt_attr in ['CODE_PLOT', 'CODE_FIX', 'COMPARE_X', 'COMPARE_Y', 'COMPARE_NUMBER', 'COMPARE_TREND']:
    if not hasattr(prompts_module, prompt_attr):
        print(f"Warning: {prompt_attr} not found in prompts module")

# Build a version identifier using the prompt name
prompt_version = f"pv2_{prompt_name}"

# -----------------------------------------------------------------------------
# Image handling
# -----------------------------------------------------------------------------
original_ext = os.path.splitext(input_plot)[-1].lower()
api_image_path = input_plot
PNGJPG = original_ext.lstrip(".")
if PNGJPG == "jpg":
    PNGJPG = "jpeg"
elif PNGJPG == "svg":
    import subprocess as sp
    import tempfile

    temp_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    abs_svg = os.path.abspath(input_plot).replace("\\", "/")
    html_content = f"""<!DOCTYPE html>
<html><head><style>
html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background: white; }}
img {{ width: 100%; height: 100%; object-fit: contain; }}
</style></head>
<body><img src=\"file:///{abs_svg}\"></body></html>"""

    temp_html = tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8")
    temp_html.write(html_content)
    temp_html.close()

    browsers = [
        r"C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
        r"C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
        "msedge",
        r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        r"C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
        "chrome",
    ]

    converted = False
    for browser in browsers:
        try:
            sp.run(
                [
                    browser,
                    "--headless",
                    "--disable-gpu",
                    f"--screenshot={temp_png}",
                    "--window-size=1600,1200",
                    "--force-device-scale-factor=2",
                    f"file:///{temp_html.name.replace(os.sep, '/')}",
                ],
                capture_output=True,
                timeout=60,
            )
            if os.path.exists(temp_png) and os.path.getsize(temp_png) > 0:
                api_image_path = temp_png
                PNGJPG = "png"
                converted = True
                print("Converted SVG to PNG (high resolution)")
                break
        except (FileNotFoundError, sp.TimeoutExpired, OSError):
            continue

    try:
        os.unlink(temp_html.name)
    except Exception:
        pass

    if not converted:
        print("ERROR: Cannot convert SVG to PNG. Install Edge or Chrome.")
        sys.exit(1)

base64_image = encode_image(api_image_path)

# Initialize Mistral client
client = Mistral(api_key=API_KEY)

# -----------------------------------------------------------------------------
# Preprocessing: Convert article info to structured JSON (if provided)
# -----------------------------------------------------------------------------
ARTICLE_INFO_PROMPT = r"""You are given free-text information copied from a research article that describes a plot (e.g. figure caption, legend text, brief experimental context).

Your task is to convert this text into a structured JSON object called `article_info`, following the schema below.

Rules (non-negotiable):
- Use ONLY the information explicitly present in the provided text.
- Do NOT guess, infer, or fill in missing fields.
- If a field cannot be confidently populated, OMIT it entirely.
- Do NOT paraphrase figure captions or legend text; copy verbatim where applicable.
- Output VALID JSON only. No explanations, no commentary, no extra text.

The resulting JSON will be treated as hard facts and passed unchanged to later extraction stages.

JSON schema to follow:

{schema}

{constraints}

Input:
{article_text}

Output:
(valid JSON only)"""

article_info_json = ""
if article_info_text:
    print("Preprocessing article information...")
    article_prompt = ARTICLE_INFO_PROMPT.format(
        schema=ARTICLE_INFO_SCHEMA,
        constraints=SCHEMA_CONSTRAINTS,
        article_text=article_info_text
    )
    messages = [{"role": "user", "content": article_prompt}]
    _, article_response = prompt_mistral(client, messages)
    
    # Clean response and validate JSON
    article_response = article_response.strip()
    if article_response.startswith("```json"):
        article_response = article_response[len("```json"):].strip()
    if article_response.startswith("```"):
        article_response = article_response[3:].strip()
    if article_response.endswith("```"):
        article_response = article_response[:-3].strip()
    
    try:
        json.loads(article_response)  # Validate JSON
        article_info_json = article_response
        print("Article information structured successfully")
    except json.JSONDecodeError:
        print("Warning: Article information could not be parsed as valid JSON, skipping")
        article_info_json = ""

# -----------------------------------------------------------------------------
# Output naming
# -----------------------------------------------------------------------------
image_filename = os.path.basename(input_plot)
base_name = os.path.splitext(image_filename)[0]
name_for_folder = image_filename.replace(".", "_")
full_prompt_name = f"pv2_{prompt_name}"
version_num, version_dir = get_next_version(os.path.dirname(input_plot), name_for_folder, full_prompt_name)
os.makedirs(version_dir, exist_ok=True)

output_out = os.path.join(version_dir, f"{image_filename}.{full_prompt_name}.v{version_num}.mistral.out")
replot_plot = os.path.join(version_dir, f"{name_for_folder}-replot.{full_prompt_name}.v{version_num}.png")

print(f"Input plot: {input_plot}")
print(f"Using prompt set: {prompt_name}")
print(f"Output folder: {version_dir} (version {version_num})")

stage_context = {}
conversation_log = []
console_timeline = ""

# Initialize extraction tracker
tracker = ExtractionTracker(input_plot, prompt_name)
tracker.initialize_stages(EXTRACT_STAGES)

# If article info was provided, add it as initial context
if article_info_json:
    stage_context["article_info"] = article_info_json

# -----------------------------------------------------------------------------
# Stage 1: extraction (and additional stages)
# Load the complete extraction schema
extraction_schema = load_extraction_schema()
complete_schema_str = json.dumps(extraction_schema, indent=2)

# Initialize accumulated facts with article info if provided
accumulated_facts = {}
if article_info_json:
    accumulated_facts = json.loads(article_info_json)

# Stage processing loop
for stage_index, stage_name in enumerate(EXTRACT_STAGES):
    # Get the prompt text from the prompts module
    if not hasattr(prompts_module, stage_name):
        print(f"Error: Stage '{stage_name}' not found in prompts module")
        sys.exit(1)
    
    tracker.start_stage(stage_name)
    stage_start_time = time.time()
    
    prompt_payload = getattr(prompts_module, stage_name)
    
    # Format prompt with complete schema and accumulated facts
    accumulated_facts_str = json.dumps(accumulated_facts, indent=2) if accumulated_facts else "Empty (no facts extracted yet)"
    
    if "{complete_schema}" in prompt_payload or "{accumulated_facts}" in prompt_payload:
        prompt_payload = prompt_payload.format(
            complete_schema=complete_schema_str,
            accumulated_facts=accumulated_facts_str,
            replot_path=replot_plot,
        )
    elif "{data_context}" in prompt_payload:
        # Fallback for old-style prompts
        accumulated_context = ""
        if article_info_json:
            accumulated_context = f"=== Article Information ===\n{article_info_json}\n\n"
        
        for prev_stage_name in EXTRACT_STAGES[:stage_index]:
            if prev_stage_name in stage_context:
                accumulated_context += f"\n=== {prev_stage_name} ===\n{stage_context[prev_stage_name]}\n"
        
        prompt_payload = prompt_payload.format(
            data_context=accumulated_context.strip() if accumulated_context else "No previous context",
            replot_path=replot_plot,
        )

    messages = create_Q_1p([[base64_image, prompt_payload]])
    conversation_log.extend(messages)
    messages, result_text = prompt_mistral(client, messages)
    conversation_log.append({"role": "assistant", "content": result_text})
    stage_context[stage_name] = result_text
    
    stage_time = (time.time() - stage_start_time) * 1000  # Convert to ms
    
    # Parse stage output as JSON (supports ```json fenced blocks) and merge into accumulated facts
    result_json = _extract_json_object_from_text(result_text)
    if isinstance(result_json, dict):
        for key in result_json:
            if (
                key in accumulated_facts
                and isinstance(accumulated_facts.get(key), dict)
                and isinstance(result_json.get(key), dict)
            ):
                accumulated_facts[key].update(result_json[key])
            else:
                accumulated_facts[key] = result_json[key]
    
    # Check for abort signal (supports legacy and structured formats)
    abort, abort_reason, abort_confidence = detect_abort_signal(result_text)

    # Extract facts from CSV if this is an extraction stage
    facts = None
    if "csv" in result_text.lower() or stage_index == len(EXTRACT_STAGES) - 1:
        facts = tracker.extract_facts_from_csv(result_text, stage_name)
    
    # Save CSV immediately after data extraction stage (e.g., EXTRACT_STAGE_4)
    # This ensures CSV is preserved and can be reused by later stages
    if stage_name == "EXTRACT_STAGE_4":
        try:
            csv_data = extract_csv_from_text(result_text)
            print(f"[DEBUG STAGE 4] Initial extracted CSV lines: {len(csv_data.split(chr(10)))}")
            print(f"[DEBUG STAGE 4] First line: {csv_data.split(chr(10))[0][:100] if csv_data else 'EMPTY'}")
            
            # Try to rebuild CSV from JSON curves for better quality
            # This avoids using the LLM's often-malformed csv_output string
            try:
                accumulated_facts_copy = accumulated_facts.copy()
                # Try to parse result_text as JSON to get fresh marker_facts
                result_json = _extract_json_object_from_text(result_text)
                
                if isinstance(result_json, dict) and "marker_facts" in result_json:
                    print(f"[DEBUG STAGE 4] Found marker_facts in result_json")
                    curves = result_json["marker_facts"].get("curves", [])
                    print(f"[DEBUG STAGE 4] Number of curves: {len(curves)}")
                    if curves:
                        print(f"[DEBUG STAGE 4] First curve: {curves[0].get('curve_label')} with {len(curves[0].get('points', []))} points")
                    
                    accumulated_facts_copy["marker_facts"] = result_json["marker_facts"]
                    rebuilt_csv = rebuild_csv_from_json_curves(accumulated_facts_copy)
                    if rebuilt_csv:
                        print(f"[DEBUG STAGE 4] Successfully rebuilt CSV from JSON curves")
                        print(f"[DEBUG STAGE 4] Rebuilt CSV first line: {rebuilt_csv.split(chr(10))[0][:100]}")
                        csv_data = rebuilt_csv
                    else:
                        print(f"[DEBUG STAGE 4] rebuild_csv_from_json_curves returned None")
                else:
                    print(f"[DEBUG STAGE 4] No marker_facts found in result_json or result_json is None")
                    if result_json:
                        print(f"[DEBUG STAGE 4] result_json keys: {result_json.keys()}")
            except Exception as e:
                print(f"[DEBUG STAGE 4] Could not rebuild CSV from JSON: {e}, using extracted CSV")
                import traceback
                traceback.print_exc()
            
            if csv_data and csv_data != "None" and not csv_data.startswith("Here is the"):
                # Store in a module-level variable for reuse if Stage 5 fails
                stage_context["_saved_csv"] = csv_data
                csv_backup_path = output_out + "_data_stage4"
                with open(csv_backup_path, "w", encoding="utf-8") as f:
                    f.write(csv_data)
                print(f"CSV data successfully saved from EXTRACT_STAGE_4")
        except Exception as e:
            print(f"Warning: Could not save CSV from EXTRACT_STAGE_4: {e}")
    
    # Use stage confidence if abort provided a confidence, else default
    stage_confidence = abort_confidence if abort_confidence is not None else 0.7
    tracker.complete_stage(stage_name, result_text, confidence=stage_confidence, 
                          execution_time_ms=stage_time, facts=facts)

    # REQUIRED: after each stage completes, dump full accumulated facts to console
    # as an immutable snapshot (timeline semantics; no overwriting).
    stage_dump = _format_stage_facts_dump(stage_index + 1, accumulated_facts)
    console_timeline += stage_dump
    print(stage_dump)
    
    # Save real-time progress update for web UI
    save_stage_update(version_dir, stage_name, stage_index + 1, len(EXTRACT_STAGES), 
                     accumulated_facts, stage_time, console_output=console_timeline)

    # Early stop on abort
    if abort:
        reason_display = abort_reason or "Abort signaled by prompt"
        tracker.fail_stage(stage_name, f"ABORT: {reason_display}")

        # Write legacy data file for compatibility
        with open(output_out + "_data", "w", encoding="utf-8") as file:
            file.write("None")

        # Write abort meta information
        abort_meta = {
            "stage": stage_name,
            "stage_index": stage_index,
            "reason": reason_display,
            "confidence": abort_confidence,
            "accumulated_facts": accumulated_facts,
        }
        try:
            with open(output_out + "_abort", "w", encoding="utf-8") as f:
                json.dump(abort_meta, f, indent=2)
        except Exception:
            pass

        print(f"ABORTED at stage '{stage_name}' - {reason_display}")
        tracker.save_tracking_report(output_out + "_tracking")
        print(f"VERSION_DIR:{version_dir}")
        sys.exit(0)

# Mark extraction as complete
tracker.mark_complete()

# -----------------------------------------------------------------------------
# Stage outputs
# -----------------------------------------------------------------------------
# Extract CSV from final stage
final_stage = EXTRACT_STAGES[-1]
data_raw = stage_context.get(final_stage, "")
data_from_final = extract_csv_from_text(data_raw)

# Check if we have a saved CSV from Stage 4 (which is the data extraction stage)
stage4_csv = stage_context.get("_saved_csv", "")

# ALWAYS prefer Stage 4 CSV if it exists
# Stage 5 is validation only and should not regenerate CSV
if stage4_csv:
    data = stage4_csv
    print("Using CSV from EXTRACT_STAGE_4 (data extraction stage)")
else:
    # No Stage 4 backup, extract from final stage
    data = data_from_final
    print("Warning: No Stage 4 CSV found, using final stage output")

# Normalize CSV to wide format if it came back in long format
data = normalize_csv_to_wide(data)

# If still no valid CSV, try to recover from earlier stages
if not data or data.startswith("Here is the") or data == "accumulated_facts":
    print("Warning: Could not extract valid CSV data from any stage")
    data = ""

# For backward compatibility, also check if there's a CODE_PLOT stage result
code = ""
if hasattr(prompts_module, "CODE_PLOT"):
    # Generate matplotlib code to replot the data
    code_template = prompts_module.CODE_PLOT
    
    # Check which format variables the template uses
    if "{accumulated_facts}" in code_template:
        # New format: use accumulated facts as JSON
        accumulated_facts_str = json.dumps(accumulated_facts, indent=2)
        code_prompt = code_template.format(
            accumulated_facts=accumulated_facts_str,
            replot_path=replot_plot
        )
    else:
        # Old format: use data_context (CSV only)
        code_prompt = code_template.format(
            data_context=data,
            replot_path=replot_plot
        )
    
    messages = create_Q_1p([[base64_image, code_prompt]])
    conversation_log.extend(messages)
    messages, code = prompt_mistral(client, messages)
    code = clean_code_response(code)
    conversation_log.append({"role": "assistant", "content": code})

with open(output_out + "_data", "w", encoding="utf-8") as file:
    file.write(data)

# Save a clean CSV file with user-friendly naming
if data and data != "None" and not data.startswith("Here is the"):
    clean_csv_path = os.path.join(version_dir, f"{base_name}_extracted.csv")
    with open(clean_csv_path, "w", encoding="utf-8") as file:
        file.write(data)
    print(f"CSV saved to: {clean_csv_path}")

# -----------------------------------------------------------------------------
# Code execution and repair loop
# -----------------------------------------------------------------------------
error_output = None
if code:
    print("Replotting with extracted data... ", end="", flush=True)
    try:
        exec(code)
        print("FINISHED")
    except Exception:
        error_output = traceback.format_exc()
else:
    print("No CODE_PLOT prompt found, skipping replot generation")
    error_output = "SKIP_REPLOT"

# If code executed but didn't write the expected file, force a repair pass.
if code and not error_output and not os.path.exists(replot_plot):
    error_output = (
        "Replot output file was not created at the required path.\n"
        f"Expected replot_path: {replot_plot}\n"
        "Your code MUST save the image to that exact path via plt.savefig(replot_path, ...).\n"
        "Do not invent output directories or alternate filenames."
    )

if error_output and error_output != "SKIP_REPLOT":
    print("ERROR in replot code, fixing error...")
    if hasattr(prompts_module, "CODE_FIX"):
        fix_prompt = prompts_module.CODE_FIX
    else:
        fix_prompt = "Fix the code above. Respond with corrected code only."

    # Always remind the model of the required save path.
    fix_prompt = (
        fix_prompt
        + f"\n\nCRITICAL: The corrected code MUST save the plot to EXACTLY this path: {replot_plot}\n"
        + "Use os.makedirs(os.path.dirname(replot_path), exist_ok=True) and plt.savefig(replot_path, ...)."
    )
    repair_messages = conversation_log + [{"role": "user", "content": error_output + fix_prompt}]
    _, code = prompt_mistral(client, repair_messages)
    code = clean_code_response(code)
    try:
        exec(code)
        print("SUCCESS, error fixed")
        error_output = None
    except Exception:
        error_output = traceback.format_exc()

if error_output and error_output != "SKIP_REPLOT":
    print("FAILED - need to redo {input_plot}")
    print(error_output)

if code:
    with open(output_out + "_code", "w", encoding="utf-8") as file:
        file.write(code)
    with open(output_out + "_conversation", "a", encoding="utf-8") as file:
        json.dump(conversation_log + [{"role": "assistant", "content": code.replace("\n", "\\n")}], file)

# -----------------------------------------------------------------------------
# Validation (uses prompts from the loaded prompt set)
# -----------------------------------------------------------------------------

# Only validate if we successfully generated a replot
if code and error_output != "SKIP_REPLOT" and not error_output:
    comparison_original = api_image_path if original_ext == ".svg" else input_plot
    stacked = stack_images_vertically(comparison_original, replot_plot, "yes", version_dir, prompt_name, version_num)
else:
    stacked = None
    print("Skipping validation (no replot generated)")

if stacked:
    print("Comparing source and replot... ", end="", flush=True)
    wrong = False
    wrong_why = ""
    validation_details = {}

    def run_validation(prompt_text):
        msg = create_Q_1p([[encode_image(stacked), prompt_text]])
        _, validate_resp = prompt_mistral(client, msg)
        return validate_resp

    # Get comparison prompts from the loaded prompts module
    compare_x = getattr(prompts_module, 'COMPARE_X', 'Do the X axes match?')
    compare_y = getattr(prompts_module, 'COMPARE_Y', 'Do the Y axes match?')
    compare_number = getattr(prompts_module, 'COMPARE_NUMBER', 'Do the number of points match?')
    compare_trend = getattr(prompts_module, 'COMPARE_TREND', 'Does the trend match?')

    validate_x = run_validation(compare_x)
    validation_details['x_axis'] = validate_x
    print(f"\n\nAxis x (result: {validate_x})")
    if "no" in validate_x.lower().strip()[:10]:
        wrong = True
        wrong_why += "X; "

    validate_y = run_validation(compare_y)
    validation_details['y_axis'] = validate_y
    print(f"Axis y (result: {validate_y})")
    if "no" in validate_y.lower().strip()[:10]:
        wrong = True
        wrong_why += "Y; "

    validate_n = run_validation(compare_number)
    validation_details['num_points'] = validate_n
    print(f"Points n (result: {validate_n})")
    if "no" in validate_n.lower().strip()[:10]:
        wrong = True
        wrong_why += "N; "

    validate_t = run_validation(compare_trend)
    validation_details['trend'] = validate_t
    print(f"Trends (result: {validate_t})")
    if "no" in validate_t.lower().strip()[:10]:
        wrong = True
        wrong_why += "T"

    with open(output_out + "_validate", "w", encoding="utf-8") as file:
        if wrong:
            file.write("no")
        else:
            file.write("yes")
    if wrong:
        with open(output_out + "_validate_why", "w", encoding="utf-8") as file:
            file.write(wrong_why)
    result_flag = "no" if wrong else "yes"
    
    # Track validation results
    tracker.set_validation_result(result_flag, validation_details)
    print(f"\nFINISHED (result: {result_flag})")

    print("Stacking original and replotted images for comparison... ", end="", flush=True)
    stack_images_vertically(comparison_original, replot_plot, "no" if wrong else "yes", version_dir, prompt_name, version_num)
    print("FINISHED")
else:
    print("Skipping visual validation (comparison image could not be generated)")
    with open(output_out + "_validate", "w", encoding="utf-8") as file:
        file.write("skipped")
    tracker.set_validation_result("skipped")

if original_ext == ".svg" and api_image_path != input_plot:
    try:
        os.unlink(api_image_path)
    except Exception:
        pass

# Save tracking report
tracker_report_path = output_out + "_tracking"
tracker.save_tracking_report(tracker_report_path)
tracker.print_summary()

# Save final completion status to progress file
save_stage_update(version_dir, "COMPLETE", len(EXTRACT_STAGES), len(EXTRACT_STAGES), 
                 accumulated_facts, 0, console_output="Extraction completed successfully")

print(f"VERSION_DIR:{version_dir}")
print("\n\n")
