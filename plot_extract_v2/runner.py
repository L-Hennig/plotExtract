import base64
import json
import os
import re
import sys
import traceback
import importlib
import cv2
import numpy as np
from dotenv import load_dotenv
from mistralai import Mistral

# Import article info schema
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'prompts'))
from article_info_schema import ARTICLE_INFO_SCHEMA, SCHEMA_CONSTRAINTS

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

    # Try JSON block with marker_facts.csv_output
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    if json_match:
        try:
            obj = json.loads(json_match.group(1))
            csv_val = obj.get("marker_facts", {}).get("csv_output")
            if csv_val:
                return csv_val.strip()
        except Exception:
            pass

    # Try fenced CSV block
    csv_match = re.search(r"```csv\s*(.*?)\s*```", text, re.S)
    if csv_match:
        return csv_match.group(1).strip()

    # Fallback: return raw text
    return text


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

# If article info was provided, add it as initial context
if article_info_json:
    stage_context["article_info"] = article_info_json

# -----------------------------------------------------------------------------
# Stage 1: extraction (and additional stages)
# -----------------------------------------------------------------------------
for stage_index, stage_name in enumerate(EXTRACT_STAGES):
    # Get the prompt text from the prompts module
    if not hasattr(prompts_module, stage_name):
        print(f"Error: Stage '{stage_name}' not found in prompts module")
        sys.exit(1)
    
    prompt_payload = getattr(prompts_module, stage_name)
    
    # Build stage-specific prompt, injecting previous outputs when needed
    if "{data_context}" in prompt_payload:
        # Start with article info if available
        accumulated_context = ""
        if article_info_json:
            accumulated_context = f"=== Article Information (structured facts) ===\n{article_info_json}\n\n"
        
        # Accumulate all previous stages' outputs
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

    # Early stop
    if str(result_text).strip() == "None":
        with open(output_out + "_data", "w", encoding="utf-8") as file:
            file.write("None")
        print("NO DATA EXTRACTED (stage returned None)")
        print(f"VERSION_DIR:{version_dir}")
        sys.exit(0)

# -----------------------------------------------------------------------------
# Stage outputs
# -----------------------------------------------------------------------------
# The final extraction stage should contain the CSV data
final_stage = EXTRACT_STAGES[-1]
data_raw = stage_context.get(final_stage, "")
data = extract_csv_from_text(data_raw)

# For backward compatibility, also check if there's a CODE_PLOT stage result
code = ""
if hasattr(prompts_module, "CODE_PLOT"):
    # Generate matplotlib code to replot the data
    code_prompt = prompts_module.CODE_PLOT.format(
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

if error_output and error_output != "SKIP_REPLOT":
    print("ERROR in replot code, fixing error...")
    if hasattr(prompts_module, "CODE_FIX"):
        fix_prompt = prompts_module.CODE_FIX
    else:
        fix_prompt = "Fix the code above. Respond with corrected code only."
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
    print(f"\n\nAxis x (result: {validate_x})")
    if "no" in validate_x.lower().strip()[:10]:
        wrong = True
        wrong_why += "X; "

    validate_y = run_validation(compare_y)
    print(f"Axis y (result: {validate_y})")
    if "no" in validate_y.lower().strip()[:10]:
        wrong = True
        wrong_why += "Y; "

    validate_n = run_validation(compare_number)
    print(f"Points n (result: {validate_n})")
    if "no" in validate_n.lower().strip()[:10]:
        wrong = True
        wrong_why += "N; "

    validate_t = run_validation(compare_trend)
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
    print(f"\nFINISHED (result: {result_flag})")

    print("Stacking original and replotted images for comparison... ", end="", flush=True)
    stack_images_vertically(comparison_original, replot_plot, "no" if wrong else "yes", version_dir, prompt_name, version_num)
    print("FINISHED")
else:
    print("Skipping visual validation (comparison image could not be generated)")
    with open(output_out + "_validate", "w", encoding="utf-8") as file:
        file.write("skipped")

if original_ext == ".svg" and api_image_path != input_plot:
    try:
        os.unlink(api_image_path)
    except Exception:
        pass

print(f"VERSION_DIR:{version_dir}")
print("\n\n")
