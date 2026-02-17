import base64, sys, re, os, requests, json, traceback, cv2, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as mpl_figure

# The below library is required to import the prompts from a separate file
import importlib.util

# Changed from "import Anthropic" to the below line
from mistralai import Mistral


# This allows the API key to be stored in a .env file for security
from dotenv import load_dotenv
load_dotenv(override=True)

if len(sys.argv) < 3:
  print("Usage: python plotExtract.py <path_to_plot_image> <prompt_file> \nError: Missing required argument. Please provide the path to the plot image.")
  sys.exit(1)

# Loads API key from .env file
def _select_mistral_api_key() -> str:
  key = (os.getenv("PLOTEXTRACT_LLM_KEY") or "").strip() or "4"
  if key == "1":
    return os.getenv("API_KEY_1") or ""
  if key == "3":
    return os.getenv("API_KEY_3") or os.getenv("API_KEY_1") or ""
  if key == "4":
    return os.getenv("API_KEY_4") or os.getenv("API_KEY_3") or os.getenv("API_KEY_1") or ""
  return os.getenv("API_KEY_4") or os.getenv("API_KEY_3") or os.getenv("API_KEY_1") or ""

api_key = _select_mistral_api_key()
if not api_key:
  raise RuntimeError("Missing Mistral API key. Set API_KEY_4 (preferred), or API_KEY_3, or API_KEY_1")

input_plot = sys.argv[1]
input_dir = os.path.dirname(input_plot)

# The code below loads the prompts from a separate file
prompt_file = sys.argv[2]
spec = importlib.util.spec_from_file_location("prompts_module", prompt_file)
prompts_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prompts_module)
prompts = prompts_module.prompts


# Changed from anthropic.Anthropic to Mistral
client = Mistral(api_key=api_key)

def stack_images_vertically(image1_path, image2_path, border_color, output_dir, prompt_name, version_num, tag_suffix: str = "", border_size=30):
    """Stack original and replot images vertically with labels."""
    
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    if img1 is None or img2 is None:
        print(f"Error: Cannot read images. img1={image1_path}, img2={image2_path}")
        return None

    # Get the width of both images to determine the new size
    width = max(img1.shape[1], img2.shape[1])

    # Resize images to have the same width (if needed)
    img1_resized = cv2.resize(img1, (width, int(img1.shape[0] * width / img1.shape[1])))
    img2_resized = cv2.resize(img2, (width, int(img2.shape[0] * width / img2.shape[1])))

    # Add labels to each image - larger font and height
    label_height = 60
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.8
    font_thickness = 3
    
    # Create label bars
    label1 = np.ones((label_height, width, 3), dtype=np.uint8) * 255  # White background
    label2 = np.ones((label_height, width, 3), dtype=np.uint8) * 255
    
    # Add text to labels (vertically centered)
    cv2.putText(label1, "Original", (15, 45), font, font_scale, (0, 0, 0), font_thickness)
    cv2.putText(label2, "Extracted (Re-plotted)", (15, 45), font, font_scale, (0, 0, 0), font_thickness)
    
    # Stack: label1 + img1 + label2 + img2
    combined_image = np.vstack((label1, img1_resized, label2, img2_resized))

    # Add a border around the combined image
    if "yes" in border_color.lower():
        color = (0, 255, 0)  # Green border
    elif "no" in border_color.lower():
        color = (0, 0, 255)  # Red border
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
        value=color
    )

    # Create the output filename - use full original filename for uniqueness
    original_filename = os.path.basename(image1_path)
    # Remove extension and use underscores
    base_name = original_filename.rsplit('.', 1)[0] if '.' in original_filename else original_filename
    output_filename = os.path.join(output_dir, f"comparison_{base_name}.{prompt_name}.v{version_num}{tag_suffix}.png")

    # Save the combined image with border
    cv2.imwrite(output_filename, combined_image_with_border)
    
    return(output_filename)

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Changed from "prompt_claude" to "prompt_mistral"
# This change was made throughout the script
def prompt_mistral(Q):
  #print(Q)
  # Changed from "client.messages.create" to "client.chat.complete"
  # Changed model to an appropriate Mistral model
  # Increased max_tokens to the limit for Mistral
  # Allow overriding model + timeouts/retries via env vars.
  model = os.getenv("PLOTEXTRACT_MISTRAL_MODEL") or "mistral-large-2512"
  timeout_ms = None
  try:
    timeout_ms_env = os.getenv("PLOTEXTRACT_MISTRAL_TIMEOUT_MS")
    timeout_ms = int(timeout_ms_env) if timeout_ms_env else 240_000
  except Exception:
    timeout_ms = 240_000

  def _is_rate_limit_error(err: Exception) -> bool:
    msg = str(err).lower()
    return (
      "status 429" in msg
      or "rate limit" in msg
      or "rate_limited" in msg
      or '"code":"1300"' in msg
      or "too many requests" in msg
    )

  try:
    max_retries = int(os.getenv("PLOTEXTRACT_MISTRAL_MAX_RETRIES") or "6")
  except Exception:
    max_retries = 6
  try:
    base_sleep_s = float(os.getenv("PLOTEXTRACT_MISTRAL_RETRY_BASE_S") or "1.0")
  except Exception:
    base_sleep_s = 1.0
  try:
    max_sleep_s = float(os.getenv("PLOTEXTRACT_MISTRAL_RETRY_MAX_S") or "20.0")
  except Exception:
    max_sleep_s = 20.0

  last_err = None
  for attempt in range(max_retries + 1):
    try:
      response = client.chat.complete(
            model=model,
            messages=Q,
            max_tokens=4096,
            temperature=0,
            timeout_ms=timeout_ms,
        )
      return Q, response.choices[0].message.content
    except Exception as e:
      last_err = e
      if _is_rate_limit_error(e) and attempt < max_retries:
        sleep_s = min(max_sleep_s, base_sleep_s * (2 ** attempt))
        print(f"[WARN] Mistral rate limit (429). Retrying in {sleep_s:.1f}s (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
        time.sleep(sleep_s)
        continue
      raise

  # Should be unreachable, but keep for safety.
  raise last_err
  #print(message)
  # Changed from "return(Q,message.content[0].text)" to the below line

def create_Q_2p(convo):
  Q = []
  for ic,c in enumerate(convo):
    if ic%2 == 0:
      role = 'user'
    else:
      role = 'assistant'
    if isinstance(c,list):
      # Changed to support Mistral message format
      Q.append({
                'role': role,
                'content': [
                    {"type": "text", "text": c[2]},
                    {"type": "image_url", "image_url": {"url": f"data:image/{pngjpg};base64,{c[0]}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/{pngjpg};base64,{c[1]}"}}
                ]
            })
    else:
      Q.append({'role': role, 'content': c})
  return Q

def create_Q_1p(convo):
  Q = []
  for ic,c in enumerate(convo):
    if ic%2 == 0:
      role = 'user'
    else:
      role = 'assistant'
    if isinstance(c,list):
      # Changed to support Mistral message format
      Q.append({
                'role': role,
                'content': [
                    {"type": "text", "text": c[1]},
                    {"type": "image_url", "image_url": {"url": f"data:image/{pngjpg};base64,{c[0]}"}}
                ]
            })
    else:
      Q.append({'role': role, 'content': c})
  return Q

# Get extension and prepare for API
original_ext = os.path.splitext(input_plot)[-1].lower()  # e.g., '.png', '.svg'

# Set MIME type based on extension
api_image_path = input_plot
pngjpg = original_ext.lstrip('.')
if pngjpg == 'jpg':
    pngjpg = 'jpeg'
elif pngjpg == 'svg':
    # Mistral API doesn't support SVG - convert to PNG using Edge browser
    import subprocess as sp
    import tempfile
    temp_png = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
    abs_svg = os.path.abspath(input_plot).replace('\\', '/')
    
    # Create HTML wrapper that scales SVG to fill viewport
    html_content = f'''<!DOCTYPE html>
<html><head><style>
html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background: white; }}
img {{ width: 100%; height: 100%; object-fit: contain; }}
</style></head>
<body><img src="file:///{abs_svg}"></body></html>'''
    
    temp_html = tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w', encoding='utf-8')
    temp_html.write(html_content)
    temp_html.close()
    
    # Browser paths to try
    browsers = [
        r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
        r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
        'msedge',
        r'C:\Program Files\Google\Chrome\Application\chrome.exe',
        r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
        'chrome'
    ]
    
    converted = False
    for browser in browsers:
        try:
            # Screenshot the HTML wrapper which scales SVG to fill viewport
            sp.run([browser, '--headless', '--disable-gpu', 
                    f'--screenshot={temp_png}', '--window-size=1600,1200',
                    '--force-device-scale-factor=2',  # 2x scale for higher resolution
                    f'file:///{temp_html.name.replace(os.sep, "/")}'], 
                   capture_output=True, timeout=60)
            if os.path.exists(temp_png) and os.path.getsize(temp_png) > 0:
                api_image_path = temp_png
                pngjpg = 'png'
                converted = True
                print(f"Converted SVG to PNG (high resolution)")
                break
        except (FileNotFoundError, sp.TimeoutExpired, OSError):
            continue
    
    # Clean up temp HTML
    try:
        os.unlink(temp_html.name)
    except:
        pass
    
    if not converted:
        print("ERROR: Cannot convert SVG to PNG. Install Edge or Chrome.")
        sys.exit(1)

base64_image = encode_image(api_image_path)
# Include prompt file name (shortened) in output filenames
prompt_name = os.path.splitext(os.path.basename(prompt_file))[0].replace('prompt_', 'p')

# Get base name and directory info - include extension in naming for uniqueness
image_filename = os.path.basename(input_plot)
base_name = os.path.splitext(image_filename)[0]  # e.g., 'A-1'
ext = os.path.splitext(image_filename)[1]  # e.g., '.png' or '.svg'
# Use full filename (with extension) for folder naming to differentiate .png from .svg
name_for_folder = image_filename.replace('.', '_')  # e.g., 'A-1_png' or 'A-1_svg'

# Optional output tag (e.g. WebExtract runs append ".web" to folder + filenames)
output_tag = str(os.getenv('PLOTEXTRACT_OUTPUT_TAG', '') or '').strip()
if output_tag.startswith('.'):
  output_tag = output_tag[1:]
if output_tag and output_tag.lower() != 'web':
  output_tag = ''
tag_suffix = f".{output_tag}" if output_tag else ""

# Find next version number - uses name_for_folder to include extension
def get_next_version(parent_dir, name_for_folder, prompt_name):
  """Find the next available version number for this image+prompt combination.

  Version numbers are monotonic across output tags (e.g. .web), so WebExtract
  runs and regular runs share the same counter.
  """
  import re

  max_version = 0
  try:
    pat = re.compile(rf'^{re.escape(name_for_folder)}\.{re.escape(prompt_name)}\.v(\d+)(?:\.web)?$')
    if os.path.isdir(parent_dir):
      for item in os.listdir(parent_dir):
        m = pat.match(item)
        if not m:
          continue
        try:
          v = int(m.group(1))
        except Exception:
          continue
        if v > max_version:
          max_version = v
  except Exception:
    max_version = 0

  version = max_version + 1
  folder_name = f"{name_for_folder}.{prompt_name}.v{version}{tag_suffix}"
  folder_path = os.path.join(parent_dir, folder_name)
  return version, folder_path

version_num, version_dir = get_next_version(input_dir, name_for_folder, prompt_name)
os.makedirs(version_dir, exist_ok=True)

# Set output paths inside the version folder - include version in filenames
output_out = os.path.join(version_dir, f"{image_filename}.{prompt_name}.v{version_num}{tag_suffix}.mistral.out")
# Replot is always PNG (matplotlib output)
replot_plot = os.path.join(version_dir, f"{name_for_folder}-replot.{prompt_name}.v{version_num}{tag_suffix}.png")

print(f"Input plot: {input_plot}")
print(f"Using prompt: {prompt_name}")
print(f"Output folder: {version_dir} (version {version_num})")


QQ = create_Q_1p([[base64_image, prompts['extract']]])

print("Extracting data... ", end = '', flush=True)
QQ, data = prompt_mistral(QQ)
QQ.append({'role': 'assistant', 'content': data})

def _clean_extracted_csv_text(raw_text: str) -> str:
  """Keep only CSV-like lines and drop known empty/None-only rows.

  Some extractions include a trailing line like `None,,,,` which previously
  triggered a false-positive "NO DATA EXTRACTED" check.
  """
  if raw_text is None:
    return ""

  lines = str(raw_text).splitlines()
  cleaned = []
  in_code_fence = False
  for line in lines:
    s = line.strip()
    if s.startswith("```"):
      in_code_fence = not in_code_fence
      continue

    # Keep only CSV-like lines
    if ',' not in line:
      continue

    # Skip rows that are entirely empty or start with a lone 'None'
    if s == "" or s.lower() == "none" or s.lower().startswith("none,"):
      continue
    if s.replace(',', '').strip() == "":
      continue

    cleaned.append(line)

  return "\n".join(cleaned).strip() + "\n" if cleaned else ""


def _has_any_numeric_data(csv_text: str) -> bool:
  """Return True if CSV text contains at least one numeric value row."""
  if not csv_text:
    return False
  lines = csv_text.splitlines()
  if len(lines) < 2:
    return False

  numeric_count = 0
  for row in lines[1:]:
    for tok in row.split(','):
      t = tok.strip()
      if t == "" or t.lower() in ("none", "nan"):
        continue
      try:
        float(t)
        numeric_count += 1
        if numeric_count >= 2:
          return True
      except Exception:
        continue
  return False


cleaned_data = _clean_extracted_csv_text(data)
with open(output_out+'_data', 'w', encoding='utf-8') as file:
  file.write(cleaned_data if cleaned_data else (data or ""))
print(f"FINISHED")

# Loads code prompt from prompt file
code_prompt = prompts['code_plot'].format(replot_plot=replot_plot, data=data)

if not _has_any_numeric_data(cleaned_data):
  print(f"NO DATA EXTRACTED")
  # Still print VERSION_DIR so app.py knows where files are
  print(f"VERSION_DIR:{version_dir}")
  sys.exit(2)

print("Generating replot code... ", end = '', flush=True)
QQ.append({'role': 'user', 'content': code_prompt})
QQ, code = prompt_mistral(QQ)
print(f"FINISHED")

error_output = None
print("Replotting with extracted data... ", end = '', flush=True)

# Force consistent axis starts at save-time (applies even if the LLM code sets limits).
_orig_plt_savefig = plt.savefig
_orig_fig_savefig = mpl_figure.Figure.savefig

_plotextract_in_overlay_save = False

def _derive_overlay_paths_from_replot(path: str):
  p = str(path)
  if '-replot.' in p:
    full_p = p.replace('-replot.', '-replot_overlay_full.', 1)
    minmax_p = p.replace('-replot.', '-replot_overlay_minmax.', 1)
    return full_p, minmax_p
  if p.lower().endswith('.png'):
    stem = p[:-4]
    return stem + '_overlay_full.png', stem + '_overlay_minmax.png'
  return p + '_overlay_full', p + '_overlay_minmax'

def _derive_overlay_axes_paths_from_replot(path: str):
  p = str(path)
  if '-replot.' in p:
    return (
      p.replace('-replot.', '-replot_overlay_axes_full.', 1),
      p.replace('-replot.', '-replot_overlay_axes_minmax.', 1),
    )
  if p.lower().endswith('.png'):
    stem = p[:-4]
    return stem + '_overlay_axes_full.png', stem + '_overlay_axes_minmax.png'
  return p + '_overlay_axes_full', p + '_overlay_axes_minmax'

def _derive_overlay_curve_path_from_replot(path: str, idx: int, axis_mode: str):
  p = str(path)
  tag = f"-replot_overlay_curve_{idx}_{axis_mode}."
  if '-replot.' in p:
    return p.replace('-replot.', tag, 1)
  suffix = f"_overlay_curve_{idx}_{axis_mode}"
  if p.lower().endswith('.png'):
    return p[:-4] + suffix + '.png'
  return p + suffix

def _save_axes_and_curve_overlay_layers(fig, replot_path: str, save_kwargs=None):
  try:
    axes = getattr(fig, 'axes', []) or []
  except Exception:
    axes = []

  lines = []
  for ax in axes:
    try:
      for ln in ax.get_lines() or []:
        lines.append(ln)
    except Exception:
      continue

  if not lines:
    return

  vis_state = []
  for ln in lines:
    try:
      vis_state.append(bool(ln.get_visible()))
    except Exception:
      vis_state.append(True)

  axes_full, axes_minmax = _derive_overlay_axes_paths_from_replot(replot_path)
  try:
    # Axes-only layers.
    for ln in lines:
      try:
        ln.set_visible(False)
      except Exception:
        pass
    _save_transparent_replot_overlay(fig, axes_full, axis_mode='full', save_kwargs=save_kwargs)
    _save_transparent_replot_overlay(fig, axes_minmax, axis_mode='minmax', save_kwargs=save_kwargs)

    # Per-curve layers (1-based index to match UI table order).
    for i, ln in enumerate(lines, start=1):
      for other in lines:
        try:
          other.set_visible(other is ln)
        except Exception:
          pass
      out_full = _derive_overlay_curve_path_from_replot(replot_path, i, 'full')
      out_minmax = _derive_overlay_curve_path_from_replot(replot_path, i, 'minmax')
      _save_transparent_replot_overlay(fig, out_full, axis_mode='full', save_kwargs=save_kwargs)
      _save_transparent_replot_overlay(fig, out_minmax, axis_mode='minmax', save_kwargs=save_kwargs)
  finally:
    for ln, was_vis in zip(lines, vis_state):
      try:
        ln.set_visible(bool(was_vis))
      except Exception:
        pass

def _format_tick_val(v):
  try:
    # Keep labels compact; works for ints and floats.
    return f"{float(v):.6g}"
  except Exception:
    try:
      return str(v)
    except Exception:
      return ''

def _save_transparent_replot_overlay(fig, out_path: str, axis_mode: str = 'full', save_kwargs=None):
  """Save a transparent overlay PNG of the current figure.

  - axis_mode='full': keep axes as-is (but transparent background)
  - axis_mode='minmax': simplify ticks to only show min/max for each axis
  """
  import os
  try:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
  except Exception:
    pass

  # Snapshot + modify style.
  fig_patch_alpha = None
  fig_legend_state = []
  try:
    fig_patch_alpha = fig.patch.get_alpha()
    fig.patch.set_alpha(0.0)
  except Exception:
    fig_patch_alpha = None

  try:
    for lg in list(getattr(fig, 'legends', []) or []):
      try:
        vis = bool(lg.get_visible())
      except Exception:
        vis = True
      fig_legend_state.append((lg, vis))
      try:
        lg.set_visible(False)
      except Exception:
        pass
  except Exception:
    fig_legend_state = []

  per_ax_state = []
  try:
    axes = getattr(fig, 'axes', []) or []
  except Exception:
    axes = []

  for ax in axes:
    st = {
      'ax': ax,
      'patch_alpha': None,
      'lines': [],
      'collections': [],
      'legend': None,
      'legend_visible': None,
      'x_major_locator': None,
      'x_major_formatter': None,
      'x_minor_locator': None,
      'x_minor_formatter': None,
      'y_major_locator': None,
      'y_major_formatter': None,
      'y_minor_locator': None,
      'y_minor_formatter': None,
      'xticks': None,
      'yticks': None,
    }
    try:
      st['patch_alpha'] = ax.patch.get_alpha()
      ax.patch.set_alpha(0.0)
    except Exception:
      st['patch_alpha'] = None

    # Recolor lines to red.
    try:
      for ln in ax.get_lines() or []:
        try:
          st['lines'].append((ln, ln.get_color(), ln.get_alpha(), ln.get_linewidth()))
          ln.set_color('#ef4444')
          if ln.get_alpha() is None:
            ln.set_alpha(1.0)
        except Exception:
          continue
    except Exception:
      pass

    try:
      lg = ax.get_legend()
      if lg is not None:
        st['legend'] = lg
        try:
          st['legend_visible'] = bool(lg.get_visible())
        except Exception:
          st['legend_visible'] = True
        try:
          lg.set_visible(False)
        except Exception:
          pass
    except Exception:
      pass

    # Recolor common collection artists (scatter, etc.).
    try:
      for coll in getattr(ax, 'collections', []) or []:
        try:
          ec = coll.get_edgecolor()
        except Exception:
          ec = None
        try:
          fc = coll.get_facecolor()
        except Exception:
          fc = None
        st['collections'].append((coll, ec, fc))
        try:
          coll.set_edgecolor('#ef4444')
        except Exception:
          pass
        try:
          coll.set_facecolor('#ef4444')
        except Exception:
          pass
    except Exception:
      pass

    if axis_mode == 'minmax':
      try:
        st['x_major_locator'] = ax.xaxis.get_major_locator()
        st['x_major_formatter'] = ax.xaxis.get_major_formatter()
        st['x_minor_locator'] = ax.xaxis.get_minor_locator()
        st['x_minor_formatter'] = ax.xaxis.get_minor_formatter()
        st['y_major_locator'] = ax.yaxis.get_major_locator()
        st['y_major_formatter'] = ax.yaxis.get_major_formatter()
        st['y_minor_locator'] = ax.yaxis.get_minor_locator()
        st['y_minor_formatter'] = ax.yaxis.get_minor_formatter()

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        st['xticks'] = ax.get_xticks()
        st['yticks'] = ax.get_yticks()

        ax.set_xticks([xmin, xmax])
        ax.set_yticks([ymin, ymax])
        ax.set_xticklabels([_format_tick_val(xmin), _format_tick_val(xmax)])
        ax.set_yticklabels([_format_tick_val(ymin), _format_tick_val(ymax)])
        try:
          ax.minorticks_off()
        except Exception:
          pass
      except Exception:
        pass

    per_ax_state.append(st)

  dpi = None
  bbox_inches = None
  pad_inches = None
  try:
    sk = save_kwargs or {}
    dpi = sk.get('dpi', None)
    bbox_inches = sk.get('bbox_inches', None)
    pad_inches = sk.get('pad_inches', None)
  except Exception:
    dpi = None
    bbox_inches = None
    pad_inches = None

  if dpi is None:
    dpi = 300
  if bbox_inches is None:
    bbox_inches = 'tight'

  try:
    # Use the original Figure.savefig to bypass our patched wrapper.
    _orig_fig_savefig(
      fig,
      out_path,
      dpi=dpi,
      bbox_inches=bbox_inches,
      pad_inches=pad_inches,
      transparent=True,
      facecolor='none',
      edgecolor='none',
    )
  except Exception:
    # Best-effort only.
    pass
  finally:
    # Restore axes modifications.
    for st in per_ax_state:
      ax = st.get('ax')
      if not ax:
        continue
      try:
        if st.get('patch_alpha') is not None:
          ax.patch.set_alpha(st['patch_alpha'])
      except Exception:
        pass

      for (ln, c, a, lw) in st.get('lines', []) or []:
        try:
          ln.set_color(c)
          ln.set_alpha(a)
          ln.set_linewidth(lw)
        except Exception:
          continue

      for (coll, ec, fc) in st.get('collections', []) or []:
        try:
          if ec is not None:
            coll.set_edgecolor(ec)
        except Exception:
          pass
        try:
          if fc is not None:
            coll.set_facecolor(fc)
        except Exception:
          pass

      try:
        lg = st.get('legend')
        if lg is not None:
          lg.set_visible(bool(st.get('legend_visible')))
      except Exception:
        pass

      if axis_mode == 'minmax':
        try:
          if st.get('x_major_locator') is not None:
            ax.xaxis.set_major_locator(st['x_major_locator'])
          if st.get('x_major_formatter') is not None:
            ax.xaxis.set_major_formatter(st['x_major_formatter'])
          if st.get('x_minor_locator') is not None:
            ax.xaxis.set_minor_locator(st['x_minor_locator'])
          if st.get('x_minor_formatter') is not None:
            ax.xaxis.set_minor_formatter(st['x_minor_formatter'])
          if st.get('y_major_locator') is not None:
            ax.yaxis.set_major_locator(st['y_major_locator'])
          if st.get('y_major_formatter') is not None:
            ax.yaxis.set_major_formatter(st['y_major_formatter'])
          if st.get('y_minor_locator') is not None:
            ax.yaxis.set_minor_locator(st['y_minor_locator'])
          if st.get('y_minor_formatter') is not None:
            ax.yaxis.set_minor_formatter(st['y_minor_formatter'])
        except Exception:
          pass

    try:
      if fig_patch_alpha is not None:
        fig.patch.set_alpha(fig_patch_alpha)
    except Exception:
      pass

    for (lg, was_vis) in fig_legend_state:
      try:
        lg.set_visible(bool(was_vis))
      except Exception:
        pass

def _maybe_emit_replot_overlay_variants(fig, save_args, save_kwargs):
  global _plotextract_in_overlay_save
  if _plotextract_in_overlay_save:
    return

  # Determine the save path (best-effort).
  out_path = None
  try:
    if save_args and isinstance(save_args[0], (str, os.PathLike)):
      out_path = os.fspath(save_args[0])
  except Exception:
    out_path = None

  if not out_path:
    return
  try:
    if not str(out_path).lower().endswith('.png'):
      return
  except Exception:
    return

  try:
    if os.path.abspath(str(out_path)) != os.path.abspath(str(replot_plot)):
      return
  except Exception:
    # If abspath fails, fall back to string compare.
    if str(out_path) != str(replot_plot):
      return

  overlay_full, overlay_minmax = _derive_overlay_paths_from_replot(str(out_path))
  try:
    _plotextract_in_overlay_save = True
    _save_transparent_replot_overlay(fig, overlay_full, axis_mode='full', save_kwargs=save_kwargs)
    _save_transparent_replot_overlay(fig, overlay_minmax, axis_mode='minmax', save_kwargs=save_kwargs)
    _save_axes_and_curve_overlay_layers(fig, str(out_path), save_kwargs=save_kwargs)
  finally:
    _plotextract_in_overlay_save = False

def _apply_replot_axis_policy(fig):
  try:
    axes = getattr(fig, 'axes', []) or []
    for ax in axes:
      try:
        # Y: always start at 0.001 (works for log plots too)
        yscale = ax.get_yscale()
        if yscale == 'log':
          cur_bottom, cur_top = ax.get_ylim()
          top = cur_top if (cur_top is not None and cur_top > 0.001) else 0.01
          ax.set_ylim(0.001, top)
        else:
          ax.set_ylim(bottom=0.001)

        # X: start at 0 only for linear axes
        if ax.get_xscale() == 'linear':
          ax.set_xlim(left=0.0)
      except Exception:
        continue
  except Exception:
    pass

def _patched_plt_savefig(*args, **kwargs):
  try:
    _apply_replot_axis_policy(plt.gcf())
  except Exception:
    pass
  ret = _orig_plt_savefig(*args, **kwargs)
  try:
    _maybe_emit_replot_overlay_variants(plt.gcf(), args, kwargs)
  except Exception:
    pass
  return ret

def _patched_fig_savefig(self, *args, **kwargs):
  _apply_replot_axis_policy(self)
  ret = _orig_fig_savefig(self, *args, **kwargs)
  try:
    _maybe_emit_replot_overlay_variants(self, args, kwargs)
  except Exception:
    pass
  return ret

plt.savefig = _patched_plt_savefig
mpl_figure.Figure.savefig = _patched_fig_savefig

try:
  exec(code)
  print(f"FINISHED")
except Exception as e:
  error_output = traceback.format_exc()
if error_output:
  print(f"ERROR in replot code, fixing error...")
  QQ.append({'role': 'assistant', 'content': code})
  QQ.append({'role': 'user', 'content': error_output+prompts['code_fix']})
  QQ, code = prompt_mistral(QQ)
  try:
    exec(code)
    print(f"SUCCESS, error fixed")
    error_output = None
  except Exception as e:
    error_output = traceback.format_exc()

if error_output:
  print(f"ERROR in replot code, fixing error...")
  QQ.append({'role': 'assistant', 'content': code})
  QQ.append({'role': 'user', 'content': error_output+prompts['code_fix']})
  QQ, code = prompt_mistral(QQ)
  try:
    exec(code)
    print(f"SUCCESS, error fixed")
    error_output = None
  except Exception as e:
    error_output = traceback.format_exc()
    print(f"FAILED - need to redo {input_plot}")
    print(error_output)
    print("\n\n")



with open(output_out+'_code', 'w', encoding='utf-8') as file:
  file.write(code)
with open(output_out+'_conversation', 'a', encoding='utf-8') as file:
  QQ.append({"role": "assistant", "content": code.replace("\n", "\\n")})
  json.dump(QQ, file)

# For comparison, use the converted PNG if original was SVG (cv2 can't read SVG)
comparison_original = api_image_path if original_ext == '.svg' else input_plot
stacked = stack_images_vertically(comparison_original, replot_plot, "yes", version_dir, prompt_name, version_num, tag_suffix=tag_suffix)

# Only run validation comparisons if we have a stacked comparison image
if stacked:
    print("Comparing source and replot... ", end = '', flush=True)
    wrong = False
    wrong_why = ""

    QQ = create_Q_1p([[encode_image(stacked), prompts['compare_x']]])
    QQ, validate = prompt_mistral(QQ)
    print(f"\n\nAxis x (result: {validate})")
    if 'no' in validate.lower().strip()[:10]:
      wrong = True
      wrong_why += "X; "
    with open(output_out+'_conversation', 'a', encoding='utf-8') as file:
      QQ.append({"role": "assistant", "content": validate.replace("\n", "\\n")})
      json.dump(QQ, file)

    QQ = create_Q_1p([[encode_image(stacked), prompts['compare_y']]])
    QQ, validate = prompt_mistral(QQ)
    print(f"Axis y (result: {validate})")
    if 'no' in validate.lower().strip()[:10]:
      wrong = True
      wrong_why += "Y; "
    with open(output_out+'_conversation', 'a', encoding='utf-8') as file:
      QQ.append({"role": "assistant", "content": validate.replace("\n", "\\n")})
      json.dump(QQ, file)

    QQ = create_Q_1p([[encode_image(stacked), prompts['compare_number']]])
    QQ, validate = prompt_mistral(QQ)
    print(f"Points n (result: {validate})")
    if 'no' in validate.lower().strip()[:10]:
      wrong = True
      wrong_why += "N; "
    with open(output_out+'_conversation', 'a', encoding='utf-8') as file:
      QQ.append({"role": "assistant", "content": validate.replace("\n", "\\n")})
      json.dump(QQ, file)

    QQ = create_Q_1p([[encode_image(stacked), prompts['compare_trend']]])
    QQ, validate = prompt_mistral(QQ)
    print(f"Trends (result: {validate})")
    if 'no' in validate.lower().strip()[:10]:
      wrong = True
      wrong_why += "T"
    with open(output_out+'_conversation', 'a', encoding='utf-8') as file:
      QQ.append({"role": "assistant", "content": validate.replace("\n", "\\n")})
      json.dump(QQ, file)

    with open(output_out+'_validate', 'w', encoding='utf-8') as file:
      if wrong:
        validate = 'no'
        file.write(validate)
      else:
        validate = 'yes'
        file.write(validate)
    if wrong:
      with open(output_out+'_validate_why', 'w', encoding='utf-8') as file:
        file.write(wrong_why)
    print(f"\nFINISHED (result: {validate})")

    print("Stacking original and replotted images for comparison... ", end = '', flush=True)
    stack_images_vertically(comparison_original, replot_plot, validate, version_dir, prompt_name, version_num, tag_suffix=tag_suffix)
    print(f"FINISHED")
else:
    # Could not create comparison image
    print("Skipping visual validation (comparison image could not be generated)")
    with open(output_out+'_validate', 'w', encoding='utf-8') as file:
      file.write('skipped')

# Cleanup temp PNG from SVG conversion
if original_ext == '.svg' and api_image_path != input_plot:
    try:
        os.unlink(api_image_path)
    except:
        pass

# Print the version directory for use by calling scripts
print(f"VERSION_DIR:{version_dir}")
print("\n\n")
