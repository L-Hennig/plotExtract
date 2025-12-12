import base64, sys, re, os, requests, json, traceback, cv2
import numpy as np
import matplotlib.pyplot as plt

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
api_key = os.getenv("API_KEY_1")

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

def stack_images_vertically(image1_path, image2_path, border_color, output_dir, prompt_name, border_size=30):
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    if img1 is None or img2 is None:
        print("Error: One or both image paths are invalid.")
        sys.exit(1)

    # Get the width of both images to determine the new size
    width = max(img1.shape[1], img2.shape[1])

    # Resize images to have the same width (if needed)
    img1_resized = cv2.resize(img1, (width, int(img1.shape[0] * width / img1.shape[1])))
    img2_resized = cv2.resize(img2, (width, int(img2.shape[0] * width / img2.shape[1])))

    # Stack images vertically
    combined_image = np.vstack((img1_resized, img2_resized))

    # Add a border around the combined image
    if "yes" in border_color.lower():
        color = (0, 255, 0)  # Green border
    elif "no" in border_color.lower():
        color = (0, 0, 255)  # Red border
    else:
        print("Invalid border color input. Use 'yes' for green or 'no' for red.")
        sys.exit(1)

    combined_image_with_border = cv2.copyMakeBorder(
        combined_image,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=color
    )

    # Create the output filename in the same folder as the input image
    base_name = os.path.splitext(os.path.basename(image1_path))[0]
    output_filename = os.path.join(output_dir, f"comparison_{base_name}.{prompt_name}.png")

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
  response = client.chat.complete(
        model="mistral-large-2512",
        messages=Q,
        max_tokens=4096,
        temperature=0,
    )
  #print(message)
  # Changed from "return(Q,message.content[0].text)" to the below line
  return Q, response.choices[0].message.content

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

pngjpg = os.path.splitext(input_plot)[-1].lstrip('.')
if pngjpg == 'jpg':
  pngjpg = 'jpeg'
base64_image = encode_image(input_plot)
# Include prompt file name (shortened) in output filenames
prompt_name = os.path.splitext(os.path.basename(prompt_file))[0].replace('prompt_', 'p')

# Get base name and directory info
image_filename = os.path.basename(input_plot)
base_name = os.path.splitext(image_filename)[0]  # e.g., 'A-1'
ext = os.path.splitext(image_filename)[1]  # e.g., '.png'

# Find next version number
def get_next_version(parent_dir, base_name, prompt_name):
    """Find the next available version number for this image+prompt combination."""
    version = 1
    while True:
        folder_name = f"{base_name}.{prompt_name}.v{version}"
        folder_path = os.path.join(parent_dir, folder_name)
        if not os.path.exists(folder_path):
            return version, folder_path
        version += 1

version_num, version_dir = get_next_version(input_dir, base_name, prompt_name)
os.makedirs(version_dir, exist_ok=True)

# Set output paths inside the version folder
output_out = os.path.join(version_dir, f"{image_filename}.{prompt_name}.mistral.out")
replot_plot = os.path.join(version_dir, f"{base_name}-replot.{prompt_name}{ext}")

print(f"Input plot: {input_plot}")
print(f"Using prompt: {prompt_name}")
print(f"Output folder: {version_dir} (version {version_num})")


QQ = create_Q_1p([[base64_image, prompts['extract']]])

print("Extracting data... ", end = '', flush=True)
QQ, data = prompt_mistral(QQ)
QQ.append({'role': 'assistant', 'content': data})
with open(output_out+'_data', 'w') as file:
  file.write(data)
print(f"FINISHED")

# Loads code prompt from prompt file
code_prompt = prompts['code_plot'].format(replot_plot=replot_plot, data=data)

if 'none' in data.lower():
  print(f"NO DATA EXTRACTED")
  exit()

print("Generating replot code... ", end = '', flush=True)
QQ.append({'role': 'user', 'content': code_prompt})
QQ, code = prompt_mistral(QQ)
print(f"FINISHED")

error_output = None
print("Replotting with extracted data... ", end = '', flush=True)
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



with open(output_out+'_code', 'w') as file:
  file.write(code)
with open(output_out+'_conversation', 'a') as file:
  QQ.append({"role": "assistant", "content": code.replace("\n", "\\n")})
  json.dump(QQ, file)

stacked = stack_images_vertically(input_plot, replot_plot, "yes", version_dir, prompt_name, 0)

print("Comparing source and replot... ", end = '', flush=True)
wrong = False
wrong_why = ""

QQ = create_Q_1p([[encode_image(stacked), prompts['compare_x']]])
QQ, validate = prompt_mistral(QQ)
print(f"\n\nAxis x (result: {validate})")
if 'no' in validate.lower().strip()[:10]:
  wrong = True
  wrong_why += "X; "
with open(output_out+'_conversation', 'a') as file:
  QQ.append({"role": "assistant", "content": validate.replace("\n", "\\n")})
  json.dump(QQ, file)

QQ = create_Q_1p([[encode_image(stacked), prompts['compare_y']]])
QQ, validate = prompt_mistral(QQ)
print(f"Axis y (result: {validate})")
if 'no' in validate.lower().strip()[:10]:
  wrong = True
  wrong_why += "Y; "
with open(output_out+'_conversation', 'a') as file:
  QQ.append({"role": "assistant", "content": validate.replace("\n", "\\n")})
  json.dump(QQ, file)

QQ = create_Q_1p([[encode_image(stacked), prompts['compare_number']]])
QQ, validate = prompt_mistral(QQ)
print(f"Points n (result: {validate})")
if 'no' in validate.lower().strip()[:10]:
  wrong = True
  wrong_why += "N; "
with open(output_out+'_conversation', 'a') as file:
  QQ.append({"role": "assistant", "content": validate.replace("\n", "\\n")})
  json.dump(QQ, file)

QQ = create_Q_1p([[encode_image(stacked), prompts['compare_trend']]])
QQ, validate = prompt_mistral(QQ)
print(f"Trends (result: {validate})")
if 'no' in validate.lower().strip()[:10]:
  wrong = True
  wrong_why += "T"
with open(output_out+'_conversation', 'a') as file:
  QQ.append({"role": "assistant", "content": validate.replace("\n", "\\n")})
  json.dump(QQ, file)

with open(output_out+'_validate', 'w') as file:
  if wrong:
    validate = 'no'
    file.write(validate)
  else:
    validate = 'yes'
    file.write(validate)
if wrong:
  with open(output_out+'_validate_why', 'w') as file:
    file.write(wrong_why)
print(f"\nFINISHED (result: {validate})")

print("Stacking original and replotted images for comparison... ", end = '', flush=True)
stack_images_vertically(input_plot, replot_plot, validate, version_dir, prompt_name)
print(f"FINISHED")

# Print the version directory for use by calling scripts
print(f"VERSION_DIR:{version_dir}")
print("\n\n")
