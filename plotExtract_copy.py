import base64, sys, re, os, requests, json, traceback, cv2
import numpy as np
import matplotlib.pyplot as plt
import anthropic

######## 
from mistralai import Mistral


######## This allows the API key to be stored in a .env file for security
from dotenv import load_dotenv
load_dotenv(override=True)

if len(sys.argv) < 2:
  print("Usage: python plotExtract.py <path_to_plot_image>\nError: Missing required argument. Please provide the path to the plot image.")
  sys.exit(1)

api_key = os.getenv("API_KEY_1")

input_plot = sys.argv[1]
replot_plot = input_plot.replace(".png", "-replot.png").replace(".jpg", "-replot.jpg")

prompts = {'extract': """The image depicts a figure from a research paper. Please carefully analyze the figure and extract the horizontal ("x") and vertical ("y") coordinates of each individual data point from each curve. Pay extremely close attention to the axes:
- Identify the tick marks and labels on both axes, and use them to determine the numeric values for every point.
- Read the labels of both axes (the text or symbols denoting what each axis represents).

Data Reporting Requirements:
1. Separate each curve’s data so that every curve has its own two columns of x and y.
2. The first row of your output must contain the axis labels (for example, "x-axis label" for the x-column and "y-axis label (Curve X)" for the y-column). Include the curve label in the y-axis header if it’s available.
3. Subsequent rows must present the extracted numeric data in comma-separated values (CSV) format, with:
   - No additional words or commentary.
   - Exactly one row per data point in each curve.
   - For multiple curves, produce additional pairs of columns (two columns per curve: one for x-values, one for y-values).
4. If, for any reason, you are unable to extract the data, respond with only the word "None".
5. Make sure that both axis are labeled and there are individual values assigned to each tickmark, and that they increase in a logical monotonius fashion. If they are not it may be impossible to extract the data. In that case respond with only the word "None".

Key Instructions:
- Look at each point individually; do not assume a pattern or formula. Each data point must be read precisely from its plotted location.
- Verify the numerical axis labels carefully so the extracted data is as accurate as possible.
- Do not include any explanations or text other than the required CSV table.

Output Format Example (if you have two curves):
x-axis label,y-axis label (Curve 1),x-axis label,y-axis label (Curve 2)
0.1,0.5,0.1,0.75
0.2,0.52,0.2,0.77
...

(Or "None" if you cannot extract the data.)

Remember: The sole output should be either the CSV table (with all columns for the curves) or "None". Nothing else.
""",
'code_fix': f'The text above is an error produced by your code, please fix the code so that this error does not appear. Repeat the whole code and only the code so that your whole response can be directly copied and executed. Do not explain and do not say anything else, respond with just the code.',
'compare_x': 'You are provided with two images of research plots extracted from academic papers. Do these two plots have the same x-axis (horizontal)? Do they have the same ranges, labels, etc.? Answer with a single word, "yes" or "no" only."',
'compare_y': 'You are provided with two images of research plots extracted from academic papers. Do these two plots have the same y-axis (vertical)? Do they have the same ranges, labels, etc.? Answer with a single word, "yes" or "no" only."',
'compare_number': 'You are provided with two images of research plots extracted from academic papers. Do these two plots have the same number of points (for point plots)? Do the curves look like they connect the same amount of pointis (for line plots)?. Answer with a single word, "yes" or "no" only."',
'compare_trend': 'You are provided with two images of research plots extracted from academic papers. Do these sets of points or curves on these two plots represent the same trends? Do they follow the same patterns? Are points distributed in the same way? Answer with a single word, "yes" or "no" only."'}

##### Changed from anthropic.Anthropic to Mistral
client = Mistral(api_key=api_key)

def stack_images_vertically(image1_path, image2_path, border_color, border_size=30):
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

    # Create the output filename
    output_filename = "comparison_" + os.path.basename(image1_path)

    # Save the combined image with border
    cv2.imwrite(output_filename, combined_image_with_border)
    return(output_filename)

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def prompt_claude(Q):
  #print(Q)
  ########### Changed from message = client.messages.create to response = client.chat.completions.create to impliment change from anthropic to mistral
  ########### Also changed max tokens from 2048 to 500 and model from claude-3-5-sonnet-20241022 to pixtral-large-latest
  ########### Apparently max tokens is 4096
  response = client.chat.completions.create(
    model = "pixtral-large-latest",
    max_tokens = 500,
    temperature = 0,
    messages = Q
  )
  #print(message)
  return(Q,message.content[0].text)

def create_Q_2p(convo):
  Q = []
  for ic,c in enumerate(convo):
    if ic%2 == 0:
      role = 'user'
    else:
      role = 'assistant'
    if isinstance(c,list):
      Q.append({'role': role, 'content': [{"type": "image","source": {"type": "base64","media_type": "image/"+pngjpg,"data": c[0]}},{"type": "image","source": {"type": "base64","media_type": "image/"+pngjpg,"data": c[1]}},{'type': 'text', 'text': c[2]}]})
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
      Q.append({'role': role, 'content': [{"type": "image","source": {"type": "base64","media_type": "image/"+pngjpg,"data": c[0]},},{'type': 'text', 'text': c[1]}]})
    else:
      Q.append({'role': role, 'content': c})
  return Q

pngjpg = os.path.splitext(input_plot)[-1].lstrip('.')
if pngjpg == 'jpg':
  pngjpg = 'jpeg'
base64_image = encode_image(input_plot)
output_out = input_plot+'.claude.out'
print(f"Input plot: {input_plot}")


QQ = create_Q_1p([[base64_image, prompts['extract']]])

print("Extracting data... ", end = '', flush=True)
QQ, data = prompt_claude(QQ)
QQ.append({'role': 'assistant', 'content': data})
with open(output_out+'_data', 'w') as file:
  file.write(data)
print(f"FINISHED")

code_prompt = {'code_plot': f'Please analyze the figure and create a python code that will reproduce the plot exactly, including colors, line types, point shapes, axis labels, axis ranges, etc. Save the plot as a file "{replot_plot}" only and do not show it. Respond with the code only so that it can be directly copied and executed.\n\nUse the following data on the plot: {data}'}

if 'none' in data.lower():
  print(f"NO DATA EXTRACTED")
  exit()

print("Generating replot code... ", end = '', flush=True)
QQ.append({'role': 'user', 'content': code_prompt['code_plot']})
QQ, code = prompt_claude(QQ)
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
  QQ, code = prompt_claude(QQ)
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
  QQ, code = prompt_claude(QQ)
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

stacked = stack_images_vertically(input_plot, replot_plot, "yes", 0)

print("Comparing source and replot... ", end = '', flush=True)
wrong = False
wrong_why = ""

QQ = create_Q_1p([[encode_image(stacked), prompts['compare_x']]])
QQ, validate = prompt_claude(QQ)
print(f"\n\nAxis x (result: {validate})")
if 'no' in validate.lower().strip()[:10]:
  wrong = True
  wrong_why += "X; "
with open(output_out+'_conversation', 'a') as file:
  QQ.append({"role": "assistant", "content": validate.replace("\n", "\\n")})
  json.dump(QQ, file)

QQ = create_Q_1p([[encode_image(stacked), prompts['compare_y']]])
QQ, validate = prompt_claude(QQ)
print(f"Axis y (result: {validate})")
if 'no' in validate.lower().strip()[:10]:
  wrong = True
  wrong_why += "Y; "
with open(output_out+'_conversation', 'a') as file:
  QQ.append({"role": "assistant", "content": validate.replace("\n", "\\n")})
  json.dump(QQ, file)

QQ = create_Q_1p([[encode_image(stacked), prompts['compare_number']]])
QQ, validate = prompt_claude(QQ)
print(f"Points n (result: {validate})")
if 'no' in validate.lower().strip()[:10]:
  wrong = True
  wrong_why += "N; "
with open(output_out+'_conversation', 'a') as file:
  QQ.append({"role": "assistant", "content": validate.replace("\n", "\\n")})
  json.dump(QQ, file)

QQ = create_Q_1p([[encode_image(stacked), prompts['compare_trend']]])
QQ, validate = prompt_claude(QQ)
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
stack_images_vertically(input_plot, replot_plot, validate)
print(f"FINISHED")
print("\n\n")
