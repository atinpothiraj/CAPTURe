import os
from openai import AzureOpenAI
from dotenv import load_dotenv # pip install python-dotenv
import base64
from tenacity import retry, wait_exponential, stop_after_attempt

import transformers
import torch
import json

client = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_API_ENDPOINT"), 
  api_key=os.getenv("AZURE_API_KEY"),  
  api_version="2024-02-01"
)

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(8))
def get_answer(image_path, question, temperature=1.0):
    # Getting the base64 string
    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-4o", # "deployment_name".
        messages=[
            {"role": "user", "content": [
                {
                "type": "text",
                "text": question
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]}
            ]
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

def safe_string_to_int(s):
    try:
        return int(s)
    except ValueError:
        return -1

def extract_num(answer: str):
    messages = [
        {"role": "system", "content": "You are an answer extractor. When given someone's answer to some question, you will only extract their final number answer and will respond with just the number. If there is no exact number answer, respond with -1"},
        {"role": "user", "content": answer},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    return safe_string_to_int(outputs[0]["generated_text"][-1]["content"])


# SYNTHETIC DATASET
# Load the JSON data
json_path = "../synthetic_metadata.json"

with open(json_path, 'r') as f:
    input_data = json.load(f)

# Output JSON file path
output_json_path = "unocc_results/gpt_syn_results.json"

# Load processed data if output file exists
if os.path.exists(output_json_path):
    with open(output_json_path, 'r') as f:
        processed_data = json.load(f)
else:
    processed_data = []

# Build a set of processed image files to avoid duplicates
processed_image_files = set(entry['image_file'] for entry in processed_data)
# Iterate over each entry in the JSON
for idx, entry in enumerate(input_data):
    num = -1
    temperature = 1.0
    image_path = entry['image_file']

    if image_path in processed_image_files:
        # Skip already processed entries
        continue
    print(image_path)

    occluded_image_path = "../generate_synthetic/unoccluded_dataset/" + image_path

    question = f"Your task is to count objects in the image. First, state what the pattern is, then give your final count."
    print(question)
    answer = get_answer(occluded_image_path, question, temperature=temperature)
    num = extract_num(answer)

    # Update entry with the result
    entry_with_result = entry.copy()
    entry_with_result['answer'] = answer
    entry_with_result['results'] = num
    # Append to processed data
    processed_data.append(entry_with_result)
    processed_image_files.add(image_path)

    # Save the updated processed data back to the file after each entry
    with open(output_json_path, 'w') as f:
        json.dump(processed_data, f, indent=4)

print(f"Updated JSON file with GPT results saved to {output_json_path}.")


# REAL DATASET
# Load the JSON data
json_path = "../real_metadata.json"
with open(json_path, 'r') as f:
    input_data = json.load(f)

# Output JSON file path
output_json_path = "unocc_results/gpt_real_results.json"

# Load processed data if output file exists
if os.path.exists(output_json_path):
    with open(output_json_path, 'r') as f:
        processed_data = json.load(f)
else:
    processed_data = []

# Build a set of processed image files to avoid duplicates
processed_image_files = set(entry['image_file'] for entry in processed_data)

# Iterate over each entry in the JSON
for idx, entry in enumerate(input_data):
    num = -1
    temperature = 1.0
    image_path = entry['image_file']

    if image_path in processed_image_files:
        # Skip already processed entries
        continue
    print(image_path)

    occluded_image_path = "../unocc_real_dataset/" + image_path
    
    question = f"Count the exact number of {entry['object']} in the image. Assume the pattern of {entry['object']} continues behind any black box. Provide the total number of {entry['object']} as if the black box were not there."
    
    print(question)
    answer = get_answer(occluded_image_path, question, temperature=temperature)
    num = extract_num(answer)

    # Update entry with the result
    entry_with_result = entry.copy()
    entry_with_result['answer'] = answer
    entry_with_result['results'] = num
    # Append to processed data
    processed_data.append(entry_with_result)
    processed_image_files.add(image_path)

    # Save the updated processed data back to the file after each entry
    with open(output_json_path, 'w') as f:
        json.dump(processed_data, f, indent=4)

print(f"Updated JSON file with GPT results saved to {output_json_path}.")