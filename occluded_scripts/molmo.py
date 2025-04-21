from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import json
import transformers
import torch
import os


# load the processor
processor = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

# load the model
model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

def get_answer(image_path, question, temperature=1.0):
    # process the image and text
    inputs = processor.process(
        images=[Image.open(image_path)],
        text=question
    )

    # move inputs to the correct device and make a batch of size 1
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=2000, stop_strings="<|endoftext|>", do_sample = True, temperature=temperature),
        tokenizer=processor.tokenizer
    )

    # only get generated tokens; decode them to text
    generated_tokens = output[0,inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # print the generated text
    print(generated_text)

    return generated_text

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

# Create occ_results directory if it doesn't exist
os.makedirs("occ_results", exist_ok=True)

# SYNTHETIC DATASET
# Load the JSON data
json_path = "../synthetic_metadata.json"
with open(json_path, 'r') as f:
    json_data = json.load(f)

# Iterate over each entry in the JSON
for entry in json_data:
    num = -1
    temperature = 1.0
    image_path = entry['image_file']
    print(image_path)
    occluded_image_path = "../generate_synthetic/occluded_dataset/" + image_path

    question = f"Your task is to count objects in the image. Assume the pattern of {entry['dot_shape']}s continues behind the black box. First, state what the pattern is, then give your final count."
    print(question)
    answer = get_answer(occluded_image_path, question, temperature=temperature)
    num = extract_num(answer)

    entry['answer'] = answer
    entry['results'] = num

# Save the updated JSON data back to the file
output_json_path = "occ_results/molmo_syn_results.json"
with open(output_json_path, 'w') as f:
    json.dump(json_data, f, indent=4)

print(f"Updated JSON file with Molmo results saved to {output_json_path}.")


# REAL DATASET
# Load the JSON data
json_path = "../real_metadata.json"
with open(json_path, 'r') as f:
    json_data = json.load(f)

# Iterate over each entry in the JSON
for entry in json_data:
    num = -1
    temperature = 1.0
    image_path = entry['image_file']
    print(image_path)
    occluded_image_path = "../real_dataset/" + image_path

    question = f"Your task is to count objects in the image. Assume the pattern of {entry['object']} continues behind the black box. First, state what the pattern is, then give your final count."
    print(question)
    answer = get_answer(occluded_image_path, question, temperature=temperature)
    num = extract_num(answer)

    entry['answer'] = answer
    entry['results'] = num


# Save the updated JSON data back to the file
output_json_path = "occ_results/molmo_real_results.json"
with open(output_json_path, 'w') as f:
    json.dump(json_data, f, indent=4)

print(f"Updated JSON file with Molmo results saved to {output_json_path}.")
