from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import transformers
import torch
import os

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

SKIPPED = 0

def get_answer(image_path, question, temperature=1.0):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": question},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=8192)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text[0])
    return output_text[0]

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

# Create unocc_results directory if it doesn't exist
os.makedirs("unocc_results", exist_ok=True)

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
    occluded_image_path = "../generate_synthetic/unoccluded_dataset/" + image_path

    question = f"Count the exact number of {entry['dot_shape']}s in the image. Assume the pattern of {entry['dot_shape']}s continues behind any black box. Provide the total number of {entry['dot_shape']}s as if the black box were not there. Only count {entry['dot_shape']}s that are visible within the frame (or would be visible without the occluding box). If {entry['dot_shape']}s are partially in the frame (i.e. if any part of {entry['dot_shape']}s are visible), count it. If the {entry['dot_shape']}s would be partially in the frame without the occluding box, count it."

    answer = get_answer(occluded_image_path, question, temperature=temperature)
    num = extract_num(answer)

    entry['answer'] = answer
    entry['results'] = num

# Save the updated JSON data back to the file
output_json_path = "unocc_results/qwen2_syn_results.json"
with open(output_json_path, 'w') as f:
    json.dump(json_data, f, indent=4)

print(f"Updated JSON file with Qwen results saved to {output_json_path}.")


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
    occluded_image_path = "../unocc_real_dataset/" + image_path

    
    question = f"Count the exact number of {entry['object']} in the image. Assume the pattern of {entry['object']} continues behind any black box. Provide the total number of {entry['object']} as if the black box were not there. Only count {entry['object']} that are visible within the frame (or would be visible without the occluding box). If {entry['object']} are partially in the frame (i.e. if any part of {entry['object']} are visible), count it. If the {entry['object']} would be partially in the frame without the occluding box, count it."
    answer = get_answer(occluded_image_path, question, temperature=temperature)
    num = extract_num(answer)

    entry['answer'] = answer
    entry['results'] = num


# Save the updated JSON data back to the file
output_json_path = "unocc_results/qwen2_real_results.json"
with open(output_json_path, 'w') as f:
    json.dump(json_data, f, indent=4)

print(f"Updated JSON file with Qwen results saved to {output_json_path}.")