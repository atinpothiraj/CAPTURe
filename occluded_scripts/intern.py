import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import transformers
import json

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

path = 'OpenGVLab/InternVL2-8B'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

def get_answer(image_path, question, temperature=1.0):
    # set the max number of tiles in `max_num`
    pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=4000, do_sample=False)

    # single-image single-round conversation (单图单轮对话)
    prompt = '\n<image>\n'
    question += prompt
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    print(f'User: {question}\nAssistant: {response}')
    return response

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
    json_data = json.load(f)

# Iterate over each entry in the JSON
for entry in json_data:
    num = -1
    temperature = 1.0
    image_path = entry['image_file']
    print(image_path)
    occluded_image_path = "../generate_synthetic/occluded_dataset/" + image_path

    question = f"Count the exact number of {entry['dot_shape']}s in the image. Assume the pattern of {entry['dot_shape']}s continues behind any black box. Provide the total number of {entry['dot_shape']}s as if the black box were not there. Only count {entry['dot_shape']}s that are visible within the frame (or would be visible without the occluding box). If {entry['dot_shape']}s are partially in the frame (i.e. if any part of {entry['dot_shape']}s are visible), count it. If the {entry['dot_shape']}s would be partially in the frame without the occluding box, count it."
    print(question)
    answer = get_answer(occluded_image_path, question, temperature=temperature)
    num = extract_num(answer)

    entry['answer'] = answer
    entry['results'] = num

# Save the updated JSON data back to the file
output_json_path = "occ_results/intern_syn_results.json"
with open(output_json_path, 'w') as f:
    json.dump(json_data, f, indent=4)

print(f"Updated JSON file with Intern results saved to {output_json_path}.")


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
    question = f"Count the exact number of {entry['object']} in the image. Assume the pattern of {entry['object']} continues behind any black box. Provide the total number of {entry['object']} as if the black box were not there. Only count {entry['object']} that are visible within the frame (or would be visible without the occluding box). If {entry['object']} are partially in the frame (i.e. if any part of {entry['object']} are visible), count it. If the {entry['object']} would be partially in the frame without the occluding box, count it."
    print(question)
    answer = get_answer(occluded_image_path, question, temperature=temperature)
    num = extract_num(answer)

    entry['answer'] = answer
    entry['results'] = num


# Save the updated JSON data back to the file
output_json_path = "occ_results/intern_real_results.json"
with open(output_json_path, 'w') as f:
    json.dump(json_data, f, indent=4)

print(f"Updated JSON file with Intern results saved to {output_json_path}.")
