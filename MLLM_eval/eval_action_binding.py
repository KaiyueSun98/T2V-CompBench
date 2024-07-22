import argparse
import torch
import csv
import json
import os
import requests
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
from torchvision.io import write_video
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def extract_json(string):
    # Find the start and end positions of the JSON part
    start = string.find('{')
    end = string.rfind('}') + 1

    # Extract the JSON part from the string
    json_part = string[start:end]

    # Load the JSON part as a dictionary
    try:
        json_data = json.loads(json_part)
    except json.JSONDecodeError:
        # Handle the case when the JSON part is not valid
        print("Invalid JSON part")
        return None

    return json_data










def eval_model(args):

    image_grid_path = args.image_grid_path
    
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    with open(args.read_prompt_file,'r') as json_data:
        prompts = json.load(json_data)
        
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    
    csv_path = os.path.join(output_path, f'{args.t2v_model}_action_binding_score.csv')
    with open(csv_path, 'w', newline='') as csvfile: 
        # Create a CSV writer
        csv_writer = csv.writer(csvfile)
        # Write the header row
        csv_writer.writerow(["name","prompt","outputs_1","outputs_2","Score"])
        
        grid_images = [f for f in os.listdir(image_grid_path) if f[0].isdigit()]
        grid_images.sort(key=lambda x: int(x.split('.')[0]))#sort
        print(len(grid_images))
        for i in range(len(grid_images)):
            grid_image_name = grid_images[i]
            this_prompt = prompts[i]["prompt"]
            phrase_0 = prompts[i]["phrase_0"] # get first obj and action in a list
            phrase_1 = prompts[i]["phrase_1"] # get second obj and action in a list

            obj1 = phrase_0[0].split("?")[0]
            obj1_action = phrase_0[1].split("?")[0]
            obj2 = phrase_1[0].split("?")[0]
            obj2_action = phrase_1[1].split("?")[0]
        
            image_files = [os.path.join(image_grid_path, grid_images[i])]
            images = load_images(image_files)
            image_sizes = [x.size for x in images]
            images_tensor = process_images(  
                images,
                image_processor,
                model.config
            ).to(model.device, dtype=torch.float16)
            
            Q1 = "The provided image arranges key frames from a video in a grid view. Describe the video within 20 words, highlight all the characters or objects that appear throughout the frames and indicate how they act."
            qs = Q1
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
            
            conv_mode = "chatml_direct"

            args.conv_mode = conv_mode
            
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature, #0.2
                    top_p=args.top_p,
                    num_beams=args.num_beams, #1
                    max_new_tokens=args.max_new_tokens, #512
                    use_cache=True,
                )
                
            outputs_1 = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            conv.messages[-1][-1] = outputs_1

            Q2 = f"According to the video and your previous answer, evaluate if the text \'{this_prompt}\' is correctly portrayed in the video. \
Assign a score from 0 to 5 according the criteria: \
5: Both {obj1} and {obj2} are present, and {obj1_action}, {obj2_action}. \
4: Both {obj1} and {obj2} are present, but only one of the actions (either {obj1_action} or {obj2_action}) is depicted. \
3: Both {obj1} and {obj2} are present, neither of the actions are depicted. \
2: Only one of {obj1} or {obj2} is present, and its action matches the text. \
1: Only one of {obj1} or {obj2} is present, but its action does not match the text. \
0: Neither {obj1} nor {obj2} appears in the video. \
Provide your analysis and explanation in JSON format with the following keys: score (e.g., 2), explanation (within 20 words)."
            
            qs = Q2
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature, #0.2
                    top_p=args.top_p,
                    num_beams=args.num_beams, #1
                    max_new_tokens=args.max_new_tokens, #512
                    use_cache=True,
                )
                
            outputs_2 = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            json_obj = extract_json(outputs_2)
            try:
                score_tmp = json_obj["score"]
            except:
                score_tmp = "bad reply"
                
            print("score for",grid_images[i] , score_tmp)
     
            csv_writer.writerow([grid_image_name,this_prompt,outputs_1,outputs_2,score_tmp])
            csvfile.flush()
        return csv_path

def model_score(csv_path):
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        lines = list(reader)
        score = 0
        cnt = 0
        for line in lines[1:]:
            try:
                score_tmp = float(line[-1])/5  # normalize
                score+=score_tmp
                cnt+=1
            except:
                continue
        
        score = score/cnt
        print("number of images evaluated: ", cnt," action binding model score: ",score)
        
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["score: ",score]) 
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="LLaVA/llava-v1.6-34b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output-path", type=str, required=True, help="path to store the video scores")
    parser.add_argument("--read-prompt-file", type=str, default="meta_data/action_binding.json", help="path of txt file with input prompts and meta data")
    parser.add_argument(
        "--image-grid-path",
        type=str,
        required=True,
        help="path to folder of image grids extracted from videos",
    )
    parser.add_argument(
        "--t2v-model",
        type=str,
        required=True,
        help="model name",
    ) 
    args = parser.parse_args()

    csv_path = eval_model(args)
    model_score(csv_path)
    
    
    