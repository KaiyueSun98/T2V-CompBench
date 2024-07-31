import argparse
import torch
import csv
import json
import os
from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re

import cv2
import moviepy.editor as mp
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

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
import numpy as np


class video_preprocess():
    def __init__(self):
        pass
    
    def extract_frames(self, video_path, num_frames=16):
        frames = []
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("total frames", total_frames)
        if total_frames <= num_frames:
            frame_indices = np.arange(total_frames)
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()    
        return frames

    def rgb_to_yuv(self, frame):
        yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return yuv_frame

    def frames_to_video(self, frames, output_path, fps=8):
        yuv_frames = [self.rgb_to_yuv(frame) for frame in frames]
        video_tensor = torch.from_numpy(np.array(yuv_frames)).to(torch.uint8)
        write_video(output_path, video_tensor, fps, video_codec='h264', options={'crf': '18'})

    def convert_video(self, input_path, output_path, num_frames):
        frames = self.extract_frames(input_path,num_frames=num_frames)
        self.frames_to_video(frames, output_path)

    def merge_grid(self, image_list):       
        # Concatenate the images horizontally
        row1 = np.concatenate((image_list[0], image_list[1]), axis=1)
        row2 = np.concatenate((image_list[2], image_list[3]), axis=1)
        row3 = np.concatenate((image_list[4], image_list[5]), axis=1)
        # Concatenate the rows vertically
        grid = np.concatenate((row1, row2, row3), axis=0)
        return grid
    

    def read_video_path(self, video_path):
        if os.path.isdir(video_path):  # if video_path is a list of videos
            video = os.listdir(video_path)
        elif os.path.isfile(video_path):  # else if video_path is a single video
            video = [os.path.basename(video_path)]
            video_path = os.path.dirname(video_path)
        video.sort()
        return video, video_path
    
    def convert_video_to_frames(self, video_path, num_frames=16):
        video, video_path = self.read_video_path(video_path)
        print(f"start converting video to {num_frames} frames from path:", video_path)
    
        output_path = os.path.join(os.path.dirname(video_path), "frames", os.path.basename(video_path))
        os.makedirs(output_path, exist_ok=True)
    
        for v in video:
            vid_id = v.split(".")[0]
            frames_dir = os.path.join(output_path, vid_id)
            os.makedirs(frames_dir, exist_ok=True)
            vid_path = os.path.join(video_path,v)
            frames = self.extract_frames(vid_path,num_frames=num_frames)
            for frame_count,frame in enumerate(frames):
                frame_filename = os.path.join(frames_dir, f'{vid_id}_{frame_count:06d}.png')
                cv2.imwrite(frame_filename, frame)
        print("finish converting from path: ", video_path)
        print("video frames stored in: ", output_path)
        return output_path
        
    def convert_video_to_standard_video(self, video_path,num_frames):
        video, video_path = self.read_video_path(video_path)
        print("start converting video to video with 16 frames from path:", video_path)
        
        output_path = os.path.join(os.path.dirname(video_path), "video_standard", os.path.basename(video_path))
        os.makedirs(output_path, exist_ok=True)
        
        for v in video:
            v_mp4 = v.split(".")[0] + ".mp4"
            self.convert_video(os.path.join(video_path, f"{v}"), os.path.join(output_path, f"{v_mp4}"),num_frames)
        print("finish converting from path: ", video_path)
        print("standard video stored in: ", output_path)
        return output_path

    def convert_video_to_grid(self, video_path,num_image=6):
        video, video_path = self.read_video_path(video_path)
        print("start converting video to image grid with 6 frames from path:", video_path)
    
        output_path = os.path.join(os.path.dirname(video_path), "image_grid", os.path.basename(video_path))
        os.makedirs(output_path, exist_ok=True)
    
        for v in video:
            vid_id = v.split(".")[0]
            vid_path = os.path.join(video_path,v)
            frames = self.extract_frames(vid_path)
            frame_indices = np.linspace(0, len(frames) - 1, num_image, dtype=int) #take 6 from 16 evenly, 1st & last included
            grid = [frames[i] for i in frame_indices]
            grid_image = self.merge_grid(grid)
            grid_filename = os.path.join(output_path, f'{vid_id}.png')
            cv2.imwrite(grid_filename, grid_image)
        print("finish converting from path: ", video_path)
        print("image grid stored in: ", output_path)
        return output_path
           

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


def eval_model(args):
    
    video_path = args.video_path
    video_preprocess = video_preprocess()
    frame_folder = video_preprocess.convert_video_to_frames(video_path)
    
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
    
    csv_path = os.path.join(output_path, f'{args.t2v_model}_dynamic_attr_score.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the header row
        csv_writer.writerow(["name","prompt", "1_answer1", "1_answer2", "_answer3","2_answer1", "2_answer2", "2_answer3", "inter_answers", "Score"])

        initial = "Describe the provided image within 20 words, highlight all the objects' attributes that appear in the image."
        
        frame_images = [f for f in os.listdir(frame_folder) if f.isdigit() ]
        frame_images.sort(key=lambda x: int(x.split('.')[0]))#sort
        
        for i in range(len(frame_images)):
            
            out = []
            inter_answers = []
            question = []
            frame_image_name = frame_images[i]  
            score = []
            score_total = 0
            
            this_prompt = prompts[i]["prompt"]
            
            phrase_0 = prompts[i]["state 0"] # get initial state
            phrase_1 = prompts[i]["state 1"] # get end state
            image_files = os.path.join(frame_folder,frame_images[i])
            image_files = os.listdir(image_files)     
            
            if len(image_files) == 0:
                print(f"no image found for {frame_images[i]}")
                continue
            
            image_files.sort()
            phrases = [phrase_0,phrase_1]
            

                            
            for j, question_group in enumerate(phrases):
                if j == 0:
                    state_num = 0 # get initial image
                else:
                    state_num = -1 # get end image
                image_file = [os.path.join(frame_folder,frame_images[i],image_files[state_num])]
                images = load_images(image_file)
                image_sizes = [x.size for x in images] 
                images_tensor = process_images( 
                    images,
                    image_processor,
                    model.config
                ).to(model.device, dtype=torch.float16)
            
                image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                qs = initial
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
                
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                out.append(outputs) #out[0]: 1_answer1. out[3]: 2_answer1
                conv.messages[-1][-1] = outputs
                
                
                for k in range(2):
                    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                    question_group_tmp = phrases[(k)%2]
                    qs = f"According to the image and your previous answer, evaluate if the text \'{question_group_tmp}\' is correctly described in the image. \
Give a score from 1 to 5, according the criteria: \
5: the image accurately describe the text. \
4: the image roughly describe the text, but the attribute is a little different. \
3: the image roughly describe the text, but the attribute is totally different. \
2: the image do not describe the text. \
1: the image did not depict any elements that match the text. \
Provide your analysis and explanation in JSON format with the following keys: score \
(e.g., 2), explanation (within 20 words)."
                    question.append(qs)

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
                            temperature=args.temperature,
                            top_p=args.top_p,
                            num_beams=args.num_beams,
                            max_new_tokens=args.max_new_tokens, 
                            use_cache=True,
                        )
                        
                    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                    out.append(outputs) #out[1]: 1_answer2, out[2]: 1_answer3, out[4]: 2_answer2, out[5]: 2_answer3, 
                    conv.messages[-1][-1] = outputs
                    
                    # get score from outputs
                    pattern = r'"score":\s*(\d+),'
                    match = re.search(pattern, outputs)

                    if match:
                        score_tmp = int(match.group(1))
                    else:
                        print('No score found')
                    score.append(score_tmp)
            
            # check the intermediate states
            flag = 0
            flag_cnt = 0 # for intermediate state
            intermediate_frames = 16 - 2
            frame_array = np.range(1,15)
            for j, inter_state in enumerate(frame_array):
                image_file = [os.path.join(frame_folder,frame_images[i],image_files[inter_state])]
                images = load_images(image_file)
                image_sizes = [x.size for x in images] 
                images_tensor = process_images(
                    images,
                    image_processor,
                    model.config
                ).to(model.device, dtype=torch.float16)

                qs = initial
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
                
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                out.append(outputs) #out[6] out[8]
                conv.messages[-1][-1] = outputs
                      
                image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                qs = f"According to the image, evaluate if the image is aligned with the text \'{phrase_0}\' or \'{phrase_1}\'. \
Give a score from 0 to 1, according the criteria: \
2: the image matches with the text {phrase_0}. \
1: the image matches with the text {phrase_1}. \
0: the image is not aligned with the two texts totally. \
Provide your analysis and explanation in JSON format with the following keys: score \
(e.g., 1), explanation (within 20 words)."
                question.append(qs)

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
                        temperature=args.temperature, 
                        top_p=args.top_p,
                        num_beams=args.num_beams, 
                        max_new_tokens=args.max_new_tokens, 
                        use_cache=True,
                    )
                    
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                out.append(outputs) #out[7] out[9] ...
                inter_answers.append(outputs)
                conv.messages[-1][-1] = outputs
                
                # get score from outputs
                pattern = r'"score":\s*(\d+(\.\d+)?),'
                match = re.search(pattern, outputs)
                if match:
                    score_tmp = float(match.group(1))
                else:
                    print('No score found')
                if score_tmp>0:
                    flag_cnt += 1
            if flag_cnt >intermediate_frames*0.8: # threshold for intermediate frames
                flag = 1
                            
            # calculate score_total
            score_1 = score[0]
            score_1_1 = score[1]
            score_2 = score[2]
            score_2_1 = score[3]
            
            score_total = ((score_1/5 * (1-score_1_1/5))*0.5 + ((1-score_2/5) * (score_2_1/5))*0.5)*flag
            print("score total for",frame_images[i] , score_total)
            
            csv_writer.writerow([frame_image_name, this_prompt, out[0], out[1], out[2], out[3], out[4], out[5], inter_answers, score_total])
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
                score_tmp = float(line[-1]) 
                score+=score_tmp
                cnt+=1
            except:
                continue
        
        score = score/cnt
        print("number of images evaluated: ", cnt," dynamic attribute binding model score: ",score)
        
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["score: ",score]) 
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="LLaVA/llava-v1.6-34b", help="path to llava model")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output-path", type=str, default="../csv_output_dynamic_attr", help="path to store the video scores")
    parser.add_argument("--read-prompt-file", type=str, default="../meta_data/dynamic_attribute_binding.json", help="path of json file with meta data")
    parser.add_argument(
        "--video-path",
        type=str,
        required=True,
        help="path to videos",
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
