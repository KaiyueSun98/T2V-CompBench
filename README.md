# T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generation

<a href='https://t2v-compbench.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://arxiv.org/abs/2407.14505'><img src='https://img.shields.io/badge/T2V--CompBench-Arxiv-red'></a> 

This repository is the official implementation of the following paper:
> **T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generation**<br>
> [Kaiyue Sun](https://scholar.google.com/citations?user=mieuBzUAAAAJ&hl=en)<sup>1</sup>, [Kaiyi Huang](https://github.com/Karine-Huang)<sup>1</sup>, [Xian Liu](https://alvinliu0.github.io/)<sup>2</sup>, [Yue Wu](https://yuewuhkust.github.io/)<sup>3</sup>, Zihan Xu<sup>1</sup>, [Zhenguo Li](https://scholar.google.com/citations?hl=en&user=XboZC1AAAAAJ&view_op=list_works&sortby=pubdate)<sup>3</sup>, [Xihui Liu](https://xh-liu.github.io/)<sup>1</sup><br>
> ***<sup>1</sup>The University of Hong Kong, <sup>2</sup>The Chinese University of Hong Kong, <sup>3</sup>Huawei Noah’s Ark Lab***

### Table of Contents
- [Updates](#updates)
- [Overview](#overview)
- [Evaluation Results](#evaluation_results)
- [Prompt Suite](#prompt_suite)
- [MLLM-based Evaluation](#mllm_eval)
  - [Consistent Attribute Binding](#consistent_attribute_binding)
  - [Dynamic Attribute Binding](#dynamic_attribute_binding)
  - [Action Binding](#action_binding)
  - [Object Interactions](#object_interactions)
- [Detection-based Evaluation](#detection_eval)
  - [Spatial Relationships](#spatial_relationships)
  - [Generative Numeracy](#generative_numeracy)
- [Tracking-based Evaluation](#tracking_eval)
  - [Motion Binding](#motion_binding)
- [Sampled Videos](#sampled_videos)
- [Evaluate Your Own Videos](#eval_your_own)
- [Citation](#citation)

<a name="updates"></a>
## 🚩 Updates
- :black_square_button: [TODO] T2V-CompBench Leaderboard
- ✅ [08/2024] Release the generated videos for T2V-CompBench evaluation.
- ✅ [07/2024] Release the evaluation code for 7 categories in compositional Text-to-Video (T2V) generation.
- ✅ [07/2024] Release the prompt dataset and metadata.
  
<a name="overview"></a>
## :mega: Overview
![teaser](./asset/teaser.png)
We propose **T2V-CompBench**, the first benchmark tailored for **compositional text-to-video generation**. T2V-CompBench encompasses diverse aspects of compositionality, including **consistent attribute binding**, **dynamic attribute binding**, **spatial relationships**, **motion binding**, **action binding**, **object interactions**, and **generative numeracy**. We further carefully design evaluation metrics of **MLLM-based metrics**, **detection-based metrics**, and **tracking-based metrics**, which can better reflect the compositional text-to-video generation quality of seven proposed categories with 700 text prompts. The effectiveness of the proposed metrics is verified by correlation with human evaluations. We also **benchmark various text-to-video generative models** and conduct in-depth analysis across different models and different compositional categories. We find that compositional text-to-video generation is highly challenging for current models, and we hope that our attempt will shed light on future research in this direction.

<a name="evaluation_results"></a>
## :mortar_board: Evaluation Results
![ranking](./asset/ranking.png)

We benchmark 13 publicly available text-to-video generation models and 7 commercial models including Kling, Gen-3, Gen-2, Pika, Luma Dream Machine, Dreamina and PixVerse. We normalize the results per categories for clearer comparisons. 

<a name="prompt_suite"></a>
## :blue_book: T2V-CompBench Prompt Suite
The T2V-CompBench prompt suite includes 700 prompts covering 7 categories, each with 100 prompts. 

Text prompts of each category are saved in a text file in the ```prompts/``` directory.

<a name="mllm_eval"></a>
## :speech_balloon: MLLM-based Evaluation
We use **LLaVA** as the MLLM model to evaluate the four categories: consistent attribute binding, dynamic attribute binding, action binding and object interactions.
### :hammer: 1. Install Requirements

MLLM-based evaluation metrics are based on the official repository of LLaVA. You can refer to [LLaVA's GitHub repository](https://github.com/haotian-liu/LLaVA) for specific environment dependencies and weights.

### :clapper: 2. Prepare Evaluation Videos

Generate videos of your model using the T2V-CompBench prompts provided in the `prompts` directory. Organize them in the following structure (using the *consistent attribute binding* category as an example):

```
../video/consistent_attr
├── 0001.mp4
├── 0002.mp4
├── 0003.mp4
├── 0004.mp4
...
└── 0100.mp4
```

Note: The numerical names of the video files are just to indicate the reading order that matches the order of prompts. You can use other naming conventions that maintain the order (*e.g.* "0.mp4", "1.mpg", *etc.*)

### :running: 3. Run the Evaluation Codes

After obtaining the official LLaVA code, place the following evaluation scripts in the `LLaVA/llava/eval` directory:

- `eval_consistent_attr.py`
- `eval_dynamic_attr.py`
- `eval_action_binding.py`
- `eval_interaction.py`

Prepare the video repository path (*e.g.*, "../video/consistent_attr") or a specific video path (*e.g.*, "../video/consistent_attr/0001.mp4") in the argument `--video-path`. Configure the folder to store the csv files with the `--output-path` argument, configure the json file containing prompts and meta information with the `--read-prompt-file` argument. The evaluation codes will automatically convert the videos into the required formats (image grid or 16 frames) and then calculate the score.

<a name="consistent_attribute_binding"></a>
#### :tangerine: Consistent Attribute Binding

Input the video path and run the command:

```
python llava/eval/eval_consistent_attr.py \
  --video-path ../video/consistent_attr \
  --output-path ../csv_output_consistent_attr \
  --read-prompt-file ../meta_data/consistent_attribute_binding.json \
  --t2v-model mymodel
```

The conversations with the MLLM will be saved in a CSV file: `../csv_output_consistent_attr/mymodel_consistent_attr_score.csv`. The video name, prompt, and score for each text-video pair will be recorded in the columns named of "name","prompt", "Score". 

The final score of the model in this category (consistent attribute binding) will be saved in the last line of this CSV file.

<a name="dynamic_attribute_binding"></a>
#### :lemon: Dynamic Attribute Binding

Input the video path and run the command:

```
python llava/eval/eval_dynamic_attr.py
  --video-path ../video/dynamic_attr \
  --output-path ../csv_output_dynamic_attr \
  --read-prompt-file ../meta_data/dynamic_attribute_binding.json \
  --t2v-model mymodel
```

The conversations with the MLLM will be saved in a CSV file: `../csv_output_dynamic_attr/mymodel_dynamic_attr_score.csv`. The video name, prompt, and score for each text-video pair will be recorded in the columns named of "name","prompt", "Score". 

The final score of the model in this category (dynamic attribute binding) will be saved in the last line of this CSV file.

<a name="action_binding"></a>
#### :whale: Action Binding

Input the video path and run the command:

```
python llava/eval/eval_action_binding.py
  --video-path ../video/action_binding \
  --output-path ../csv_output_action_binding \
  --read-prompt-file ../meta_data/action_binding.json \
  --t2v-model mymodel
```

The conversations with the MLLM will be saved in a CSV file: `../csv_output_action_binding/mymodel_action_binding_score.csv`. The video name, prompt, and score for each text-video pair will be recorded in the columns named of "name","prompt", "Score". 

The final score of the model in this category (action binding) will be saved in the last line of this CSV file.

<a name="object_interactions"></a>
#### :crystal_ball: Object Interactions

Input the video path and run the command:

```
python llava/eval/eval_interaction.py
  --video-path ../video/interaction \
  --output-path ../csv_output_object_interactions \
  --read-prompt-file ../meta_data/object_interactions.json \
  --t2v-model mymodel
```

The conversations with the MLLM will be saved in a CSV file: `../csv_output_object_interactions/mymodel_object_interactions_score.csv`. The video name, prompt, and score for each text-video pair will be recorded in the columns named of "name","prompt", "Score". 

The final score of the model in this category (object interactions) will be saved in the last line of this CSV file.

<a name="detection_eval"></a>
## :mag_right: Detection-based Evaluation
We use **GroundingDINO** as the detection tool to evaluate the two categories: 2D spatial relationships and generative numeracy.

We use **Depth Anything + GroundingSAM** to evaluate 3D spatial relationships ("in front of" & "behind").

### :hammer: 1. Install Requirements

Detection-based Evaluation metrics are based on the official repositories of Depth Anything and GroundingSAM. You can refer to [Depth Anything's GitHub repository](https://github.com/LiheYoung/Depth-Anything/tree/main) and [GroundingSAM's GitHub repository](https://github.com/IDEA-Research/GroundingDINO/tree/main) for specific environment dependencies and weights.

### :clapper: 2. Prepare Evaluation Videos

Generate videos of your model using the T2V-CompBench prompts provided in the `prompts` directory. Organize them in the following structure (using the *spatial relationships* category as an example):

```
../video/spatial_relationships
├── 0001.mp4
├── 0002.mp4
├── 0003.mp4
├── 0004.mp4
...
└── 0100.mp4
```

Note: Please put all the videos of spatial relationships (both 2D and 3D) together. The numerical names of the video files are just to indicate the reading order that matches the order of prompts. You can use other naming conventions that maintain the order (*e.g.* "0.mp4", "1.mpg", *etc.*)

### :running: 3. Run the Evaluation Codes

<a name="spatial_relationships"></a>
#### :cactus: Spatial Relationships

After obtaining the official Depth Anything code, place the following evaluation scripts in the `Depth-Anything/` directory:

- `run_depth.py`

After obtaining the official GroundingSAM code, place the following evaluation scripts in the `Grounded-Segment-Anything/` directory:

- `eval_spatial_relationships.py`

Compute the evaluation metric:

##### step 1: prepare the input images

```
python Depth-Anything/run_depth.py
  --video-path ../video/spatial_relationships \
  --output_dir ../output_spatial_depth \
  --read-prompt-file ../meta_data/spatial_relationships.json \
  --t2v-model mymodel
```
This script will convert the videos into the required formats.

The depth images will be stored in the `../output_spatial_depth/mymodel` directory.

The frame images will be stored in the default directory: `../video/frames/spatial_relationships/`

##### step 2: evaluate spatial relationships

```
python Grounded-Segment-Anything/eval_spatial_relationships.py
  --frame_folder ../video/frames/spatial_relationships/ \
  --depth_folder ../output_spatial_depth \
  --output-path ../csv_spatial \
  --read-prompt-file ../meta_data/spatial_relationships.json \
  --t2v-model mymodel \
  --output_dir_2d ../output_2D_spatial/ \
  --output_dir_3d ../output_3D_spatial/
```

The output frame images showing the object bounding boxes with 2D spatial relationships will be stored in the `../output_2D_spatial/mymodel` directory.

The output frame images showing the object bounding boxes and segmentations with 3D spatial relationship will be stored in the `../output_3D_spatial/mymodel` directory.

The frame scores will be saved in `../csv_spatial/mymodel_2dframe.csv` and `../csv_spatial/mymodel_3dframe.csv`.

Frame scores will be combined to calculate the video scores, which will be saved in `../csv_spatial/mymodel_2dvideo.csv` and `../csv_spatial/mymodel_3dvideo.csv`.

The final score of the model in this category (spatial relationships) will be saved in the last line of `../csv_spatial/mymodel_3dvideo.csv`.

<a name="generative_numeracy"></a>
#### :apple: Generative Numeracy

You can reuse the official implementation of GroundingSAM and its environment by placing the following evaluation script in the `Grounded-Segment-Anything/GroundingDINO/demo` directory:

- `eval_numeracy.py`

Or you can refer to [GroundingDINO's GitHub repository](https://github.com/IDEA-Research/GroundingDINO/tree/main) to install the required environment dependencies and download the weights. Then place the the same evaluation script in the `GroundingDINO/demo` directory

Compute the evaluation metric:

```
python eval_numeracy.py
  --video-path ../video/generative_numeracy \
  --output-path ../csv_numeracy \
  --read-prompt-file ../meta_data/generative_numeracy.json \
  --t2v-model mymodel \
  --output_dir ../output_numeracy/ \

```
The output frame images showing the object bounding boxes will be stored in the `../output_numeracy/mymodel` directory.

The frame scores will be saved in `../csv_numeracy/mymodel_numeracy_frame.csv`.

They will be combined to calculate the video scores, which will be saved in `../csv_numeracy/mymodel_numeracy_video.csv` and `../csv_spatial/mymodel_3dvideo.csv`.

The final score of the model in this category (generative numeracy) will be saved in the last line of `../csv_numeracy/mymodel_numeracy_video.csv`.

<a name="tracking_eval"></a>
## :tractor: Tracking-based Evaluation
We use **GroundingSAM + DOT** to evaluate motion binding.

### :hammer: 1. Install Requirements

Tracking-based Evaluation metric is based on the official repositories of GroundingSAM and Dense Optical Tracking. You can refer to [GroundingSAM's GitHub repository](https://github.com/IDEA-Research/GroundingDINO/tree/main) and [Dense Optical Tracking's GitHub repository](https://github.com/16lemoing/dot?tab=readme-ov-file) for specific environment dependencies and weights.

### :clapper: 2. Prepare Evaluation Videos

Generate videos of your model using the T2V-CompBench prompts provided in the `prompts` directory. Organize them in the following structure:

```
../video/motion_binding
├── 0001.mp4
├── 0002.mp4
├── 0003.mp4
├── 0004.mp4
...
└── 0100.mp4
```

Note: The numerical names of the video files are just to indicate the reading order that matches the order of prompts. You can use other naming conventions that maintain the order (*e.g.* "0.mp4", "1.mpg", *etc.*)

### :running: 3. Run the Evaluation Codes

<a name="motion_binding"></a>
#### :white_circle: Motion Binding

After obtaining the official GroundingSAM code, place the following script in the `Grounded-Segment-Anything/` directory:

- `motion_binding_seg.py`

After obtaining the official DOT code, place the following evaluation scripts in the `dot/` directory:

- `eavl_motion_binding_foreground.py`
- `eval_motion_binding_background.py`
- `motion_binding_score_cal.py`

Then, replace the original `dot/dot/utils/options/demo_options.py` by 

- `demo_options.py`

Compute the evaluation metric:

##### step 1: prepare the input images

Configure the video frame number with the `--total_frame` argument, the video fps (frames per second) with the `--fps` argument. The script will  convert the videos into the required formats.

```
python Grounded-Segment-Anything/motion_binding_seg.py
  --video_folder ../video/motion_binding \
  --read-prompt-file ../meta_data/motion_binding.json \
  --t2v-model mymodel \
  --total_frame 16 \
  --fps 8 \
  --output_dir ../output_motion_binding_seg
```

The downsampled video with fps≈8 will be stored in the default directory: `../video/video_standard/motion_binding/`

The background and forground segmentations of the 1st frame of the videos will be stored in the `output_motion_binding_seg/mymodel` directory.

##### step 2: Track the foregroud points

```
python dot/eavl_motion_binding_foreground.py
  --video_folder ../video/video_standard/motion_binding \
  --mask_folder ../output_motion_binding_seg \
  --read-prompt-file ../meta_data/motion_binding.json \
  --t2v_model mymodel \
  --output_path ../csv_motion_binding \
  --output_dir ../vid_output_motion_binding
```

##### step 3: Track the background points

```
python dot/eavl_motion_binding_background.py
  --video_folder ../video/video_standard/motion_binding \
  --mask_folder ../output_motion_binding_seg \
  --read-prompt-file ../meta_data/motion_binding.json \
  --t2v_model mymodel \
  --output_path ../csv_motion_binding \
  --output_dir ../vid_output_motion_binding \
```

The output videos showing the foreground and background point tracking will be stored in the `../vid_output_motion_binding/mymodel` directory.

The change in centre of foreground points will be saved in `../csv_motion_binding/mymodel_foreground.csv`.

The change in centre of background points will be saved in `../csv_motion_binding/mymodel_background.csv`.

They will be combined to calculate the absolute displacement of the forefround object(s).

##### step 4: Calculate the score

```
python dot/motion_binding_score_cal.py --t2v-model mymodel --output_path ../csv_motion_binding
```
The absolute displacement of the forefround object(s) in each video will be saved in `../csv_motion_binding/mymodel_back_fore.csv`

The score for each video will be saved in `../csv_motion_binding/mymodel_score.csv`

The final score of the model in this category (motion) will be saved in the last line of `../csv_motion_binding/mymodel_score.csv`.

<a name="sampled_videos"></a>
## :film_strip: Sampled Videos
To facilitate future research and ensure complete transparency, we release all the videos we sampled and used for the T2V-CompBench evaluation.
You can download them on [OneDrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/kaiyue_connect_hku_hk/EjXmmz0XXQBFu_EMmBM2WCkBI9iuATOib-dC3GLBUmCuIw?e=a6lFCC).


<a name="eval_your_own"></a>
## :surfer: Evaluate Your Own Videos

To evaluate your own videos, prepare the evaluation videos and prompt or metadata files similar to the provided examples. Follow the same steps to run the evaluation codes.

<a name="citation"></a>
## :black_nib: Citation
If you find T2V-CompBench useful for your research, please cite our paper. :)
```
@article{sun2024t2v,
  title={T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generation},
  author={Sun, Kaiyue and Huang, Kaiyi and Liu, Xian and Wu, Yue and Xu, Zihan and Li, Zhenguo and Liu, Xihui},
  journal={arXiv preprint arXiv:2407.14505},
  year={2024}
}
```
