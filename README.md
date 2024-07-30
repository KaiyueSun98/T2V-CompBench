# T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generation
Kaiyue Sun<sup>1</sup>, Kaiyi Huang<sup>1</sup>, Xian Liu<sup>2</sup>, Yue Wu<sup>3</sup>, Zihan Xu<sup>1</sup>, Zhenguo Li<sup>3</sup>, and Xihui Liu<sup>1</sup>.

**<sup>1</sup>The University of Hong Kong, <sup>2</sup>The Chinese University of Hong Kong, <sup>3</sup>Huawei Noah’s Ark Lab**

<a href='https://t2v-compbench.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://arxiv.org/pdf/2407.14505'><img src='https://img.shields.io/badge/T2V--CompBench-Arxiv-red'></a> 


## T2V-CompBench dataset
The T2V-CompBench dataset includes 700 prompts covering 7 categories, each with 100 prompts. 

Text prompts of each category are saved in a text file in the ```prompts/``` directory.


## MLLM-based Evaluation
We use LLaVA as the MLLM model to evaluate the four categories: consistent attribute binding, dynamic attribute binding, action binding and object interactions.
#### 1: Install Requirements

MLLM-based evaluation metrics are based on the official repository of LLaVA. You can refer to [LLaVA's GitHub repository](https://github.com/haotian-liu/LLaVA) for specific environment dependencies and weights.

#### 2: Prepare Evaluation Videos

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

#### 3: Run the Evaluation Codes

After obtaining the official LLaVA code, place the following evaluation scripts in the `LLaVA/llava/eval` directory:

- `eval_consistent_attr.py`
- `eval_action_binding.py`
- `eval_interaction.py`

Prepare the video repository path (*e.g.*, "../video/consistent_attr") or a specific video path (*e.g.*, "../video/consistent_attr/0001.mp4") in the argument `--video-path`. Configure the folder to store the csv files with the `--output-path` argument, the prompts and meta information with the `--read-prompt-file` argument. The evaluation codes will automatically convert the video into the required formats (image grid or 16 frames) and then calculate the score.

##### Consistent Attribute Binding

Prepare the video path for `--video-path` and run the following command:

```
python llava/eval/eval_consistent_attr.py --video-path ../video/consistent_attr --output-path ../csv_output_consistent_attr --read-prompt-file ../meta_data/consistent_attribute_binding.json --t2v-model mymodel
```

The output will be a CSV file named f"{mymodel}_consistent_attr_score.csv" in the `../csv_output_consistent_attr` directory. The video name, prompt, and score for each text-video pair will be recorded in the columns named of "name","prompt", "Score".

##### Action Binding

Input the video path and run:

```
python llava/eval/eval_action_binding.py --video-path ../video/action_binding --output-path ../csv_output_action_binding --read-prompt-file ../meta_data/action_binding.json --t2v-model mymodel
```

The output will be a CSV file named f"{mymodel}_action_binding_score.csv" in the `../csv_output_action_binding` directory. The video name, prompt, and score for each text-video pair will be recorded in the columns named of "name","prompt", "Score".

##### Object Interactions

Input the video path and run:

```
python llava/eval/eval_interaction.py --video-path ../video/interaction --output-path ../csv_output_object_interactions --read-prompt-file ../meta_data/object_interactions.json --t2v-model mymodel
```

The output will be a CSV file named f"{mymodel}_object_interactions_score.csv" in the `../csv_output_object_interactions` directory. The video name, prompt, and score for each text-video pair will be recorded in the columns named of "name","prompt", "Score".

## Detection-based Evaluation
We use GroundingDINO as the detection tool to evaluate the two categories: 2D spatial relationships and generative numeracy.

We use Depth Anything + GroundingSAM to evaluate 3D spatial relationships ("in front of" & "behind").

#### 1: Install Requirements

Detection-based Evaluation metrics are based on the official repositories of Depth Anything and GroundingSAM. You can refer to [Depth Anything's GitHub repository](https://github.com/LiheYoung/Depth-Anything/tree/main) and [GroundingSAM's GitHub repository](https://github.com/IDEA-Research/GroundingDINO/tree/main) for specific environment dependencies and weights.

#### 2: Prepare Evaluation Videos

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

#### 3: Run the Evaluation Codes

##### Spatial Relationships

After obtaining the official Depth Anything code, place the following evaluation scripts in the `Depth-Anything/` directory:

- `run_depth.py`

After obtaining the official GroundingSAM code, place the following evaluation scripts in the `Grounded-Segment-Anything/` directory:

- `eval_spatial_relationships.py`

Compute the evaluation metric:

step 1: prepare the input images

```
python Depth-Anything/run_depth.py --output_dir ../output_spatial_depth --read-prompt-file ../meta_data/spatial_relationships.json --video-path ../video/spatial_relationships --t2v-model mymodel
```
This script will convert the video into the required formats.

The depth images will be stored in the `../output_spatial_depth/mymodel` directory.

The frame images will be stored in the default path: `../video/frames/spatial_relationships/`

step 2: evaluation

```
python Grounded-Segment-Anything/eval_spatial_relationships.py --output-path ../csv_spatial --read-prompt-file ../meta_data/spatial_relationships.json --frame_folder ../video/frames/spatial_relationships/ --t2v-model mymodel --output_dir_3d ../output_3D_spatial --depth_folder ../output_spatial_depth --output_dir_2d ../output_2D_spatial/
```

The output frame images showing the object bounding boxes with 2d spatial relationships will be stored in the `../output_2D_spatial/mymodel` directory.

The output frame images showing the object bounding boxes and segmentations with 3d spatial relationship will be stored in the `../output_3D_spatial/mymodel` directory.

The frame scores will be saved in `../csv_spatial/mymodel_2dframe.csv` and `../csv_spatial/mymodel_3dframe.csv`.

They will be combined to calculate the video scores, which will be saved in `../csv_spatial/mymodel_2dvideo.csv` and `../csv_spatial/mymodel_3dvideo.csv`.

The final score of the model in this category (spatial relationships) will be saved in the last line of `../csv_spatial/mymodel_3dvideo.csv`.

## Tracking-based Evaluation
We use GroundingSAM + DOT to evaluate motion binding.

#### 1: Install Requirements

Tracking-based Evaluation metric is based on the official repositories of GroundingSAM and Dense Optical Tracking. You can refer to [GroundingSAM's GitHub repository](https://github.com/IDEA-Research/GroundingDINO/tree/main) and [Dense Optical Tracking's GitHub repository](https://github.com/16lemoing/dot?tab=readme-ov-file) for specific environment dependencies and weights.

#### 2: Prepare Evaluation Videos

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

#### 3: Run the Evaluation Codes

##### Motion Binding

After obtaining the official GroundingSAM code, place the following script in the `Grounded-Segment-Anything/` directory:

- `motion_binding_seg.py`

After obtaining the official DOT code, place the following evaluation scripts in the `dot/` directory:

- `eavl_motion_binding_foreground.py`
- `eval_motion_binding_background.py`
- `motion_binding_score_cal.py`

Then, replace the original `dot/dot/utils/options/demo_options.py` by 

- `demo_options.py`

Compute the evaluation metric:

step 1: prepare the input images

Configure the video frame number with the `--total_frame` argument, the video fps (frames per second) with the `--fps` argument. The script will  convert the video into the required formats.

```
python Grounded-Segment-Anything/motion_binding_seg.py --output_dir ../output_motion_binding_seg --read-prompt-file ../meta_data/motion_binding.json --video_folder ../video/motion_binding --t2v-model mymodel --total_frame 16 --fps 8
```

The downsampled video with fps≈8 will be stored in the default path: `../video/video_standard/motion_binding/`

The background and forground segmentations of the 1st frame of the videos will be stored in the `output_motion_binding_seg/mymodel` directory.

step 2: Track the foregroud points

```
python dot/eavl_motion_binding_foreground.py --read-prompt-file ../meta_data/motion_binding.json --video_folder ../video/video_standard/motion_binding --mask_folder ../output_motion_binding_seg --t2v_model mymodel --output_path ../csv_motion_binding --output_dir ../vid_output_motion_binding
```

step 3: Track the background points

```
python dot/eavl_motion_binding_background.py --read-prompt-file ../meta_data/motion_binding.json --video_folder ../video/video_standard/motion_binding --mask_folder ../output_motion_binding_seg --t2v_model mymodel --output_path ../csv_motion_binding --output_dir ../vid_output_motion_binding
```

The output videos showing the foreground and background point tracking will be stored in the `../vid_output_motion_binding/mymodel` directory.

The change in centre of foreground points will be saved in `../csv_motion_binding/mymodel_foreground.csv`.

The change in centre of background points will be saved in `../csv_motion_binding/mymodel_background.csv`.

They will be combined to calculate the absolute displacement of the forefround object(s).

step 4: 

```
python dot/motion_binding_score_cal.py --t2v-model mymodel --output_path ../csv_motion_binding
```
The absolute displacement of the forefround object(s) in each video will be saved in `../csv_motion_binding/mymodel_back_fore.csv`

The score for each video will be saved in `../csv_motion_binding/mymodel_score.csv`

The final score of the model in this category (motion) will be saved in the last line of `../csv_motion_binding/mymodel_score.csv`.




**Evaluate Your Own Videos**

To evaluate your own videos, prepare the evaluation videos and prompt or metadata files similar to the provided examples. Follow the same steps to run the evaluation codes.
