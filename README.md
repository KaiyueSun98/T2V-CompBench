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

#### 2. Prepare Evaluation Videos

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

#### 3. Run the Evaluation Codes

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

The output will be a CSV file named f"{mymodel}_consistent_attr_score.csv" in the "../csv_output_consistent_attr" directory. The video name, prompt, and score for each text-video pair will be recorded in the columns named of "name","prompt", "Score".

##### Action Binding

Input the video path and run:

```
python llava/eval/eval_action_binding.py --video-path ../video/action_binding --output-path ../csv_output_action_binding --read-prompt-file ../meta_data/action_binding.json --t2v-model mymodel
```

The output will be a CSV file named f"{mymodel}_action_binding_score.csv" in the "../csv_output_action_binding" directory. The video name, prompt, and score for each text-video pair will be recorded in the columns named of "name","prompt", "Score".

##### Object Interaction

Input the video path and run:

```
python llava/eval/eval_interaction.py --video-path ../video/interaction --output-path ../csv_output_object_interactions --read-prompt-file ../meta_data/object_interactions.json --t2v-model mymodel
```

The output will be a CSV file named f"{mymodel}_object_interactions_score.csv" in the "../csv_output_object_interactions" directory. The video name, prompt, and score for each text-video pair will be recorded in the columns named of "name","prompt", "Score".


