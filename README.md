# MiniGPT-V

<font size='5'>**MiniGPT-v2: Large Language Model as a Unified Interface for Vision-Language Multi-task Learning**</font>

Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechun Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong☨, Mohamed Elhoseiny☨

☨equal last author

<a href='https://minigpt-v2.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='https://arxiv.org/abs/2310.09478.pdf'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>  <a href='https://huggingface.co/spaces/Vision-CAIR/MiniGPT-v2'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'> <a href='https://minigpt-v2.github.io'><img src='https://img.shields.io/badge/Gradio-Demo-blue'></a> [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=atFCwV2hSY4)


<font size='5'>**MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models**</font>

Deyao Zhu*, Jun Chen*, Xiaoqian Shen, Xiang Li, Mohamed Elhoseiny

*equal contribution

<a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='https://arxiv.org/abs/2304.10592'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://huggingface.co/spaces/Vision-CAIR/minigpt4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a> <a href='https://huggingface.co/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing) [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

*King Abdullah University of Science and Technology*

## Getting Started
### Installation

**1. Prepare the code and the environment**

Git clone our repository, creating a python environment and activate it via the following command

```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigptv
```


**2. Prepare the pretrained LLM weights**

**MiniGPT-v2** is based on Llama2 Chat 7B. For **MiniGPT-4**, we have Vicuna V0 version.

|                            Llama 2 Chat 7B                             |                                           Vicuna V0 13B                                           |                                          Vicuna V0 7B                                          |
:------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:
[Download](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main) | [Downlad](https://huggingface.co/Vision-CAIR/vicuna/tree/main) | [Download](https://huggingface.co/Vision-CAIR/vicuna-7b/tree/main) 


Then, set the variable *llama_model* in the model config file to the LLM weight path.

* For MiniGPT-v2, set the LLM path 
[here](minigpt4/configs/models/minigpt_v2.yaml#L15) at Line 14.

* For MiniGPT-4 (Vicuna), set the LLM path 
[here](minigpt4/configs/models/minigpt4_vicuna0.yaml#L18) at Line 18

**3. Prepare the pretrained model checkpoints**

Download the pretrained model checkpoints


| MiniGPT-v2 (after stage-2) | MiniGPT-v2 (after stage-3) | MiniGPT-v2 (online developing demo)| 
|------------------------------|------------------------------|------------------------------|
| [Download](https://drive.google.com/file/d/1Vi_E7ZtZXRAQcyz4f8E6LtLh2UXABCmu/view?usp=sharing) |[Download](https://drive.google.com/file/d/1jAbxUiyl04SFJMN4sF1vvUU69Etuz4qa/view?usp=sharing) | [Download](https://drive.google.com/file/d/1aVbfW7nkCSYx99_vCRyP1sOlQiWVSnAl/view?usp=sharing) |


For **MiniGPT-v2**, set the path to the pretrained checkpoint in the evaluation config file 
in [eval_configs/minigptv2_eval.yaml](eval_configs/minigptv2_eval.yaml#L10) at Line 8.



| MiniGPT-4 (Vicuna 13B) | MiniGPT-4 (Vicuna 7B) |
|----------------------------|---------------------------|
| [Download](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link) | [Download](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing) |

For **MiniGPT-4**, set the path to the pretrained checkpoint in the evaluation config file 
in [eval_configs/minigpt4_eval.yaml](eval_configs/minigpt4_eval.yaml#L10) at Line 8 for Vicuna version.



### Launching Demo Locally

For MiniGPT-v2, run
```
python demo_v2.py --cfg-path eval_configs/minigptv2_eval.yaml  --gpu-id 0
```

For MiniGPT-4 (Vicuna version), run

```
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

To save GPU memory, LLMs loads as 8 bit by default, with a beam search width of 1. 
This configuration requires about 23G GPU memory for 13B LLM and 11.5G GPU memory for 7B LLM. 
For more powerful GPUs, you can run the model
in 16 bit by setting `low_resource` to `False` in the relevant config file:

* MiniGPT-v2: [minigptv2_eval.yaml](eval_configs/minigptv2_eval.yaml#6) 
* MiniGPT-4 (Vicuna): [minigpt4_eval.yaml](eval_configs/minigpt4_eval.yaml#6)


### Training
For training details of MiniGPT-4, check [here](MiniGPT4_Train.md).

For finetuning details of MiniGPT-v2, check [here](MiniGPTv2_Train.md)


### Evaluation

```
cd MiniGPT-4
pip install -v -e .
```
