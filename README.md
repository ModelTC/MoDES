<div align="center" style="font-family: charter;">
<h1> MoDES:<br>Accelerating Mixture-of-Experts Multimodal Large Language Models
via Dynamic Expert Skipping</h1>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv-2511.15690-b31b1b)](https://arxiv.org/abs/2511.15690)&nbsp;
[![GitHub Stars](https://img.shields.io/github/stars/ModelTC/MoDES.svg?style=social&label=Star&maxAge=60)](https://github.com/ModelTC/MoDES)&nbsp;

**[ [Conference Paper](https://arxiv.org/abs/2511.15690) ]**

[Yushi Huang](https://Harahan.github.io), [Zining Wang](https://scholar.google.com/citations?user=DJRj6M0AAAAJ&hl=zh-CN), [Zhihang Yuan](https://zhihang.cc/)📧, [Ruihao Gong](https://xhplus.github.io/), [Yifu Ding](https://yifu-ding.github.io/), [Jinyang Guo](https://jinyangguo.github.io/), [Xianglong Liu](https://xlliu-beihang.github.io/), [Jun Zhang](https://eejzhang.people.ust.hk/)📧

(📧 denotes corresponding author.)

</div>

This is the official implementation of our paper [MoDES](https://arxiv.org/abs/2511.15690), the first training-free framework that adaptively skips experts to enable efficient and accurate MoE MLLM inference. Through extensive experiments on 3 model series across 13 benchmarks, MoDES significantly outperforms prior methods. For example, when skipping 88% of experts for Qwen3-VL-MoE-30B-A3B-Instruct, MoDES achieves a performance boost of up to **10.67%** (97.33% vs. 86.66%). It also improves inference speed with **2.16×** prefilling and **1.26×** decoding speedup.

## 📖 Overview

<div align="center" style="font-family: charter;">

<img src=./assets/overview.png width="80%"/>

<h align="justify"><strong>Overview pipeline of the proposed MoDES.</strong> At inference, use a text token (<i>e.g.</i>, blue square above) at the $l$-th FFN layer as an example. (a) We compute importance scores $s^{(l)}\_i$ ($i\in\\{2, 4, M\\}$) by combining the offline-calibrated globally-modulated factor $\alpha^{(l)}$ with the local routing probability $\pi^{(l)}\_i$. These scores evaluate the top- $k$ ($k=3$) routed experts for the token. (b) We then apply a modality-specific threshold— $\tau\_{\text{t}}$ for text and $\tau\_{\text{v}}$ for vision—found by an efficient and effective <i>frontier search</i>. Experts with scores below the threshold are skipped. This method significantly reduces computation while preserving performance for MoE MLLMs.
</h>

</div>

## :fire: News

* **Feb 20, 2026**: 🔥 We release the Python code for expert skipping presented in our paper. Have a try!

* **Feb 20, 2026**: 🌟 Our paper has been accepted by CVPR 2026! 🎉 Cheers!

## ✨ Quick Start

### Requirements

8× H100/H200/H800/A100/A800 GPUs (adjust scripts as needed for fewer GPUs)

### Installation

```shell
conda create -n modes python=3.11 -y
conda activate modes
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
pip install -e ./lmms-eval
pip install qwen-vl-utils==0.0.14
```

For Qwen3-VL, use the latest `transformers`:

```shell
git clone https://github.com/huggingface/transformers.git
pip install -e transformers/
```

### Data and Models

```shell
# Login to huggingface-cli first
mkdir -p storage/models/Kimi-VL-A3B-Instruct
# Replace with Qwen3-VL-MoE for that model
huggingface-cli download moonshotai/Kimi-VL-A3B-Instruct --local-dir ./storage/models/Kimi-VL-A3B-Instruct

# Download GQA (or VideoMMMU / COCO similarly)
mkdir -p storage/datasets/GQA/testdev_balanced_images
mkdir -p storage/datasets/GQA/testdev_balanced_instructions
huggingface-cli download lmms-lab/GQA testdev_balanced_images/testdev-00000-of-00001.parquet --repo-type dataset ./storage/datasets/GQA/testdev_balanced_images
huggingface-cli download lmms-lab/GQA testdev_balanced_instructions/testdev-00000-of-00001.parquet --repo-type dataset ./storage/datasets/GQA/testdev_balanced_instructions
```

### Calibration and Frontier Search

```shell
export prefix=/path/to/your/dir
export PYTHONPATH=$prefix:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Step 1: Calibration (compute layer importance)
accelerate launch --num_processes 8 --main_process_port 5678 \
    get_layer_importance_ddp.py \
    --name_or_path $prefix/storage/models/Kimi-VL-A3B-Instruct \
    --save_dir $prefix/storage/importance \
    --dataset gqa \
    --loss_type kl \
    --batch_size 8 \
    --num_samples 1024

# Step 2: Frontier search (find optimal tau)
export target_skip_proportion=0.8  # or 0.7, 0.6, etc.
accelerate launch --num_processes 8 --main_process_port 5678 \
    grid_search_tau_ddp.py \
    --name_or_path $prefix/storage/models/Kimi-VL-A3B-Instruct \
    --save_dir $prefix/storage/search/ \
    --dataset gqa \
    --loss_type kl \
    --batch_size 8 \
    --num_samples 1024 \
    --layer_importance_path path/to/importance/pkl \
    --target_skip_proportion $target_skip_proportion \
    --grid_num 100 \
    --grid_map exp
```

### Evaluation

```shell
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$prefix:$PYTHONPATH
export DECORD_EOF_RETRY_MAX=204800

accelerate launch --main_process_port 5678 \
    eval/kimi.py \
    --model kimi_vl \
    --model_args pretrained=storage/models/Kimi-VL-A3B-Instruct,tau_skip_path=path/to/tau,layer_importance_path=path/to/importance \
    --tasks videomme,gqa,chartqa \
    --batch_size 1 \
    --output_path $prefix/storage/eval/
```

>[!NOTE] 
>Replace `eval/kimi.py` with `eval/qwen3.py`, `kimi_vl` with `qwen3_vl` and `storage/models/Kimi-VL-A3B-Instruct` with `storage/models/Qwen3-VL-30B-A3B-Instruct` for Qwen3-VL.

## 💪 TODO

- [ ] Fast CUDA implementation for MoE layers.
- [ ] Code for InternVL series.

## 🙏 Acknowledgments

Our code was developed based on [transformers](https://github.com/huggingface/transformers) and [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).


## 📝 Citation

If you find MoDES useful in your research, please cite:

```bibtex
@InProceedings{huang2025modesacceleratingmixtureofexpertsmultimodal,
    title = {MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping}, 
    author = {Yushi Huang and Zining Wang and Zhihang Yuan and Yifu Ding and Ruihao Gong and Jinyang Guo and Xianglong Liu and Jun Zhang},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2026},
}
```