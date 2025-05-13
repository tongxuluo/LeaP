<div align="center">
  <img src="https://github.com/tongxuluo/LeaP/blob/main/figures/logo.png" width="256" height="256" alt="LeaP Logo">
</div>

# LeaP: Learning from Peers in Reasoning Models

Official implementation for the paper **"Learning from Peers in Reasoning Models"**

<p align="center">
<a href="https://github.com/tongxuluo/LeaP/blob/main/LICENSE">
<img src='https://img.shields.io/badge/License-Apache_2.0-green.svg' alt='License: Apache 2.0'></a>
<img src='https://img.shields.io/badge/Python-3.10+-blue.svg' alt='Python 3.10+'>
<a href="https://arxiv.org/abs/2405.15319" target="_blank">
<img src="https://img.shields.io/badge/arXiv-2505.07787-b31b1b.svg" alt="arXiv:2505.07787"></a>
<a href="https://huggingface.co/Learning-from-Peers" target="_blank">
<img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Models%20%26%20Datasets-yellow.svg" alt="Hugging Face Models & Datasets"></a>
</p>

<p align="center">
<a href="https://learning-from-peers.github.io/" target="_blank">🌐 Project Page</a> •
<a href="https://github.com/tongxuluo/LeaP" target="_blank">💻 Code</a> •
<a href="https://arxiv.org/abs/2505.07787" target="_blank">📃 Paper</a> •
<a href="https://huggingface.co/Learning-from-Peers" target="_blank">🤗 Datasets & Models</a>
</p>

## 🧠 Abstract
Large Reasoning Models (LRMs) have the ability to self-correct even when they make mistakes in their reasoning paths. However, our study reveals that when the reasoning process starts with a short but poor beginning, it becomes difficult for the model to recover. We refer to this phenomenon as the *"Prefix Dominance Trap"*. Inspired by psychological findings that peer interaction can promote self-correction without negatively impacting already accurate individuals, we propose **Learning from Peers (LeaP)** to address this phenomenon. Specifically, every T tokens, each reasoning path summarizes its intermediate reasoning and shares it with others through a routing mechanism, enabling paths to incorporate peer insights during inference. However, we observe that smaller models sometimes fail to follow summarization and reflection instructions effectively. To address this, we fine-tune them into our **LeaP-T** model series. Experiments on AIME 2024, AIME 2025, AIMO 2025, and GPQA Diamond show that LeaP provides substantial improvements. For instance, QwQ-32B with LeaP achieves nearly 5 absolute points higher than the baseline on average, and surpasses DeepSeek-R1-671B on three math benchmarks with an average gain of 3.3 points. Notably, our fine-tuned LeaP-T-7B matches the performance of DeepSeek-R1-Distill-Qwen-14B on AIME 2024. In-depth analysis reveals LeaP's robust error correction by timely peer insights, showing strong error tolerance and handling varied task difficulty. LeaP marks a milestone by enabling LRMs to collaborate during reasoning.

## ✨ Key Features

* **Addresses the "Prefix Dominance Trap"**: LeaP helps LRMs recover from poor initial reasoning steps.
* **Collaborative Reasoning**: Enables multiple reasoning paths to share insights and learn from each other during inference.
* **LeaP-T Models**: A series of fine-tuned models (1.5B, 7B, 14B) that are optimized for the LeaP framework, showing improved summarization and reflection capabilities, especially for smaller model sizes.
* **Significant Performance Gains**:
    * QwQ-32B with LeaP achieves nearly **+5 absolute points** on average over baseline.
    * QwQ-32B with LeaP surpasses the much larger DeepSeek-R1-671B on three math benchmarks.
    * Our LeaP-T-7B model matches the performance of DeepSeek-R1-Distill-Qwen-14B on AIME 2024.
* **Robust Error Correction**: Demonstrates strong error tolerance and effectiveness across varied task difficulties through timely peer insights.

## 📐 LeaP Framework Explained

The LeaP framework enhances Large Reasoning Models (LRMs) by enabling cross-path interaction during parallel inference. Instead of relying solely on individual reasoning paths, LeaP allows LRMs to share and incorporate insights from their "peers" (i.e., other concurrently generated reasoning paths for the same problem).

<p align="center">
  <img src="https://github.com/tongxuluo/LeaP/blob/main/figures/main.png" width="80%" alt="LeaP Framework Diagram">
</p>

The core mechanism involves strategically inserting **LeaP blocks** into the parallel inference process. Each LeaP block orchestrates a two-stage process:
1.  **Summarization**: Every T tokens, each reasoning path condenses its current state, key insights, and intermediate results into a concise summary (e.g., limited to 256 tokens). This is guided by dynamic prompts.
2.  **Routing**: These summaries are then disseminated to other peer paths. We explore various heuristic routing mechanisms (e.g., *Dispersed Routing* to maximize diverse perspectives, *Clustered Routing* to reinforce promising trajectories, and *Hybrid Routing*) to facilitate effective collaboration. Each path then uses the received peer summaries for self-verification and refinement before continuing its generation.

This structured peer interaction extends the model's self-verification capabilities beyond its own reasoning to include the reasoning of others, broadening the search space for refinement and reducing cognitive burden.

---

## 🚀 Getting Started

### Installation
Clone the repository and set up the environment:
```bash
git clone [https://github.com/tongxuluo/LeaP.git](https://github.com/tongxuluo/LeaP.git)
cd LeaP
conda create -n LeaP python==3.10
conda activate LeaP
pip install -r requirements.txt
```

### Inference with LeaP
We provide scripts to run different inference modes. The following examples use a 7B model. You can adapt these scripts for other model sizes or configurations.

#### Independent Reasoning
```
bash scripts/cot_7B.sh
```

#### Reasoning with MoA
```
bash scripts/moa_7B.sh
```

#### LeaP for Non-Trained LRMs 
```
bash scripts/leap_7B.sh
```

#### LeaP-T Series
This script uses our fine-tuned LeaP-T models, which are already adapted for the LeaP framework. Ensure you have downloaded the LeaP-T model weights (available on Hugging Face) and updated the script with the correct model path.
```
bash scripts/leap_t_7B.sh
```

#### LeaP in Single Generation
```
bash scripts/leap_s_7B.sh
```

## TODO
- [x] Open source our code, datasets and models -- 2025.5.13 .
- [ ] Publish our LeaP-R1 models trained by RL -- 2025.8 .


## Citation
If you find our paper inspiring and have utilized it in your work, please cite our paper.
```
@article{luo2025learning,
  title={Learning from Peers in Reasoning Models},
  author={Luo, Tongxu and Du, Wenyu and Bi, Jiaxi and Chung, Stephen and Tang, Zhengyang and Yang, Hao and Zhang, Min and Wang, Benyou},
  journal={arXiv preprint arXiv:2505.07787},
  year={2025}
}
```