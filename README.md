<div align="center">
  <img src="https://github.com/tongxuluo/LeaP/blob/main/LeaP.png" width="256" height="256">
</div>

# LeaP: Learning from Peers Reasoning

Official implementation for the paper **"LeaP: Collaborative Reasoning Among Language Models via Peer Learning"** 


<p align="center">
<a href="https://github.com/tongxuluo/LeaP/blob/main/LICENSE">
<img src='https://img.shields.io/badge/License-Apache_2.0-green.svg'></a>
<img src='https://img.shields.io/badge/Python-3.10+-blue.svg'>
</p>

<p align="center">
🔔 <a href="https://github.com/tongxuluo/LeaP" target="_blank">Code</a> • 📃 <a href="https://arxiv.org/abs/2405.15319" target="_blank">Paper</a> • 🤗 <a href="https://huggingface.co/Learning-from-Peers" target="_blank">Model</a>
</p>



## 🧠 Abstract
Large Reasoning Models (LRMs) have the ability to self-correct even when they make mistakes in their reasoning paths.  However, our study reveals that when the reasoning process starts with a short but poor beginning, it becomes difficult for the model to recover.  We refer to this phenomenon as the *"Prefix Dominance Trap"*.  Inspired by psychological findings that peer interaction can promote self-correction without negatively impacting already accurate individuals, we propose **Learning from Peers** (LeaP) to address this phenomenon.  Specifically, every T tokens, each reasoning path summarizes its intermediate reasoning and shares it with others through a routing mechanism, enabling paths to incorporate peer insights during inference.  However, we observe that smaller models sometimes fail to follow summarization and reflection instructions effectively.  To address this, we fine-tune them into our **LeaP-T** model series.  Experiments on AIME 2024, AIME 2025, AIMO 2025, and GPQA Diamond show that LeaP provides substantial improvements.  For instance, QwQ-32B with LeaP achieves nearly 5 absolute points higher than the baseline on average, and surpasses DeepSeek-R1-671B on three math benchmarks with an average gain of 3.3 points.  Notably, our fine-tuned LeaP-T-7B matches the performance of DeepSeek-R1-Distill-Qwen-14B on AIME 2024.  In-depth analysis reveals LeaP's robust error correction by timely peer insights, showing strong error tolerance and handling varied task difficulty.  LeaP marks a milestone by enabling LRMs to collaborate during reasoning.  Our code, datasets, and models are available at [https://learning-from-peers.github.io/](https://learning-from-peers.github.io/).


## 📐 LeaP Framework

<p align="center">
  <img src="https://github.com/tongxuluo/LeaP/blob/main/main.png" width="80%">
</p>

---

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/tongxuluo/LeaP.git
cd LeaP
pip install -r requirements.txt
```



### Model Inference
We provide scripts to run different inference modes using the 7B model. Below are example commands:
#### Run Inference with LeaP
```
bash scripts/leap_7B.sh
```
#### Run Inference with LeaP-T
```
bash scripts/leap_t_7B.sh
```
#### Run Inference of Independence
```
bash scripts/leap_7B.sh
```
### Model fine-tune
You can fine-tune smaller models for better summarization and reflection using the provided script:
```
bash scripts/leap_s_7B.sh
```
## TODO
- ✅ Open source our code, datasets and V1 models -- 2025.5.10
- [] Publish our LeaP-R1 models trained by RL -- 2025.8

 

## Acknowledgement


## Citation
If you find our paper inspiring and have utilized it in your work, please cite our paper.
```
@article{du2024stacking,
      title={Stacking Your Transformers: A Closer Look at Model Growth for Efficient LLM Pre-Training},
      author={Du, Wenyu and Luo, Tongxu and Qiu, Zihan and Huang, Zeyu and Shen, Yikang and Cheng, Reynold and Guo, Yike and Fu, Jie},
      journal={arXiv preprint arXiv:2405.15319},
      year={2024}
    }
```
