import os
import sys
import socket
import ray
import math
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from typing import List, Union
from dataclasses import dataclass
from vllm import SamplingParams, LLM

from .util import split_list

@dataclass
class GenerateConfig:
    temperature: float = 0.6
    top_p: float = 0.95
    min_p: float = 0.05
    top_k: int = 40
    max_tokens: int = 1024
    min_tokens: int = 0
    stop: Union[str, list, None] = None
    n: int = 4
    logits_processors: Union[list, None] = None
    include_stop_str_in_output: bool = True

class NaiveSampler:
    def __init__(self, tokenizer, model: LLM, logger=None) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.logger = logger

    def generate(self, prompts, config: GenerateConfig) -> List:
        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            min_p=config.min_p,
            top_k=config.top_k,
            max_tokens=config.max_tokens,
            min_tokens=config.min_tokens,
            stop=config.stop,
            n=config.n,
            logits_processors=config.logits_processors,
            include_stop_str_in_output=config.include_stop_str_in_output,
        )
        outputs = self.model.generate(prompts, sampling_params)
        all_texts = []
        for output in outputs:
            # output.outputs 是一个列表，每个元素包含 .text 属性
            texts = [resp.text for resp in output.outputs]
            all_texts.append(texts)
        return all_texts

# @ray.remote(num_gpus=1, num_cpus=1)
# @ray.remote(num_gpus=0)
class InferenceActor:
    def __init__(self, tokenizer, model_path, gpu_memory_utilization, tensor_parallel_size):
        # 获取 Ray 分配的 GPU ID，并限制进程可见的 GPU
        gpu_ids = ray.get_gpu_ids()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(int(g)) for g in gpu_ids)

        # 用 vllm 包装模型，便于高效推理
        model = LLM(model=model_path, gpu_memory_utilization=gpu_memory_utilization, tensor_parallel_size=tensor_parallel_size)
        self.sampler = NaiveSampler(tokenizer, model)
        self.tokenizer = tokenizer

    def process_batch(self, batch, config):
        generated_texts = self.sampler.generate(batch, config)
        results = []
        for i, texts in enumerate(generated_texts):
            # 如果只生成一个回复，则直接取出，否则保留所有回复
            result_text = texts[0] if config.n == 1 else texts
            results.append(result_text)
        return results

class RayInferPipeline:
    def __init__(self, tokenizer, model_path, num_gpus, gpu_memory_utilization, tensor_parallel_size):
        # 如果在特定主机下需要设置环境变量（例如修改 sys.path），可以参考如下：
        if socket.gethostname() == 'skampere1':
            sys.path = ['', '/lfs/skampere1/0/brando9/miniconda/envs/beyond_scale_2/lib/python311.zip',
                        '/lfs/skampere1/0/brando9/miniconda/envs/beyond_scale_2/lib/python3.11',
                        '/lfs/skampere1/0/brando9/miniconda/envs/beyond_scale_2/lib/python3.11/lib-dynload',
                        '/lfs/skampere1/0/brando9/miniconda/envs/beyond_scale_2/lib/python3.11/site-packages',
                        '/afs/cs.stanford.edu/u/brando9/beyond-scale-2-alignment-coeff/py_src',
                        '/afs/cs.stanford.edu/u/brando9/ultimate-utils/py_src']
            print(f'{sys.path=}')

        # 初始化 Ray
        # if tensor_parallel_size == 1:
        ray.init(ignore_reinit_error=True, num_cpus=num_gpus)

        # 创建 placement group，每个 bundle 分配 1 个 GPU 和 1 个 CPU
        if num_gpus % tensor_parallel_size != 0:
            raise ValueError(f"num_gpus ({num_gpus}) is not divisible by tensor_parallel_size ({tensor_parallel_size})")
        else:
            self.num_workers = num_gpus // tensor_parallel_size
        if tensor_parallel_size == 1:
            pg = placement_group(
                name="llm_pg",
                bundles=[{"GPU": 1, "CPU": 1 } for _ in range(num_gpus)],
                strategy="STRICT_PACK"  # 可根据需求调整为 "PACK" 或 "SPREAD"
            )
            ray.get(pg.ready())

            # 创建 actor，每个 actor 分配到 placement group 的一个 bundle
            self.actors = []
            for i in range(self.num_workers):
                actor = ray.remote(num_gpus=1, num_cpus=1)(InferenceActor).options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=i
                    )
                ).remote(tokenizer, model_path, gpu_memory_utilization, tensor_parallel_size)
                self.actors.append(actor)
        else:
            self.actors = []
            for i in range(self.num_workers):
                actor = ray.remote(num_gpus=0, num_cpus=0)(InferenceActor).remote(tokenizer, model_path, gpu_memory_utilization, tensor_parallel_size)
                self.actors.append(actor)

    def split_list(self, prompts, num_workers):
        base_size = len(prompts) // num_workers  # 基础批次大小
        remainder = len(prompts) % num_workers  # 余数
        
        # 先分配前 `remainder` 个批次多一个元素
        sizes = [base_size + 1 if i < remainder else base_size for i in range(num_workers)]
        
        # 计算每个batch的起始和结束位置
        batches = []
        start = 0
        for size in sizes:
            end = start + size
            batches.append(prompts[start:end])
            start = end
    
        return batches
    
    def infer(self, prompts, config, micro_batch_size=16):
        batched_prompts = split_list(prompts, min(micro_batch_size, math.ceil(len(prompts) / self.num_workers)))
        # 轮询方式将 batch 分发给各个 actor
        tasks = []
        for i, batch in enumerate(batched_prompts):
            worker = self.actors[i % self.num_workers]
            tasks.append(worker.process_batch.remote(batch, config))

        results = []
        for task in tasks:
            results.extend(ray.get(task))
        return results
