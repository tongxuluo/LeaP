from .utils import (
    RayInferPipeline,
    GenerateConfig,
    all_gather,
    get_topk_all_gather,
    contains_think_tag
)
from .prompts import GPQA_temp, leap_prefix_MATH, leap_prefix_GPQA, leap_subfix_MATH, leap_subfix_GPQA, GPQA_answer_prompt, MATH_answer_prompt
from .leap import LeaP


class LeaPS(LeaP):
    def __init__(self, num_mix: int = 8, top_k = None, router: str = "dispersed", micro_batch_size=16):
        self.num_mix = num_mix
        self.top_k = top_k
        self.micro_batch_size = micro_batch_size
        self.all_gather = all_gather if top_k is None else get_topk_all_gather(top_k, router)

    def infer_batch(self, batched_data, ray_pipeline: RayInferPipeline, tokenizer, global_n, first_config: GenerateConfig, second_config: GenerateConfig, summarize_config: GenerateConfig, question):
        # data processing
        is_gpqa = "options" in batched_data[0]
        answer_prompt = GPQA_answer_prompt if is_gpqa else MATH_answer_prompt
        results = [[] for _ in range(len(batched_data))]
        prompts = []
        for one_data in batched_data:
            if is_gpqa:
                problem = GPQA_temp.format(
                    problem=one_data[question],
                    A=one_data["options"]["A"],
                    B=one_data["options"]["B"],
                    C=one_data["options"]["C"],
                    D=one_data["options"]["D"])
                inputs = [{"role": "user", "content": problem + leap_prefix_GPQA}]
            else:
                inputs = [{"role": "user", "content": one_data[question] + " " + leap_prefix_MATH}]
            if is_gpqa:
                prompt = tokenizer.apply_chat_template(
                    inputs, tokenize=False, add_generation_prompt=True
                ) + leap_subfix_GPQA
            else:
                prompt = tokenizer.apply_chat_template(
                    inputs, tokenize=False, add_generation_prompt=True
                ) + leap_subfix_MATH
            # prompts.append(prompt)
            prompts += [prompt] * (global_n * self.num_mix)
        # first sampling
        first_sampling = ray_pipeline.infer(prompts, first_config)
        prompts = [prompts[i] + first_sampling[i] for i in range(len(first_sampling))]
        batch_counter = [self.num_mix] * len(batched_data) * global_n
        prompts = self.mixture(prompts, ray_pipeline, summarize_config, batch_counter)
        
        if self.comunicate_n > 1:
            for _ in range(self.comunicate_n - 1):
                sampling = ray_pipeline.infer(prompts, first_config)
                prompts = [prompts[i] + sampling[i] for i in range(len(sampling))]
                prompts = self.mixture(prompts, ray_pipeline, summarize_config, batch_counter)
        
        prompts = [prompts[i * self.num_mix] for i in range(len(batch_counter))]
        sampling_results = ray_pipeline.infer(prompts, second_config)
        
        need_extract = []
        index = []
        for i in range(len(batched_data)):
            for j in range(global_n):
                if not contains_think_tag(sampling_results[i * global_n + j]):
                    need_extract.append(prompts[i] + sampling_results[i * global_n + j] + answer_prompt)
                    index.append((i, j))
        
        final_results = ray_pipeline.infer(need_extract, summarize_config)
        for i in range(len(index)):
            sampling_results[index[i][0] * global_n + index[i][1]] += answer_prompt + final_results[i]
        prompts = [prompts[i] + sampling_results[i] for i in range(len(prompts))]
        
        for i in range(len(batched_data)):
            for j in range(global_n):
                results[i].append(prompts[i * global_n + j])
        return results
