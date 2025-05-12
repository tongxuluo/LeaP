from copy import deepcopy

from .utils import (
    RayInferPipeline,
    GenerateConfig,
    all_gather,
    get_topk_all_gather,
    is_stop,
    split_list_by_lengths,
    find_batch_id,
    normlize_summary
)
from .prompts import GPQA_temp, leap_prefix_MATH, leap_prefix_GPQA, leap_subfix_MATH, leap_subfix_GPQA, get_leap, GPQA_answer_prompt, MATH_answer_prompt, MATH_cot, GPQA_cot


class LeaP:
    def __init__(self, max_turns: int, top_k = None, router: str = "dispersed", part: list = [], cot_prompt=False, micro_batch_size=16):
        self.max_turns = max_turns
        self.top_k = top_k
        self.all_gather = all_gather if top_k is None else get_topk_all_gather(top_k, router)
        self.part = part
        self.cot_prompt = cot_prompt
        self.micro_batch_size = micro_batch_size

    def mixture(self, prompts, ray_pipeline, summarize_config, batch_counter):
        # summarize
        prompts = ["\n\n".join(p.split("\n\n")[:-1]) + "\n\n" + get_leap() for p in prompts]
        summarize_results = ray_pipeline.infer(prompts, summarize_config, micro_batch_size=self.micro_batch_size)
        summarize_results = normlize_summary(summarize_results)
        prompts = [prompts[i] + summarize_results[i] for i in range(len(summarize_results))]
        # all gather
        batched_prompts = split_list_by_lengths(batch_counter, prompts)
        prompts = []
        for one in batched_prompts:
            prompts += self.all_gather(one)
        return prompts

    def select_next_turn(self, prompts, sampling_results, results, batch_counter):
        next_turn = []
        next_batch_counter = deepcopy(batch_counter)
        for i in range(len(sampling_results)):
            if is_stop(sampling_results[i], "</think>"):
                batch_id = find_batch_id(i, batch_counter)
                results[batch_id].append(prompts[i] + sampling_results[i])
                next_batch_counter[batch_id] -= 1
                continue
            next_turn.append(prompts[i] + sampling_results[i])
        return next_turn, next_batch_counter, results

    def infer_batch(self, batched_data, ray_pipeline: RayInferPipeline, tokenizer, config: GenerateConfig, summarize_config: GenerateConfig,question):
        # data processing
        is_gpqa = "options" in batched_data[0]
        answer_prompt = GPQA_answer_prompt if is_gpqa else MATH_answer_prompt
        results = [[] for _ in range(len(batched_data))]
        sub_config = deepcopy(config)
        sub_config.n = 1
        prompts = []
        for one_data in batched_data:
            if is_gpqa:
                problem = GPQA_temp.format(
                    problem=one_data[question],
                    A=one_data["options"]["A"],
                    B=one_data["options"]["B"],
                    C=one_data["options"]["C"],
                    D=one_data["options"]["D"])
                if self.cot_prompt:
                    inputs = [{"role": "user", "content": problem + GPQA_cot}]
                else:
                    inputs = [{"role": "user", "content": problem + leap_prefix_GPQA}]
            else:
                if self.cot_prompt:
                    inputs = [{"role": "user", "content": one_data[question] + " " + MATH_cot}]
                else:
                    inputs = [{"role": "user", "content": one_data[question] + " " + leap_prefix_MATH}]
            prompt = tokenizer.apply_chat_template(
                inputs, tokenize=False, add_generation_prompt=True
            )
            if not self.cot_prompt:
                prompt += (leap_subfix_GPQA if is_gpqa else leap_subfix_MATH)
                
            prompts.append(prompt)

        # first sampling
        first_sampling = ray_pipeline.infer(prompts, config, micro_batch_size=self.micro_batch_size)
        batch_counter = [config.n] * len(prompts)
        new_prompts = []
        for p in prompts:
            new_prompts += [p] * config.n
        prompts = new_prompts
        first_sampling = [one for sublist in first_sampling for one in sublist]
        prompts, batch_counter, results = self.select_next_turn(prompts, first_sampling, results, batch_counter)
        
        if "0" in self.part or self.part == []:
            prompts = self.mixture(prompts, ray_pipeline, summarize_config, batch_counter)
        
        for turn in range(self.max_turns - 1):
            # second sampling
            sampling_results = ray_pipeline.infer(prompts, sub_config, micro_batch_size=self.micro_batch_size)
            # select
            prompts, batch_counter, results = self.select_next_turn(prompts, sampling_results, results, batch_counter)
            if len(prompts) == 0 or turn == self.max_turns - 2:
                break
            if f"{turn + 1}" in self.part or self.part == []:
                prompts = self.mixture(prompts, ray_pipeline, summarize_config, batch_counter)
                
        if len(prompts):
            prompts = [prompts[i] + answer_prompt for i in range(len(prompts))]
            finial_results = ray_pipeline.infer(prompts, summarize_config, micro_batch_size=self.micro_batch_size)
            prompts = [prompts[i] + finial_results[i] for i in range(len(finial_results))]
            for i in range(len(prompts)): 
                batch_id = find_batch_id(i, batch_counter)
                results[batch_id].append(prompts[i])
        
        return results
