
from .utils import RayInferPipeline, GenerateConfig, contains_think_tag
from .prompts import GPQA_temp, MATH_cot, GPQA_cot, moa_template, MATH_stop_think


class MoA:
    def __init__(self, num_layers=3, num_agents=3):
        self.num_layers = num_layers
        self.num_agents = num_agents

    def ensure_stop_think(self, prompts, config, ray_pipeline):
        need_stop = []
        index = []
        for i in range(len(prompts)):
            if not contains_think_tag(prompts[i]):
                need_stop.append(prompts[i] + MATH_stop_think)
                index.append(i)
        results = ray_pipeline.infer(need_stop, config)
        for idx, result in zip(index, results):
            prompts[idx] = prompts[idx] + MATH_stop_think + result
        return prompts

    def extract_response(self, prompts):
        return [p.split("</think>")[-1].strip() for p in prompts]

    def mixture(self, extracted_responses, batched_data, tokenizer, question, is_last=False):
        new_prompts = []
        is_gpqa = "options" in batched_data[0]
        for i, one_data in enumerate(batched_data):
            if is_gpqa:
                problem = GPQA_temp.format(
                    problem=one_data[question],
                    A=one_data["options"]["A"],
                    B=one_data["options"]["B"],
                    C=one_data["options"]["C"],
                    D=one_data["options"]["D"])
                problem = problem + GPQA_cot
            else:
                problem = one_data[question] + " " + MATH_cot
            
            moa_problem = moa_template.format(problem=problem)
            prelayer_res = []
            for j in range(self.num_agents):
                prelayer_res.append(f"Model {j+1}:" + extracted_responses[self.num_agents * i + j])
            inputs = [{"role": "user", "content": moa_problem + '\n'.join(prelayer_res)}]
            prompt = tokenizer.apply_chat_template(
                inputs, tokenize=False, add_generation_prompt=True
            )
            if is_last:
                new_prompts += [prompt]
            else:
                new_prompts += [prompt] * self.num_agents
        return new_prompts

    def save_inter(self, prompts, intermediate):
        for i in range(len(intermediate)):
            for j in range(self.num_agents):
                intermediate[i].append(prompts[self.num_agents * i + j])

    def infer_batch(self, batched_data, ray_pipeline: RayInferPipeline, tokenizer, config: GenerateConfig, config_finall_answer: GenerateConfig, question):
        is_gpqa = "options" in batched_data[0]
        prompts = []
        intermediate = [[] for _ in range(len(batched_data))]
        for one_data in batched_data:
            if is_gpqa:
                problem = GPQA_temp.format(
                    problem=one_data[question],
                    A=one_data["options"]["A"],
                    B=one_data["options"]["B"],
                    C=one_data["options"]["C"],
                    D=one_data["options"]["D"])
                inputs = [{"role": "user", "content": problem + GPQA_cot}]
            else:
                inputs = [{"role": "user", "content": one_data[question] + " " + MATH_cot}]

            prompt = tokenizer.apply_chat_template(
                inputs, tokenize=False, add_generation_prompt=True
            )
            # prompts.append(prompt)
            prompts += [prompt] * self.num_agents
        
        sampling = ray_pipeline.infer(prompts, config)
        prompts = [prompts[i] + sampling[i] for i in range(len(sampling))]
        prompts = self.ensure_stop_think(prompts, config_finall_answer, ray_pipeline)
        for layer_id in range(1, self.num_layers):
            self.save_inter(prompts, intermediate)
            extracted_responses = self.extract_response(prompts)
            # mix with responses -> last or not last
            is_last = True if layer_id == self.num_layers - 1 else False
            prompts = self.mixture(extracted_responses, batched_data, tokenizer, question, is_last)
            # gen 
            sampling = ray_pipeline.infer(prompts, config)
            prompts = [prompts[i] + sampling[i] for i in range(len(sampling))]
            prompts = self.ensure_stop_think(prompts, config_finall_answer, ray_pipeline)
            
        return prompts, intermediate
