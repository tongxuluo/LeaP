from .utils import RayInferPipeline, GenerateConfig, contains_think_tag
from .prompts import GPQA_temp, MATH_cot, GPQA_cot, GPQA_answer_prompt, MATH_answer_prompt, GPQA_retry, MATH_retry

class CotSc:
    def __init__(self, retry_prompt=False):
        self.retry_prompt = retry_prompt

    def infer_batch(self, batched_data, ray_pipeline: RayInferPipeline, tokenizer, config: GenerateConfig, config_finall_answer: GenerateConfig, question):
        is_gpqa = "options" in batched_data[0]
        answer_prompt = GPQA_answer_prompt if is_gpqa else MATH_answer_prompt
        prompts = []
        for one_data in batched_data:
            if is_gpqa:
                problem = GPQA_temp.format(
                    problem=one_data[question],
                    A=one_data["options"]["A"],
                    B=one_data["options"]["B"],
                    C=one_data["options"]["C"],
                    D=one_data["options"]["D"])
                if self.retry_prompt:
                    inputs = [{"role": "user", "content": problem + GPQA_retry}]
                else:
                    inputs = [{"role": "user", "content": problem + GPQA_cot}]
            else:
                if self.retry_prompt:
                    inputs = [{"role": "user", "content": one_data[question] + " " + MATH_retry}]
                else:
                    inputs = [{"role": "user", "content": one_data[question] + " " + MATH_cot}]

            prompt = tokenizer.apply_chat_template(
                inputs, tokenize=False, add_generation_prompt=True
            )
            
            prompts.append(prompt)
        solution = ray_pipeline.infer(prompts, config)
        need_extract = []
        index = []
        for i in range(len(solution)):
            for j in range(len(solution[i])):
                if not contains_think_tag(solution[i][j]):
                    need_extract.append(prompts[i] + solution[i][j] + answer_prompt)
                    index.append((i, j))
        final_results = ray_pipeline.infer(need_extract, config_finall_answer)
        for i in range(len(index)):
            solution[index[i][0]][index[i][1]] += answer_prompt + final_results[i]
        return solution
