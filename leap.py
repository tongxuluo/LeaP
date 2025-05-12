import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer

from leap import (
    RayInferPipeline,
    GenerateConfig,
    is_correct,
    split_list,
    LeaP,
)


def main(
    model_path = "../models/DeepSeek-R1-Distill-Qwen-7B",
    data_dir = "./data",
    save_dir = "./outputs/",
    tasks = "aime",
    max_turns: int = 6,
    peer_top_k: int = -1,
    router: str = "dispersed", # ["dispersed", "similar", "random", "hybrid"]
    top_k: int = 40,
    temperature: float = 0.6,
    top_p: float = 0.95,
    min_p: float = 0.05,
    max_tokens: int = 2048,
    part: str = "",
    summarize_max_tokens: int = 256,
    n: int = 4,
    num_gpus = 2,
    gpu_memory_utilization = 0.95,
    tensor_parallel_size = 1,
    batch_size = 4,
    question = "problem",
    resume = False,
    is_leap_t_model = False,
    micro_batch_size = 16
):
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ray_pipline = RayInferPipeline(tokenizer, model_path, num_gpus, gpu_memory_utilization, tensor_parallel_size)
    config = GenerateConfig(
        stop=[tokenizer.eos_token, "</think>", "<summarize>"] if is_leap_t_model else [tokenizer.eos_token, "</think>"],
        n=n,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        top_k=top_k,
    )
    summarize_config = GenerateConfig(
        stop=[tokenizer.eos_token, "</summarize>", "</think>"],
        n=1,
        max_tokens=summarize_max_tokens,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        top_k=top_k,
    )
    part = [] if part == "" else part.split(",")
    
    leap_infer = LeaP(max_turns, peer_top_k if peer_top_k != -1 else None, router, part, True if is_leap_t_model else False, micro_batch_size)

    tasks = tasks.split(",")
    for task in tasks:
        data_path = os.path.join(data_dir, f"{task}.json")
        save_path = os.path.join(save_dir, f"{task}.json")
        with open(data_path, "r", encoding="utf-8") as file:
            test_data = json.load(file)
        # resume
        gen_results = []
        if resume and os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as file:
                gen_results = json.load(file)
        existing_question = [one["problem"] for one in gen_results]
        test_data = [one for one in test_data if one[question] not in existing_question]

        batched_data = split_list(test_data, batch_size)
    
        for batch in tqdm(batched_data):
            if len(batch) == 0:
                continue
            results = []
            problems = [one_data.get(question, "") for one_data in batch]
            answers = [str(one_data.get("answer", "")) for one_data in batch]
            results = leap_infer.infer_batch(batch, ray_pipline, tokenizer, config, summarize_config , question)
            scores = [[is_correct(solution, answers[i]) for solution in results[i]] for i in range(len(results))]
    
            for i in range(len(results)):
                if "options" in batch[i]:
                    gen_results.append({
                        "problem": problems[i],
                        "answer": answers[i],
                        "options": batch[i]["options"],
                        "solutions": results[i],
                        "scores_all": scores[i],
                    })
                else:
                    gen_results.append({
                        "problem": problems[i],
                        "answer": answers[i],
                        "solutions": results[i],
                        "scores_all": scores[i],
                    })
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(gen_results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
