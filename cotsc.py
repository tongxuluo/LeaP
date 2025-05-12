import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from leap import (
    RayInferPipeline,
    GenerateConfig,
    split_list,
    CotSc,
    is_correct
)

def main(
    model_path = "../models/DeepSeek-R1-Distill-Qwen-7",
    data_path = "./data/aime.json",
    save_path = "./outputs/cot/save_path.json",
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 40,
    min_p: float = 0.05,
    max_tokens: int = 2048*6,
    n: int = 8,
    num_gpus = 2,
    gpu_memory_utilization = 0.95,
    tensor_parallel_size = 2,
    batch_size = 1,
    question = "problem",
    retry_prompt = False,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if data_path.endswith('.json'):
        # 如果是JSON文件，直接加载整个文件
        with open(data_path, "r", encoding="utf-8") as file:
            test_data = json.load(file)
    elif data_path.endswith('.jsonl'):
        # 如果是JSONL文件，逐行读取并解析每行
        test_data = []
        with open(data_path, "r", encoding="utf-8") as file:
            for line in file:
                test_data.append(json.loads(line))
    batched_data = split_list(test_data, batch_size)
    gen_results = []
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ray_pipline = RayInferPipeline(tokenizer, model_path, num_gpus, gpu_memory_utilization, tensor_parallel_size)
    
    config = GenerateConfig(
        stop=[tokenizer.eos_token],
        n=n,
        max_tokens= max_tokens,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        top_k=top_k,
    )
    config_finall_answer = GenerateConfig(
        stop=[tokenizer.eos_token],
        n=1,
        max_tokens=256,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        top_k=top_k,
    )
    cot = CotSc(retry_prompt)
    for batch in tqdm(batched_data):
        results = []
        problems = [one_data.get(question, "") for one_data in batch]
        answers = [str(one_data.get("answer", "")) for one_data in batch]
        results = cot.infer_batch(batch, ray_pipline, tokenizer, config, config_finall_answer, question)
        scores = [[is_correct(solution, answers[i]) for solution in results[i]] for i in range(len(results))]
        for i in range(len(problems)):
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
