import Levenshtein
import random
from ..prompts import comment_subfix


def contains_think_tag(s: str) -> bool:
    """
    判断字符串是否包含 '</think>' 标签。

    :param s: 输入字符串
    :return: 如果包含 '</think>' 返回 True，否则返回 False
    """
    return "</think>" in s

    
def normlize_summary(str_list):
    str_list = [s.replace("</think>", "") for s in str_list]
    target = '</summarize>'
    result = []
    for s in str_list:
        max_k = 0
        for k in range(len(target), 1, -1):
            if len(s) >= k and s.endswith(target[:k]):
                max_k = k
                break
        if max_k > 0:
            s += target[max_k:]
        else:
            s += '... ' + target
        result.append(s)
    return result


def split_list(input_list, batch_size):
    total_length = len(input_list)
    # 如果列表长度大于batch_size
    if total_length > batch_size:
        result = [input_list[i:i + batch_size] for i in range(0, total_length, batch_size)]
    else:
        result = [input_list]
    
    return result


def all_gather(prompts):
    # Extract summary
    if len(prompts) > 1:
        summaries = [prompt.split("<summarize>")[-1].split("</summarize>")[0].strip() for prompt in prompts]
    
        result = [
            " <comment> " + "\n".join(f"Peer {i+1}: \"{summaries[j]}\"" for i, j in enumerate([j for j in range(len(summaries)) if j != k])) + " </comment>\n\n" + comment_subfix
            for k in range(len(summaries))
        ]
    
        for i in range(len(prompts)):
            prompts[i] = prompts[i] + result[i]
    elif len(prompts):
        prompts[0] = prompts[0] + " <comment> No comments </comment>\n\nHmm, there are no comments, so let's continue reasoning."
    return prompts


def get_topk_all_gather(top_k=2, router="dispersed"):
    
    def topk_all_gather(prompts):
        if len(prompts) > 1:
            summaries = [p.split("<summarize>")[-1].split("</summarize>")[0].strip() for p in prompts]
            
            results = []
            for idx, current_summary in enumerate(summaries):
                other_indices = [jdx for jdx in range(len(summaries)) if jdx != idx]
                
                if router == "dispersed":
                    similarities = [(Levenshtein.ratio(current_summary, summaries[jdx]), jdx) for jdx in other_indices]
                    similarities.sort(key=lambda x: x[0])  # ascending
                    selected_indices = [j for _, j in similarities[:top_k]]

                elif router == "clustered":
                    similarities = [(Levenshtein.ratio(current_summary, summaries[jdx]), jdx) for jdx in other_indices]
                    similarities.sort(key=lambda x: -x[0])  # descending
                    selected_indices = [j for _, j in similarities[:top_k]]

                elif router == "random":
                    selected_indices = random.sample(other_indices, min(top_k, len(other_indices)))

                elif router == "hybrid":
                    similarities = [(Levenshtein.ratio(current_summary, summaries[jdx]), jdx) for jdx in other_indices]
                    similarities_sorted = sorted(similarities, key=lambda x: x[0])  # ascending for dissimilar
                    half_k = top_k // 2
                    dissimilar_indices = [j for _, j in similarities_sorted[:half_k]]
                    similar_indices = [j for _, j in sorted(similarities, key=lambda x: -x[0])[:top_k - half_k]]
                    selected_indices = dissimilar_indices + similar_indices

                else:
                    raise ValueError(f"Unknown router type: {router}")
                
                comment_str = (
                    " <comment> " +
                    "\n".join(f"Peer {i+1}: \"{summaries[j]}\"" for i, j in enumerate(selected_indices)) +
                    " </comment>\n\n" + comment_subfix
                )
                results.append(comment_str)
            
            for i in range(len(prompts)):
                prompts[i] = prompts[i] + results[i]
        elif prompts:
            prompts[0] = prompts[0] + " <comment> No comments </comment>\n\nHmm, there are no comments, so let's continue reasoning."
        return prompts

    return topk_all_gather


def is_stop(p, stop_token):
    if p[-len(stop_token):] == stop_token:
        return True
    else:
        return False


def split_list_by_lengths(int_list, str_list):
    result = []
    start_idx = 0

    for length in int_list:
        # 根据当前的长度分割 str_list 中的部分
        end_idx = start_idx + length
        result.append(str_list[start_idx:end_idx])
        start_idx = end_idx  # 更新起始位置为分割后的下一个位置

    return result

def find_batch_id(i, length_list):
    start = 0
    for index, length in enumerate(length_list):
        end = start + length - 1  # 计算当前区间的结束位置
        if start <= i <= end:
            return index  # 如果 i 在当前区间内，返回对应的索引
        start = end + 1  # 更新下一个区间的起始位置
    return -1  # 如果 i 不在任何区间内，返回 -1
