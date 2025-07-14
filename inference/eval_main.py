import json

import ray
import torch
from ray.util.placement_group import (PlacementGroupSchedulingStrategy,
                                      placement_group)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from vllm_worker import MyLLM
from collections import defaultdict

import sys

sys.path.append('.')
from verl.utils.reward_score import _default_compute_score


def red_print(content):
    red_start = '\033[31m'
    red_end = '\033[0m'
    print(f"{red_start}{content}{red_end}")


def vllm_init(model_path, num_infer_workers):
    infer_tp_size = 1
    pg_inference = [placement_group([{"GPU": 1, "CPU": 8}] * infer_tp_size, strategy="STRICT_PACK") for _ in
                    range(num_infer_workers)]
    for pg in pg_inference:
        ray.get(pg.ready())
    infer_llms = []
    for i, pg in tqdm(enumerate(pg_inference)):
        strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=0,
            placement_group_capture_child_tasks=True
        )
        llm = ray.remote(
            num_cpus=0,
            num_gpus=0,
            scheduling_strategy=strategy,
        )(MyLLM).remote(
            model=model_path,
            enforce_eager=False,
            tensor_parallel_size=infer_tp_size,
            distributed_executor_backend="ray",
            max_num_seqs=512,
            max_num_batched_tokens=13192,
            seed=i,
            gpu_memory_utilization=0.95
        )
        infer_llms.append(llm)
    return infer_llms


def get_reward(questions, answers, smps):
    train_sample_groups = []
    summay_of_rewards = []
    for q, q_answers, smp in tqdm(zip(questions, answers, smps)):
        data_source = smp['data_source']
        ground_truth = smp['reward_model']['ground_truth']
        rewards = [_default_compute_score(data_source, q_answer, ground_truth, {}) for q_answer in q_answers]
        rewards = [1 if _['score'] > 0 else 0 for _ in rewards]
        rewards = torch.tensor(rewards)
        summay_of_rewards.append(rewards)
        for ans, reward in zip(q_answers, rewards):
            train_sample_groups.append([q, ans, reward])
    return train_sample_groups, summay_of_rewards


def run_eval(eval_dataset, infer_llms, dump_file=None, rollout_num=32):
    num_infer_workers = len(infer_llms)
    data_source_rewards = defaultdict(list)
    if dump_file is not None:
        of = open(dump_file, 'w')
    smps = [eval_dataset[idx] for idx in range(len(eval_dataset))]
    questions = [eval_dataset[idx]['prompt'][0]['content'] for idx in range(len(eval_dataset))]
    assert int(rollout_num / num_infer_workers) >= 1  # 如果很多机器需实现对questions按机器切分, 评测暂不刚需；
    answers = ray.get([llm.do_generate.remote(questions,
                                              n=int(rollout_num / num_infer_workers),
                                              eval_infer=False,
                                              max_tokens=3072,
                                              temperature=1.0,
                                              top_p=1.0,
                                              ) for llm in infer_llms])
    answers = [sum(groups, []) for groups in zip(*answers)]
    train_sample_groups, summay_of_rewards = get_reward(questions, answers, smps)
    for question, q_ans, q_reward, smp in zip(questions, answers, summay_of_rewards, smps):
        smp_info = {"question": question,
                    'answers': q_ans,
                    'rewards': q_reward.tolist(),
                    'data_source': smp['data_source'],
                    'ground_truth': smp['reward_model']['ground_truth']}
        if dump_file is not None:
            ln = json.dumps(smp_info, ensure_ascii=True) + '\n'
            of.write(ln)
        data_source_rewards[smp['data_source']].extend(q_reward.tolist())

    if dump_file is not None:
        of.flush()
    for data_source, rewards in data_source_rewards.items():
        print(rewards)
        average_reward = sum(rewards) / len(rewards)
        red_print(f'{data_source} Eval Acc: {average_reward}')


def deduplicate(eval_set):
    seen = set()
    uniq_examples = []
    for ex in eval_set:
        key = ex['prompt'][0]['content']
        if key not in seen:  # 第一次见到这个取值 → 保留
            seen.add(key)
            uniq_examples.append(ex)
    dedup_set = Dataset.from_list(uniq_examples)
    return dedup_set


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path")
    parser.add_argument("--test_parquet", default='RLEP_dataset/dapo_format_aime2024_aime2025_amc2023.parquet')
    parser.add_argument("--eval_log_file", default=None)

    import ray

    ray.init()
    nodes = ray.nodes()
    args = parser.parse_args()
    model_path = args.model_path
    TEST_FILE = args.test_parquet
    eval_log_file = args.eval_log_file

    infer_llms = vllm_init(model_path, num_infer_workers=8)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    from datasets import Dataset

    eval_set = Dataset.from_parquet(TEST_FILE)
    eval_set = deduplicate(eval_set)
    run_eval(eval_set, infer_llms, dump_file=eval_log_file)
