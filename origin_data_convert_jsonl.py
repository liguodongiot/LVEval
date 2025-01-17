import os
import re
import tiktoken
import argparse
from tqdm import tqdm
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

from config import (
    DATASET_MAXGEN, 
    DATASET_PROMPT, 
    DATASET_SELECTED, 
    DATASET_LENGTH_LEVEL,
)
from utils import (
    ensure_dir, 
    seed_everything,
    get_dataset_names,
    post_process,
    load_jsonl,
    load_LVEval_dataset,
    dump_preds_results_once,
)

def get_pred(
    url,
    tokenizer,
    data,
    max_length,
    max_gen,
    prompt_format,
    model_name,
    save_path,
    model_id,
):
    preds = []

    existed_questions = [data['input'] for data in load_jsonl(save_path)]

    for json_obj in tqdm(data):
        if json_obj['input'] in existed_questions:
            print(f'pred already exists in {save_path}, jump...')
            continue
        
        prompt = prompt_format.format(**json_obj)
        # following LongBench, we truncate to fit max_length
        tokenized_prompt = tokenizer.encode(prompt)
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenized_prompt[:half]) + tokenizer.decode(tokenized_prompt[-half:])

        item = {
            "text": prompt,
            "answers": json_obj["answers"],
            "gold_ans": json_obj["answer_keywords"] if "answer_keywords" in json_obj else None,
            "input": json_obj["input"],
            "all_classes": json_obj["all_classes"] if "all_classes" in json_obj else None,
            "length": json_obj["length"],
        }

        dump_preds_results_once(item, save_path)
        preds.append(item)
    return preds

def single_processing(datasets, args):
    model_id = args.model_name
    if 'qwen' in model_id:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    for dataset in tqdm(datasets):
        datas = load_LVEval_dataset(dataset, args.data_path)
        # dataset_size = 50
        # datas = datas[:dataset_size]
        dataset_name = re.split('_.{1,3}k', dataset)[0]
        prompt_format = DATASET_PROMPT[dataset_name]
        max_gen = DATASET_MAXGEN[dataset_name]
        save_path = os.path.join(args.output_dir, dataset + ".jsonl")
        get_pred(
            args.url,
            tokenizer,
            datas,
            args.model_max_length,
            max_gen,
            prompt_format,
            args.model_name,
            save_path,
            model_id,
        )
        
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://10.10.2.145:9009/v1/chat/completions", required=True)
    parser.add_argument("--model-name", type=str, default=None, required=True, choices=["qwen2", 
                                                                                        "qwen2.5"])
    parser.add_argument("--model-path", type=str, default="", required=True)
    parser.add_argument("--model-max-length", type=int, default=15500)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args(args)

if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    ensure_dir(args.output_dir)
    # datasets = get_dataset_names(DATASET_SELECTED, DATASET_LENGTH_LEVEL)
    # datasets = get_dataset_names(['cmrc_mixup'], ['16k'])
    datasets = get_dataset_names(['cmrc_mixup'], ['16k','32k','64k','128k'])
    single_processing(datasets, args)
  