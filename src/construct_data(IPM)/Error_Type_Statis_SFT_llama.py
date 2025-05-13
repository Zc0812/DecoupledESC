import os
import sys
import json
import argparse
from loguru import logger
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from llm import gpt_llm
from api_keys import openai_apikey_list
from file_utils import (
    load_txt, load_json,
    save_data, _norm,
)
from datetime import datetime
import re

random.seed(42)

strategy_pattern = re.compile(r'\[(.*?)\]', re.S)

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='data_path')
    parser.add_argument('--save_path', type=str, default='save_path')
    parser.add_argument("--model_name", type=str, default="model_name")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--data_num", type=int)
    parser.add_argument("--max_workers", type=int, default=16)

    parser.add_argument('--sys_prompt_path', type=str)
    parser.add_argument('--prompt_path', type=str, default='prompt_path')
    return parser.parse_args()


def process_single_dialog(dialogs, prompt, sys_prompt, args, api_key):
    results = []
    context = ""
    user_input = ""
    for i, dialog in enumerate(dialogs["conversations"]):
        if i < len(dialogs["conversations"]) - 1:
            content = _norm(dialog["content"])
            if i == 0:
                context += f"system: {content}\n"
            elif i % 2:
                context += f"user: {content}\n"
            elif i // 2:
                context += f"assistant: {content}\n"
        elif i == len(dialogs["conversations"]) - 1:
            user_input = _norm(dialog["content"])

    response_text = _norm(dialogs["predict_content"])
    strategy = dialogs["predict_strategy"]

    prompt_back = prompt.format(
        context=context,
        user_input=user_input,
        strategy=strategy,
        response=response_text
    )

    logger.info(f"User Input: {user_input}\n")

    logger.info(f"Model Prompt Input: {prompt_back}\n")

    try:
        model_response = gpt_llm(
            model_name=args.model_name,
            sys_prompt=sys_prompt,
            input=prompt_back,
            temperature=args.temperature,
            api_key=api_key
        )
        logger.info(f"Model Output: {model_response}\n")

        result_json = json.loads(model_response)
        result_json["user_input"] = user_input
        result_json["strategy"] = strategy
        result_json["response"] = response_text
        results.append(result_json)

    except Exception as e:
        logger.error(f"Error in processing dialog: {e}")

    return results

def Error_Type_Statis_Multithread(args):
    sys_prompt = load_txt(args.sys_prompt_path)
    prompt = load_txt(args.prompt_path)
    all_data = load_json(args.data_path)
    selected_data = all_data[:args.data_num]

    logger.info(f"Loaded {len(selected_data)} dialogs, starting multithreaded processing...")

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for i, data in enumerate(selected_data):
            api_key = openai_apikey_list[i % len(openai_apikey_list)]
            futures.append(executor.submit(
                process_single_dialog,
                data,
                prompt,
                sys_prompt,
                args,
                api_key
            ))

        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                res = future.result()
                results.extend(res)
            except Exception as e:
                logger.error(f"Error in future: {e}")

    logger.info(f"Finished processing. Saving results to {args.save_path}...")
    save_data(args.save_path, results)
    logger.success("All done!")

if __name__ == "__main__":
    args = parse_arguments()
    Error_Type_Statis_Multithread(args)
