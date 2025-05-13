import os
import sys

import argparse
from transformers import AutoTokenizer
from llm_judge import LLM_Judge
from metrics import Metric
from bias import evaluate_strategy
from file_utils import load_txt, load_json, save_data, _norm
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from datetime import datetime
import random
random.seed(42)

time = datetime.now()
time = time.strftime("%Y%m%d%H%M%S")
test_model_name = ""
gpt_model_name = "model_name"
print(test_model_name)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument(
        "--data_path", type=str, default="data_path"
    )
    parser.add_argument(
        "--BERT_tokenizer", type=str, default="BERT_tokenizer"
    )
    parser.add_argument(
        "--metric_path", type=str, default=f"data/results/metric/{test_model_name}.json"
    ) 
    parser.add_argument(
        "--model_name", type=str, default="gpt-4.1-mini-2025-04-14"
    )    
    parser.add_argument(
        "--prompt_dir", type=str, default="src/evaluation/eval_prompts"
    )   
    parser.add_argument(
        "--llm_judge_path", type=str, default=f"data/results/llm_judge/{test_model_name}.json"
    )  
    parser.add_argument(
        "--bias_path", type=str, default=f"data/results/bias/{test_model_name}.json"
    )  
    args = parser.parse_args()
    return args

def get_average_results(results, signal):
    count = 0
    if not results:
        return {}
    if signal == "metric":
        sum_results = {
            "length": 0.0,
            "dist-1": 0.0,
            "dist-2": 0.0,
            "dist-3": 0.0,
            "bleu-1": 0.0,
            "bleu-2": 0.0,
            "bleu-3": 0.0,
            "bleu-4": 0.0,
            "f1": 0.0,
            "rouge-l": 0.0,
        }
    elif signal == "llm_judge":
        sum_results = {
            "Fluency": 0.0,
            "Professionalism": 0.0,
            "Empathy": 0.0,
            "Helpfulness": 0.0,
        }
    for result in results[1:]:
        flag = True
        for key in sum_results.keys():
            if key not in result:
                flag = False
                break
            if result[key] is None:
                flag = False
                print(f"Skipping result with None value for {key}")
                break
        if flag:
            for key in sum_results.keys():
                sum_results[key] += result[key]
            count += 1
    avg_results = {key: value / count for key, value in sum_results.items()}
    
    return avg_results

############################################################################################################################################
##########################################################   1.evaluate_metric    ##########################################################
############################################################################################################################################
def eval_metric_single(data, toker):
    metric = Metric(toker)
    results = []
    metric.forward([data["GT_content"]], data["predict_content"])
    res, res_l = metric.close()
    results.append(res)
    return results

def eval_metric(datas, toker, metric_path):
    results = []
    results.append({"eavl_tokenizer": args.BERT_tokenizer, "model_name": test_model_name, "infer_data":args.data_path, "time": time})
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(eval_metric_single, data, toker) for data in datas]
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.extend(future.result())
    avg_results = get_average_results(results, "metric")
    results.append({'avg_results': avg_results})
    save_data(metric_path, results)

############################################################################################################################################
##########################################################      2.llm_judge       ##########################################################
############################################################################################################################################
def llm_eval(datas, model_name, prompt_dir, save_path):
    results = []
    results.append({"eavl_gpt": args.model_name, "model_name": test_model_name, "infer_data":args.data_path, "time": time})
    for data in datas:
        Context = ""
        User_input = ""
        for i, dialog in enumerate(data["conversations"]):
            if i < len(data["conversations"]) - 1:
                role = dialog["role"]
                content = _norm(dialog["content"])
                if role == "system":
                    Context += f"system: {content}\n"
                elif role == "user":
                    Context += f"seeker: {content}\n"
                elif role == "assistant":
                    Context += f"supporter: {content}\n"
            elif i == len(data["conversations"]) - 1:
                User_input = _norm(dialog["content"])
                
        GT = _norm(data["GT_content"])
        pred = _norm(data["predict_content"])
        llm_eva = LLM_Judge(model_name, prompt_dir, Context, User_input, GT, pred)
        results.append(llm_eva.evaluate())
    print(results)
    avg_results = get_average_results(results, "llm_judge")
    results.append({'avg_results': avg_results})
    logger.info(results[-1])
    save_data(save_path, results)

############################################################################################################################################
########################################################      3.Strategy Bias       ########################################################
############################################################################################################################################
def bias_eval(datas, save_path):
    evaluate_strategy(datas, save_path)

def main(args):
    datas = load_json(args.data_path)
    
    data = datas
    toker = AutoTokenizer.from_pretrained(args.BERT_tokenizer)
    if os.path.exists(args.metric_path):
        logger.info(f"{os.path.abspath(args.metric_path)} is exist，Skip saving!")
    else:
        logger.info(f"Start evaluating metric...")
        eval_metric(data, toker, args.metric_path)

    if os.path.exists(args.bias_path):
        logger.info(f"{os.path.abspath(args.bias_path)} is exist，Skip saving!")
    else:
        logger.info(f"Start evaluating bias...")
        bias_eval(data, args.bias_path)
    
    if os.path.exists(args.llm_judge_path):
        logger.info(f"{os.path.abspath(args.llm_judge_path)} is exist，Skip saving!")
    else:
        logger.info(f"Start evaluating llm_judge with {args.model_name}...")
        llm_eval(data, args.model_name, args.prompt_dir, args.llm_judge_path)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)