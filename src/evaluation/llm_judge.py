import os
import sys


import re
from llm import gpt_llm
from api_keys import gpt_apikey_list, openai_apikey_list
import json
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from file_utils import load_txt, load_json, save_data

apikey_list = openai_apikey_list

metrics = [
    "Fluency",
    "Professionalism",
    "Empathy",
    "Helpfulness",
]

class LLM_Judge(object):
    def __init__(self, model_name, prompt_dir, Context, User_input, GT, pred):
        self.model_name = model_name
        self.prompt_dir = prompt_dir
        self.Context = Context
        self.User_input = User_input
        self.GT = GT
        self.pred = pred
        
        prompts = []
        for metric in metrics:
            data_path = os.path.join(self.prompt_dir, metric+".txt")
            prompt = load_txt(data_path)
            prompt = prompt.format(Context=self.Context, User_input=self.User_input, GT_Res=self.GT, Pred_Res=self.pred)
            prompts.append({"metric":metric, "prompt":prompt})
        self.prompts = prompts

    def _evaluate(self, prompt, api_key):
        response = gpt_llm(model_name=self.model_name, sys_prompt=None, input=prompt, temperature=0.0, api_key=api_key)
        print("-"*100)
        print(f"{response}")
        print("-"*100)
        match = re.fullmatch(r"\s*([0-5](?:\.\d+)?)\s*", response)
        if match:
            score = float(match.group(1))
            return score
        else:
            raise ValueError(f"Invalid GPT response: expected a number between 0-5, got '{response}'")
        

    def evaluate(self):
        scores = {}
        num_keys = len(apikey_list)
        
        with ThreadPoolExecutor(max_workers=num_keys) as executor:
            futures = {
                executor.submit(self._evaluate, prompt_data["prompt"], apikey_list[i % num_keys]): prompt_data["metric"]
                                for i, prompt_data in enumerate(self.prompts)
            }
            for future in futures:
                metric = futures[future]
                try:
                    score = future.result()
                    scores[metric] = score
                except Exception as e:
                    logger.error(f"Error evaluating {metric}: {e}")
                    scores[metric] = None
        return scores