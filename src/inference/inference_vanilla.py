#inference for Vanilla Model
from openai import OpenAI
import json
import argparse
def extract_first_bracket(text): #get the strategy
    start = text.find('[')
    end = text.find(']')
    if start != -1 and end != -1:
        return text[start + 1:end]
    return None

def remove_first_bracket(text): #get the response
    start = text.find('[')
    end = text.find(']')
    if start != -1 and end != -1:
        return text[end+1:].strip()
    return text

def main(args):
    openai_api_key = args.api_key_VM
    openai_api_base = args.api_base_VM
    model_VM = args.model_name_VM
    client = OpenAI(base_url=openai_api_base, api_key=openai_api_key)
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    output_datas = []
    for cnt_data, item in enumerate(data):
        mesaages = item["messages"]
        new_messages = []
        for message in mesaages:
            role = message["role"]
            content = message["content"]
            if role == "assistant":
                chat_response = client.chat.completions.create(
                    model=model_VM,
                    messages=new_messages,
                    temperature=0.7,
                    top_p=0.95
                )
                GT_content = remove_first_bracket(message['content'])
                GT_strategy = extract_first_bracket(message["content"])
                predict_strategy_content = chat_response.choices[0].message.content
                predict_strategy = extract_first_bracket(predict_strategy_content)
                predict_content = remove_first_bracket(predict_strategy_content)
                conversations = new_messages[:]
                output_data = {
                    "conversations": conversations,
                    "GT_strategy": GT_strategy,
                    "predict_strategy": predict_strategy,
                    "GT_content": GT_content,
                    "predict_content": predict_content
                }
                output_datas.append(output_data)
            new_messages.append(message)
    with open(args.output_file, 'w') as f:
        json.dump(output_datas, f, indent=2)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for Vanilla paradigm")
    parser.add_argument("--input_file", type=str, default='./data/ESC_sft_test.json', help='Path to the input file')
    parser.add_argument("--output_file", type=str, help='Path to the output file')
    parser.add_argument("--model_name_VM",type=str,help="Model name for Vanilla Model")
    parser.add_argument("--api_base_VM",type=str,help="API base URL for Vanilla Model")
    parser.add_argument("--api_key_VM",type=str,help="API key for Vanilla Model")
    args = parser.parse_args()
    main(args)
