### generate preference datas for Vanilla Model
from openai import OpenAI
import json
import argparse
def main(args):
    openai_api_key = args.api_key_VM
    openai_api_base = args.api_base_VM
    model_VM = args.model_VM
    client = OpenAI(base_url=openai_api_base, api_key=openai_api_key)
    with open(args.input_file, 'r') as file:
        data =  json.load(file)
    output_datas = []
    for cnt_item, item in enumerate(data):
        input_messages = item["messages"]
        new_messages = []
        for message in input_messages:
            role = message["role"]
            content = message["content"]
            if role == "assistant":
                chat_response = client.chat.completions.create(
                    model=model_VM,
                    messages=new_messages,
                    temperature=0.7,
                    top_p=0.95
                )
                output_data = {
                    "conversations": new_messages[:],
                    "chosen": message, 
                    "rejected": {"role": "assistant", "content": chat_response.choices[0].message.content}
                }
                output_datas.append(output_data)
            new_messages.append(message)
    with open(args.output_file, 'w') as f:
        json.dump(output_datas, f, indent=2)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get DPO train data for Vanilla Model")
    parser.add_argument("--input_file", type=str, default='./data/Vanilla/ESC_sft_train.json', help='Path to the input file')
    parser.add_argument("--output_file", type=str, help='Path to the output file')
    parser.add_argument("--model_name_VM",type=str,help="Model name for Vanilla Model")
    parser.add_argument("--api_base_VM",type=str,help="API base URL for Vanilla Model")
    parser.add_argument("--api_key_VM",type=str,help="API key for Vanilla Model")
    args = parser.parse_args()
    main(args)