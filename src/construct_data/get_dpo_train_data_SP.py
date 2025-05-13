### generate preference datas for Strategy Planner
import json
from openai import OpenAI
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
    openai_api_key = args.api_key_SP
    openai_api_base = args.api_base_SP
    model_SP = args.model_name_SP
    client = OpenAI(base_url=openai_api_base, api_key=openai_api_key)
    system_content_SP = """You are an emotional strategy identifier. Based on the seeker's questions, as well as the conversation between the seeker and the supporter, please provide the supporter with reference emotional strategies.\nThe strategies include:Question, Restatement or Paraphrasing, Reflection of feelings, Self-disclosure, Affirmation and Reassurance, Providing Suggestions, Information, Others. Specially, your response should consist of only one part: the reply strategy.\nThe conversation is as follows:\n"""
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    SP_datas_diff = []
    SP_datas_same = []
    for cnt_data, item in enumerate(data):
        user_input = ""
        for message in item["messages"]:
            if message["role"] == "assistant":
                GT_strategy = extract_first_bracket(message["content"])
                conversation = [
                    {"role": "system", "content": system_content_SP},
                    {"role": "user", "content": user_input}
                ]
                chat_response = client.chat.completions.create(
                    model=model_SP,
                    messages=conversation,
                    temperature=0.7,
                    top_p=0.95
                )
                predict_strategy = chat_response.choices[0].message.content
                SP_data = {
                    "conversations": conversation,
                    "chosen": {"role": "assistant", "content": GT_strategy},
                    "rejected": {"role": "assistant", "content": predict_strategy},
                }
                if GT_strategy != predict_strategy:
                    SP_datas_diff.append(SP_data)
                else:
                    SP_datas_same.append(SP_data)
                content = remove_first_bracket(message['content'])
                user_input += "supporter: " + content + "\n"
            if message["role"] == "user":
                user_input += "seeker: " + message["content"] + "\n"
    with open(args.output_file_DiffStrategy, 'w') as f:
        json.dump(SP_datas_diff, f, indent=2)
    with open(args.output_file_SameStrategy, 'w') as f:
        json.dump(SP_datas_same, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get DPO train data for Strategy Planner")
    parser.add_argument("--input_file", type=str, default='./data/Vanilla/ESC_sft_train.json', help='Path to the input file')
    parser.add_argument("--output_file_SameStrategy", type=str, help='Path to the output file. Save the same strategy data.')
    parser.add_argument("--output_file_DiffStrategy", type=str, help='Path to the output file. Save the different strategy data.')
    parser.add_argument("--model_name_SP",type=str,help="Model name for Strategy Planner")
    parser.add_argument("--api_base_SP",type=str,help="API base URL for Strategy Planner")
    parser.add_argument("--api_key_SP",type=str,help="API key for Strategy Planner")
    args = parser.parse_args()
    main(args)

        