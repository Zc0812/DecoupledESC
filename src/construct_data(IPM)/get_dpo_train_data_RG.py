### generate preference datas for Response Generator
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
    openai_api_key = args.api_key_RG
    openai_api_base = args.api_base_RG
    model_RG = args.model_name_RG
    client = OpenAI(base_url=openai_api_base, api_key=openai_api_key)
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    output_datas = []
    for cnt_data, item in enumerate(data):
        RG_conversations = []
        for message in item["messages"]:
            if message["role"] == "user":
                RG_conversations.append({"role": "user", "content": message["content"]})
            elif message["role"] == "assistant":
                content = remove_first_bracket(message['content'])
                strategy = extract_first_bracket(message['content'])
                system_content_RG = """You are a psychologist with a professional background in psychology and mental health, providing empathic responses to help seekers' psychological problems and completing psychological counseling.\nYour response should flow naturally, avoid being too long or too short, avoid using overly instructional language, and make sure your response fits the client's needs and context. You will be provided with a reference response strategy, and you are to generate reply content based on that strategy. For example, the format of response: <your reply content>. The strategies include:Question, Restatement or Paraphrasing, Reflection of feelings, Self-disclosure, Affirmation and Reassurance, Providing Suggestions, Information, Others. Your current reference response strategy is {reply_strategy}.\nThe conversation is as follows:\n"""
                system_content_RG = system_content_RG.format(reply_strategy=strategy)
                new_messages = []
                new_messages.append({"role": "system", "content": system_content_RG})
                new_messages.extend(RG_conversations)
                chat_response = client.chat.completions.create(
                    model=model_RG,
                    messages=new_messages,
                    temperature=0.7,
                    top_p=0.95
                )
                predict_content = chat_response.choices[0].message.content
                RG_data = {
                    "conversations": new_messages,
                    "chosen": {"role": "assistant", "content": content}, 
                    "rejected": {"role": "assistant", "content": predict_content}
                }
                output_datas.append(RG_data)
                RG_conversations.append({"role": "assistant", "content": content})
    with open(args.output_file, 'w') as f:
        json.dump(output_datas, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get DPO train data for Response Generator")
    parser.add_argument("--input_file", type=str, default='./data/Vanilla/ESC_sft_train.json', help='Path to the input file')
    parser.add_argument("--output_file", type=str, help='Path to the output file')
    parser.add_argument("--model_name_RG",type=str,help="Model name for Response Generator")
    parser.add_argument("--api_base_RG",type=str,help="API base URL for Response Generator")
    parser.add_argument("--api_key_RG",type=str,help="API key for Response Generator")
    args = parser.parse_args()
    main(args)
