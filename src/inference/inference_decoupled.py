#inference for Strategy Planner and Response Generator
from openai import OpenAI
import re
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
    # Initialize OpenAI API client
    openai_api_key_SP = args.api_key_SP
    openai_api_base_SP = args.api_base_SP
    openai_api_key_RG = args.api_key_RG
    openai_api_base_RG = args.api_base_RG
    model_SP = args.model_name_SP
    model_RG = args.model_name_RG
    client_SP = OpenAI(base_url=openai_api_base_SP, api_key=openai_api_key_SP)
    client_RG = OpenAI(base_url=openai_api_base_RG, api_key=openai_api_key_RG)
    system_content_SP = """You are an emotional strategy identifier. Based on the seeker's questions, as well as the conversation between the seeker and the supporter, please provide the supporter with reference emotional strategies.\nThe strategies include:Question, Restatement or Paraphrasing, Reflection of feelings, Self-disclosure, Affirmation and Reassurance, Providing Suggestions, Information, Others. Specially, your response should consist of only one part: the reply strategy.\nThe conversation is as follows:\n"""
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    output_datas = []
    for cnt_data, item in enumerate(data):
        user_input = ""
        conversations = []
        for message in item["messages"]:
            if message["role"] == "user":
                conversations.append({"role": "user", "content": message["content"]})
                user_input += "seeker: " + message["content"] + "\n"
            elif message["role"] == "assistant":
                GT_content = remove_first_bracket(message['content'])
                GT_strategy = extract_first_bracket(message["content"])
                SP_messages = [
                    {"role": "system", "content": system_content_SP},
                    {"role": "user", "content": user_input}
                ]
                chat_response_SP = client_SP.chat.completions.create(
                    model=model_SP,
                    messages=SP_messages,
                    temperature=0.7,
                    top_p=0.95
                )
                predict_strategy = chat_response_SP.choices[0].message.content
                system_content_RG = """You are a psychologist with a professional background in psychology and mental health, providing empathic responses to help seekers' psychological problems and completing psychological counseling.\nYour response should flow naturally, avoid being too long or too short, avoid using overly instructional language, and make sure your response fits the client's needs and context. You will be provided with a reference response strategy, and you are to generate reply content based on that strategy. For example, the format of response: <your reply content>. The strategies include:Question, Restatement or Paraphrasing, Reflection of feelings, Self-disclosure, Affirmation and Reassurance, Providing Suggestions, Information, Others. Your current reference response strategy is {reply_strategy}.\nThe conversation is as follows:\n"""
                system_content_RG = system_content_RG.format(reply_strategy=predict_strategy)
                RG_messages = []
                RG_messages.append({"role": "system", "content": system_content_RG})
                RG_messages.extend(conversations)
                chat_response_RG = client_RG.chat.completions.create(
                    model=model_RG,
                    messages=RG_messages,
                    temperature=0.7,
                    top_p=0.95
                )
                predict_content = chat_response_RG.choices[0].message.content
                cur_conversations = conversations[:]
                output_data = {
                    "conversations": cur_conversations,
                    "GT_strategy": GT_strategy, 
                    "predict_strategy": predict_strategy,
                    "GT_content": GT_content,
                    "predict_content": predict_content
                }
                output_datas.append(output_data)
                content = remove_first_bracket(message['content'])
                user_input += "supporter: " + GT_content + "\n"
                conversations.append({"role": "assistant", "content": content})
    with open(args.output_file, 'w') as f:
        json.dump(output_datas, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for Decoupled paradigm")
    parser.add_argument("--input_file", type=str, default='./data/Vanilla/ESC_sft_test.json', help='Path to the input file')
    parser.add_argument("--output_file", type=str, help='Path to the output file')
    parser.add_argument("--model_name_SP",type=str,help="Model name for Strategy Planner")
    parser.add_argument("--api_base_SP",type=str,help="API base URL for Strategy Planner")
    parser.add_argument("--api_key_SP",type=str,help="API key for Strategy Planner")
    parser.add_argument("--model_name_RG",type=str,help="Model name for Response Generator")
    parser.add_argument("--api_base_RG",type=str,help="API base URL for Response Generator")
    parser.add_argument("--api_key_RG",type=str,help="API key for Response Generator")
    args = parser.parse_args()
    main(args)


