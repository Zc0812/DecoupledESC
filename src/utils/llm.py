import os
from openai import OpenAI
import anthropic
 
proxy_url = "proxy_url"
base_url = "base_url"

def set_proxy(url=proxy_url):
    if url:
        os.environ["http_proxy"] = url
        os.environ["https_proxy"] = url
        os.environ["HTTP_PROXY"] = url
        os.environ["HTTPS_PROXY"] = url       


def claude_llm(model_name, sys_prompt=None, input=None, temperature=0.7, api_key=None):
    base_url = base_url_closeai
        
    client = anthropic.Anthropic(
        api_key=api_key,
        base_url=base_url)

    if sys_prompt:
        sys_prompt_content = sys_prompt
    else:
        sys_prompt_content = "You are a helpful assistant." 

    # set_proxy(proxy_url)
    response = client.messages.create(
        max_tokens=4096,
        model=model_name,
        messages=[
            {"role": "system", "content": sys_prompt_content},
            {"role": "user", "content": input}],
        temperature=temperature
    )
    return response.content