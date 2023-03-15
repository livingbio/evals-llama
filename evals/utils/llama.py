from typing import Any
import requests
import os
from evals.prompt.base import OpenAICreateChatPrompt, OpenAICreatePrompt

def llama_chat_completion_create_retrying(
    model: str,
    api_key: str,
    messages: OpenAICreateChatPrompt,
    **kwargs
) -> dict[str, Any]:
    # FIXME: need change the prompt

    r = requests.post(f"{os.environ['LLAMA_SERVER']}/prompt", json={"prompts": ["\n".join(str(k) for k in messages)]})
    result = r.json()
    print(result)

    return {
        "choices": [{"message": {"content": k}} for k in result["results"]]
    }

def llama_completion_create_retrying(
    model: str,
    api_key: str,
    prompt: OpenAICreatePrompt,
    **kwargs
) -> dict[str, Any]:
    r = requests.post(f"{os.environ['LLAMA_SERVER']}/prompt", json={"prompts": [prompt]})
    result = r.json()
    print(result)

    return {
        "choices": [{"text": "\n".join(result["results"])}]
    }