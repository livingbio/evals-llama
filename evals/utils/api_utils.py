"""
This file defines various helper functions for interacting with the OpenAI API.
"""
import logging
from typing import Any

import backoff
import openai

from evals.prompt.base import OpenAIChatMessage, OpenAICreateChatPrompt
from . import llama

def chat_completion(model: str,  **kwargs) -> dict[str, Any]:
    if model == "llama":
        return llama.llama_chat_completion_create_retrying(model=model, **kwargs)
    else:
        return openai_chat_completion_create_retrying(model=model, **kwargs)

def completion(model: str, **kwargs):
    if model == "llama":
        return llama.llama_completion_create_retrying(model=model, **kwargs)
    else:
        return openai_completion_create_retrying(model=model, **kwargs)


@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        openai.error.ServiceUnavailableError,
        openai.error.APIError,
        openai.error.RateLimitError,
        openai.error.APIConnectionError,
        openai.error.Timeout,
    ),
    max_value=60,
    factor=1.5,
)
def openai_completion_create_retrying(*args, **kwargs):
    """
    Helper function for creating a completion.
    `args` and `kwargs` match what is accepted by `openai.Completion.create`.
    """

    result = openai.Completion.create(*args, **kwargs)
    if "error" in result:
        logging.warning(result)
        raise openai.error.APIError(result["error"])
    return result


@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        openai.error.ServiceUnavailableError,
        openai.error.APIError,
        openai.error.RateLimitError,
        openai.error.APIConnectionError,
        openai.error.Timeout,
    ),
    max_value=60,
    factor=1.5,
)
def openai_chat_completion_create_retrying(*args, **kwargs):
    """
    Helper function for creating a chat completion.
    `args` and `kwargs` match what is accepted by `openai.ChatCompletion.create`.
    """
    result = openai.ChatCompletion.create(*args, **kwargs)
    if "error" in result:
        logging.warning(result)
        raise openai.error.APIError(result["error"])
    return result
