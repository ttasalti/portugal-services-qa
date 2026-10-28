import os
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from ptqa.config import ANSWER_TOKENS


def llm_settings():
    load_dotenv()
    extra = None
    if os.environ.get("LLM_DISABLE_THINKING", "1") == "1":
        extra = {"chat_template_kwargs": {"enable_thinking": False}}
    return {
        "model": os.environ["LLM_MODEL"],
        "base_url": os.environ["LLM_BASE_URL"],
        "api_key": os.environ.get("LLM_API_KEY", "none"),
        "extra_body": extra,
    }


def make_chat_model(max_tokens=ANSWER_TOKENS):
    cfg = llm_settings()
    return ChatOpenAI(
        model=cfg["model"],
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        temperature=0.0,
        max_tokens=max_tokens,
        timeout=120,
        extra_body=cfg["extra_body"],
    )


def clean_answer(text):
    # Qwen can leave an empty think block even with thinking off
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def read_reply(message):
    text = getattr(message, "content", message)
    meta = getattr(message, "response_metadata", {})
    return clean_answer(text), meta.get("finish_reason") == "length"
