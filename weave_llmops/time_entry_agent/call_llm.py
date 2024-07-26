import os
import openai
import weave
from openai import OpenAI
import anthropic
from dotenv import load_dotenv

load_dotenv()

## Calls OpenRouter, currently set to use llama3.1 - 8B as it is free
@weave.op()
def call_openrouter(context: str, query: str) -> str:
    or_client = get_oai_llm_client(url="", apikey=os.getenv("OPENAI_API_KEY"))

    completion = or_client.chat.completions.create(
        model="meta-llama/llama-3.1-8b-instruct:free",
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": query}
        ],
        response_format={"type": "text"},
        temperature=0.0,
    )

    return completion.choices[0].message.content

## Calls OpenAI's openai-4o
@weave.op()
def call_openai(context: str, query: str):
    oai_client = get_oai_llm_client("https://openrouter.ai/api/v1", os.getenv("OR_API_KEY"))

    completion = oai_client.chat.completions.create(
        model="openai-4o",
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": query}
        ],
        response_format={"type": "text"},
        temperature=0.0,
    )

    return completion.choices[0].message.content


## Calls Anthropic's claude-3-5-sonnet-20240620
@weave.op()
def call_anthropic(context: str, query: str) -> str:
    ant_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    message = ant_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        system=context,
        temperature=0.0,
        messages=[
            {"role": "user", "content": query}
        ]
    )
    return message.content[0].text



def get_oai_llm_client(url: str, apikey: str) -> openai.Client:
    if url == "":
        client = OpenAI(api_key=apikey)
    else:
        client = OpenAI(
            base_url=url,
            api_key=apikey)
    return client
