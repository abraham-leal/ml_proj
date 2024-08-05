import os
import openai
import weave
from openai import OpenAI
import anthropic
from dotenv import load_dotenv

load_dotenv()


def get_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "record_time",
                "description": "Add or subtract time in the time keeping system",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entry": {
                            "type": "string",
                            "description": """
                            A json describing a TimeEntry.
                            The TimeEntry should be returned in JSON with the following structure:
                            {"date": {"year": int,"month": int,"day": int}, "project": str, "code": int, "hours": int}
                            """,
                        }
                    },
                    "required": ["entry"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_times_for_date",
                "description": "Get all time entries associated with a date",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": """
                                A json describing a date.
                                The date should be represented in the following json format:
                                {"year": int,"month": int,"day": int}
                                """,
                        }
                    },
                    "required": ["date"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_times_for_project",
                "description": "Get all time entries associated with a project",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project": {
                            "type": "string",
                            "description": """
                                    A json describing a project.
                                    The date should be represented in the following json format:
                                    {"project": str}
                                    """,
                        }
                    },
                    "required": ["project"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_times_for_code",
                "description": "Get all time entries associated with a code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": """
                                        A json describing a code.
                                        The date should be represented in the following json format:
                                        {"code": int}
                                        """,
                        }
                    },
                    "required": ["code"],
                },
            },
        }
    ]

    return tools


class TimeSystemHelperModel(weave.Model):
    llm_type: str

    def predict(self, messages: []):
        if self.llm_type == "openai":
            return self.call_openai(messages)
        elif self.llm_type == "or":
            return self.call_openrouter(messages)
        elif self.llm_type == "ant":
            return self.call_anthropic(messages)
        else:
            print("Invalid llm type")

    ## Calls OpenRouter, currently set to use llama3.1 - 8B as it is free
    @weave.op()
    def call_openrouter(self, messages) -> str:
        or_client = self.get_oai_llm_client("https://openrouter.ai/api/v1", os.getenv("OR_API_KEY"))

        try:
            completion = or_client.chat.completions.create(
                model="meta-llama/llama-3.1-8b-instruct:free",
                messages=messages,
                response_format={"type": "text"},
                temperature=0.0,
            )
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            return e
        return completion

    ## Calls OpenAI's openai-4o
    @weave.op()
    def call_openai(self, messages):
        oai_client = self.get_oai_llm_client(url="", apikey=os.getenv("OPENAI_API_KEY"))

        try:
            completion = oai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=get_tools(),
                temperature=0.0,
            )
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            return e
        return completion

    ## Calls Anthropic's claude-3-5-sonnet-20240620
    @weave.op()
    def call_anthropic(self, messages) -> str:
        ant_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        try:
            message = ant_client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1024,
                temperature=0.0,
                messages=messages
            )
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            return e
        return message

    def get_oai_llm_client(self, url: str, apikey: str) -> openai.Client:
        if url == "":
            client = OpenAI(api_key=apikey)
        else:
            client = OpenAI(
                base_url=url,
                api_key=apikey)
        return client
