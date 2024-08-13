import os
import openai
import weave
from mistralai.client import MistralClient
from openai import OpenAI
import mistralai
from dotenv import load_dotenv
import handle_tool_calls as htc
import time_entry_system as tes

load_dotenv()

@weave.op()
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
    system: tes.TimeEntrySystem
    system = tes.TimeEntrySystem()

    @weave.op()
    def predict(self, messages: []):
        if self.llm_type == "openai":
            return self.call_openai(messages)
        elif self.llm_type == "llama":
            return self.call_llama(messages)
        elif self.llm_type == "mistral":
            return self.call_mistral(messages)
        else:
            print("Invalid llm type")

    @weave.op()
    def handle_tools(self, messages, response_message):
        ## Check if tools were called
        tool_calls = response_message.choices[0].message.tool_calls
        if tool_calls:
            ## handle tool invocation and return a messages chain with results
            htc.handle(tool_calls, self.system, messages)
            ## Call LLM to present result
            completion = self.predict(messages)
            messages.append(completion.choices[0].message)
            return completion
        else:
            return response_message

    ## Calls OpenRouter, currently set to use llama3.1 - 8B as it is free
    @weave.op()
    def _call_openrouter(self, model, messages) -> str:
        or_client = self.get_oai_llm_client("https://openrouter.ai/api/v1", os.getenv("OR_API_KEY"))

        try:
            completion = or_client.chat.completions.create(
                model=model,
                messages=messages,
                tools=get_tools(),
                temperature=0.0,
            )
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            return e

        messages.append(completion.choices[0].message)
        completion = self.handle_tools(messages, completion)
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

        messages.append(completion.choices[0].message)
        completion = self.handle_tools(messages, completion)
        return completion

    @weave.op()
    def call_mistral(self, messages):
        mistral_client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])

        try:
            completion = mistral_client.chat(
                model="mistral-large-latest",
                messages=messages,
                tools=get_tools()
            )
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            return e

        messages.append(completion.choices[0].message)
        completion = self.handle_tools(messages, completion)
        return completion

    ## Calls Llama's 3.1 8B Instruct
    @weave.op()
    def call_llama(self, messages):
        return self._call_openrouter("meta-llama/llama-3.1-8b-instruct:free", messages)

    @weave.op()
    def get_oai_llm_client(self, url: str, apikey: str) -> openai.Client:
        if url == "":
            client = OpenAI(api_key=apikey)
        else:
            client = OpenAI(
                base_url=url,
                api_key=apikey)
        return client
