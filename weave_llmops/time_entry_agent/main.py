from termcolor import colored
import handle_tool_calls as htc
import helpers
import datetime
import weave
import json
import time_entry_system as tes
import call_llm

@weave.op()
def main():
    client = weave.init("abe_timeentry_agent")
    model = call_llm.TimeSystemHelperModel(llm_type="openai")

    context = """
            You are a helpful time-keeping assistant. You are able to call functions to log and retrieve a dictionary
            of dates.
            Ask for clarification if a user request is ambiguous.
            """
    messages = [
                helpers.build_message("system", context),
            ]

    while True:
        ## Ask the user if there is something else they are thinking
        next_prompt = input('What can I help you with? Reply "stop" to stop.\n')
        if next_prompt.lower() == "stop":
            return
        messages.append(helpers.build_message("user", next_prompt))
        ## Call LLM and handle tool calling
        completion = model.predict(messages)
        print(completion.choices[0].message.content)
        ## Provide previous answer back to the LLM for context
        messages.append(completion.choices[0].message)





if __name__ == '__main__':
    main()
