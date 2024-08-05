from termcolor import colored
import handle_tool_calls as htc
import helpers
import generator
import datetime
import weave
import json
import time_entry_system as tes
import call_llm
def main():
    client = weave.init("abe_timeentry_agent")

    #generator.generate_evaluation_data(client)

    system = helpers.start_timekeeping_system()

    context = """
            You are a helpful time-keeping assistant. You are able to call functions to log and retrieve a dictionary
            of dates.
            Don't make assumptions about what values to plug into functions.
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
        ## Print contents of messages dict
        print(messages)
        ## Call LLM
        completion = call_llm.call_openai(messages)
        ## Extract answer
        response_message = completion.choices[0].message
        ## Provide previous answer back to the LLM for context
        messages.append(response_message)
        ## Check if tools were called
        tool_calls = response_message.tool_calls
        if tool_calls:
            ## handle tool invocation and return a messages chain with results
            messages = htc.handle(tool_calls, system, messages)
        else:
            print(response_message.content)


if __name__ == '__main__':
    main()
