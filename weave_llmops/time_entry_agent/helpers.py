from termcolor import colored
import time_entry_system as tes
import datetime

## Returns a time entry system with 1 dummy entry
def start_timekeeping_system() -> tes.TimeEntrySystem:
    thisSystem = tes.TimeEntrySystem()
    dummyDate = datetime.date(year=2024, month=5, day=1)
    firstTime = tes.TimeEntry(dummyDate, "first", 11, 8)
    thisSystem.record_time(firstTime)

    return thisSystem


## Prints the given conversation in a pretty way
def pretty_print_conversation(messages):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }

    for message in messages:
        if message["role"] == "system":
            print(colored(f"system: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "user":
            print(colored(f"user: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and message.get("function_call"):
            print(colored(f"assistant: {message['function_call']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and not message.get("function_call"):
            print(colored(f"assistant: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "function":
            print(colored(f"function ({message['name']}): {message['content']}\n", role_to_color[message["role"]]))

def build_message(type: str, message):
    return {"role": f'{type}', "content": f'{message}'}