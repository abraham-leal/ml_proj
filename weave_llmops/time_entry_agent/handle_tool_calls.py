import json
import datetime
import weave
import time_entry_system as tes

@weave.op()
def handle(tool_calls, system, messages):
    # If true the model will return the name of the tool / function to call and the argument(s)
    tool_call_id = tool_calls[0].id
    tool_function_name = tool_calls[0].function.name
    tool_entry = json.loads(tool_calls[0].function.arguments)
    times = {}

    # Step 3: Call the function and retrieve results. Append the results to the messages list.
    if tool_function_name == 'record_time':
        try:
            entry_json = json.loads(tool_entry['entry'])
            this_entry_date = entry_json["date"]
            this_entry_project = entry_json["project"]
            this_entry_code = entry_json["code"]
            this_entry_hours = entry_json["hours"]
            this_date = datetime.date(year=int(this_entry_date["year"]), month=int(this_entry_date["month"]), day=int(this_entry_date["day"]))
            record_time = tes.TimeEntry(this_date.strftime("%m/%d/%Y"), this_entry_project, int(this_entry_code), int(this_entry_hours))
            times.update(system.record_time(record_time))
        except Exception as e:
            print("Unable to record time given LLM directions")
            print(f"JSON Directions: {tool_entry}")
            print(f"Exception: {e}")
    elif tool_function_name == 'get_times_for_date':
        try:
            entry_json = json.loads(tool_entry['date'])
            this_entry_year = entry_json["year"]
            this_entry_month= entry_json["month"]
            this_entry_day = entry_json["day"]
            this_date = datetime.date(year=int(this_entry_year), month=int(this_entry_month), day=int(this_entry_day))
            times.update(system.get_times_for_date(this_date.strftime("%m/%d/%Y")))
        except Exception as e:
            print("Unable to get times given LLM directions")
            print(f"JSON Directions: {tool_entry}")
            print(f"Exception: {e}")
    elif tool_function_name == 'get_times_for_project':
        try:
            this_project = tool_entry['project']
            times.update(system.get_times_for_project(this_project))
        except Exception as e:
            print("Unable to get times given LLM directions")
            print(f"JSON Directions: {tool_entry}")
            print(f"Exception: {e}")
    elif tool_function_name == 'get_times_for_code':
        try:
            this_code = tool_entry['code']
            times.update(system.get_times_for_code(int(this_code)))
        except Exception as e:
            print("Unable to get times given LLM directions")
            print(f"JSON Directions: {tool_entry}")
            print(f"Exception: {e}")
    else:
        print("Error: The tool returned is not recognizable")
        print(tool_entry)

    messages.append({
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_function_name,
        "content": json.dumps(times)
    })

    return messages