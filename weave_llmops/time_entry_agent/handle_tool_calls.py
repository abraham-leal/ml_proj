import json
import datetime
import time_entry_system as tes
def handle(tool_calls, system, messages):
    # If true the model will return the name of the tool / function to call and the argument(s)
    tool_call_id = tool_calls[0].id
    tool_function_name = tool_calls[0].function.name
    tool_entry = json.loads(tool_calls[0].function.arguments)
    print(f"Processing returned LLM input in tools: {tool_entry}")
    times = {}

    # Step 3: Call the function and retrieve results. Append the results to the messages list.
    if tool_function_name == 'record_time':
        try:
            entry_json = json.loads(tool_entry['entry'])
            this_entry_date = entry_json["date"]
            this_entry_project = entry_json["project"]
            this_entry_code = entry_json["code"]
            this_entry_hours = entry_json["hours"]
            this_date = datetime.date(year=this_entry_date["year"], month=this_entry_date["month"], day=this_entry_date["day"])
            record_time = tes.TimeEntry(this_date, this_entry_project, this_entry_code, this_entry_hours)
            times = system.record_time(record_time)
            print(f"Recorded time in system for code: {times}")
        except Exception as e:
            print("Unable to record time given LLM directions")
            print(f"JSON Directions: {tool_entry}")
            print(f"Exception: {e}")
    elif tool_function_name == 'get_times_for_date':
        entry_json = json.loads(tool_entry['date'])
        this_entry_year = entry_json["year"]
        this_entry_month= entry_json["month"]
        this_entry_day = entry_json["day"]
        this_date = datetime.date(year=this_entry_year, month=this_entry_month, day=this_entry_day)
        times = system.get_times_for_date(this_date)
        print(f"Times in system associated with date {this_date}: {times}")
    elif tool_function_name == 'get_times_for_project':
        entry_json = json.loads(tool_entry['project'])
        this_project = entry_json["project"]
        times = system.get_times_for_project(this_project)
        print(f"Times in system associated with project {this_project}: {times}")
    elif tool_function_name == 'get_times_for_code':
        entry_json = json.loads(tool_entry['code'])
        this_code = entry_json["code"]
        times = system.get_times_for_code(this_code)
        print(f"Times in system associated with code {this_code}: {times}")
    else:
        print("Error: The tool returned is not recognizable")
        print(tool_entry)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_function_name,
            "content": times
        })

    return messages
