from typing import Optional

import weave
import call_llm
import helpers
import json
import asyncio
import handle_tool_calls as htc
from weave.flow.scorer import Scorer

weave.init("abe_timeentry_agent")

@weave.op()
def generate_evaluation_data():
    questions = [
        {"messages": [helpers.build_message("user","Log 8 hours of work for project chameleon and entry code 23523 for today")]},
        {"messages": [helpers.build_message("user", "Log 5 hours to project nimbus with code 44 for 10/23/2022")]},
        {"messages": [helpers.build_message("user","What times are associated with project nimbus?")]},
        {"messages": [helpers.build_message("user","What times are associated with code 23523?")]}
    ]


    prompts = [
        "Log 8 hours of work for project chameleon and entry code 23523 for today",
        "Log 5 hours to project nimbus with code 44 for 10/23/2022",
        "What times are associated with project nimbus?",
        "What times are associated with code 23523?"
    ]

    answers = [
        "I have logged 8 hours to project 'chameleon' with code 23523 for today, October 4, 2023.",
        "I have logged 5 hours to project 'nimbus' with code 44 for October 23, 2022.",
        "The times associated with project 'nimbus' are as follows: - On October 23, 2022, there are 5 hours logged with code 44.",
        "The times associated with code 44 are as follows: - On October 4, 2023, there are 5 hours logged for project 'chameleon.'"
    ]

    int_tests = [
        {'id': '0', 'sentence': prompts[0], 'target': answers[0]},
        {'id': '1', 'sentence': prompts[1], 'target': answers[1]},
        {'id': '2', 'sentence': prompts[2], 'target': answers[2]},
        {'id': '3', 'sentence': prompts[3], 'target': answers[3]}
    ]

    return questions

# Define a scorer
@weave.op()
def success_in_timekeeping_scorer(messages, model_output):
    context_precision_prompt = f'''Given the prompt and answer verify if the assistant 
    successfully and correctly logged or retrieved data from a time keeping system.
    Give the success field a value between 0 and 1, inclusive. Where 1 means the assistant was completely successful,
    and 0 means the assistant was completely unsuccessful
    Answer only in valid JSON format.

    prompt: {messages[0]["content"]}
    answer: {model_output.choices[0].message.content}'''

    messages = [
        helpers.build_message("user", context_precision_prompt),
    ]

    evaluator_model = call_llm.TimeSystemHelperModel(llm_type="openai")

    response = evaluator_model.predict(messages)
    print(response.choices[0].message.content)
    response = json.loads(response.choices[0].message.content)
    return {
        "success": int(response["success"]),
    }

@weave.op()
def evaluate():
    model = call_llm.TimeSystemHelperModel(llm_type="openai")
    evaluation = weave.Evaluation(
        dataset=generate_evaluation_data(),
        scorers=[
            success_in_timekeeping_scorer
        ],
    )
    asyncio.run(evaluation.evaluate(model))

evaluate()
