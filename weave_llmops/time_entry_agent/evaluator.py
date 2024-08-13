from typing import Optional

import weave
import call_llm
import helpers
import json
import asyncio
import handle_tool_calls as htc
from weave.flow.scorer import Scorer

weave.init("abe_timeentry_agent")



def generate_evaluation_data():
    questions = [
        {"messages": [helpers.build_message("user", "Log 8 hours of work for project chameleon and entry code 23523 for today")]},
        {"messages": [helpers.build_message("user", "Log 5 hours to project nimbus with code 44 for 10/23/2022")]},
        {"messages": [helpers.build_message("user", "What times are associated with project nimbus?")]},
        {"messages": [helpers.build_message("user", "What times are associated with code 23523?")]}
    ]

    return questions


# Define a scorer
@weave.op()
def success_in_timekeeping_scorer(messages, model_output):
    context_precision_prompt = f'''Given the prompt and answer verify if the assistant 
    successfully and correctly logged or retrieved data from a time keeping system.
    Give the success field a value between 0 and 1, inclusive. Where 1 means the assistant was completely successful,
    and 0 means the assistant was completely unsuccessful
    Answer only in valid JSON format with one field named "success" and no decorators around the json struct.

    prompt: {messages[0]["content"]}
    answer: {model_output.choices[0].message.content}'''

    messages = [
        helpers.build_message("system", context_precision_prompt),
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
    model_oai = call_llm.TimeSystemHelperModel(llm_type="openai")
    model_llama = call_llm.TimeSystemHelperModel(llm_type="llama")
    model_mistral = call_llm.TimeSystemHelperModel(llm_type="mistral")
    eval_dataset = generate_evaluation_data()
    model_list = [model_oai, model_llama, model_mistral]

    for model in model_list:
        print("Evaluating: " + model.llm_type)
        evaluation = weave.Evaluation(
            dataset=eval_dataset,
            scorers=[
                success_in_timekeeping_scorer
            ],
        )
        asyncio.run(evaluation.evaluate(model))


evaluate()
