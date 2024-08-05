import weave
from weave.flow.scorer import Scorer

weave.init("abe_timeentry_agent")

@weave.op()
def generate_evaluation_data(weave_client):
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

    return int_tests

# Define a scorer
@weave.op()
def success_in_timekeeping_scorer(Scorer):
    class_names: list[str]
    @weave.op()
    def summarize(self, score_rows: list) -> Optional[dict]:
        result = {}
        cols = transpose(score_rows)

        for class_name in self.class_names:
            col = cols[class_name]
            tp = sum(r["correct"] and not r["negative"] for r in col)
            fp = sum(not r["correct"] and not r["negative"] for r in col)
            fn = sum(not r["correct"] and r["negative"] for r in col)
            precision, recall, f1 = p_r_f1(tp, fp, fn)
            result[class_name] = {"f1": f1, "precision": precision, "recall": recall}

        return result

    @weave.op()
    def score(self, target: dict, model_output: Optional[dict]) -> dict:
        result = {}
        for class_name in self.class_names:
            class_label = target.get(class_name)
            class_model_output = model_output.get(class_name) if model_output else None
            result[class_name] = {
                "correct": class_label == class_model_output,
                "negative": not class_model_output,
            }
        return result

    context_precision_prompt = f'''Given the question, answer and context verify if the context was useful in arriving at the given answer.
    Give the verdict field a value of 1 if it is useful and a value of 0 if it is not useful.
    Answer only in valid JSON format.

    question: {question}
    context: {context}
    answer: {answer}
    verdict: '''
    client = OpenAI(api_key=userdata.get('OPEN_AI_SECRET'))

    prompt = context_precision_prompt.format(
        question=question,
        context=model_output['context'],
        answer=model_output['answer'],
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={ "type": "json_object" }
    )

    response = json.loads(response.choices[0].message.content)
    return {
        "verdict": int(response["verdict"]) == 1,
    }

def evaluate():
    evaluation = weave.Evaluation(
        dataset=generate_evaluation_data,
        scorers=[
            MultiTaskBinaryClassificationF1(class_names=["fruit", "color", "flavor"]),
            fruit_name_score
        ],
    )
    print(asyncio.run(evaluation.evaluate(model)))