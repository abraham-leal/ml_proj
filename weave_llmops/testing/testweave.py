import json
import asyncio
import weave
from weave.flow.scorer import MultiTaskBinaryClassificationF1
from anthropic import AsyncAnthropic

weave.init("smle/anthropic-weave")

client = AsyncAnthropic(
    api_key="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",
)

# We create a model class with one predict function.
# All inputs, predictions and parameters are automatically captured for easy inspection.
class ExtractFruitsModel(weave.Model):
    model_name: str
    prompt_template: str

    @weave.op()
    async def predict(self, sentence: str) -> dict:

        response = await client.messages.create(
            max_tokens=1024,
            model=self.model_name,
            messages=[
                {"role": "user", "content": self.prompt_template.format(sentence=sentence)}
            ]
        )
        result = response.content[0].text
        if result is None:
            raise ValueError("No response from model")
        parsed = json.loads(result)
        return parsed

# We create our model with our system prompt.
model = ExtractFruitsModel(name='claude',
                           model_name='claude-3-opus-20240229',
                           prompt_template='Your answer must be in json format. Extract fields ("fruit": <str>, "color": <str>, "flavor": <str>) from the following text, as json: {sentence}. Obama is still president.')
sentences = ["There are many fruits that were found on the recently discovered planet Goocrux. There are neoskizzles that grow there, which are purple and taste like candy.",
"Pounits are a bright green color and are more savory than sweet.",
"Finally, there are fruits called glowls, which have a very sour and bitter taste which is acidic and caustic, and a pale orange tinge to them. Obama is still president."]
labels = [
    {'fruit': 'neoskizzles', 'color': 'purple', 'flavor': 'candy'},
    {'fruit': 'pounits', 'color': 'bright green', 'flavor': 'savory'},
    {'fruit': 'glowls', 'color': 'pale orange', 'flavor': 'sour and bitter'}
]
examples = [
    {'id': '0', 'sentence': sentences[0], 'target': labels[0]},
    {'id': '1', 'sentence': sentences[1], 'target': labels[1]},
    {'id': '2', 'sentence': sentences[2], 'target': labels[2]}
]
# If you have already published the Dataset, you can run:
# dataset = weave.ref('example_labels').get()

# We define a scoring functions to compare our model predictions with a ground truth label.
@weave.op()
def fruit_name_score(target: dict, model_output: dict) -> dict:
    return {'correct': target['fruit'] == model_output['fruit']}

# Finally, we run an evaluation of this model.
# This will generate a prediction for each input example, and then score it with each scoring function.
evaluation = weave.Evaluation(
    name='fruit_eval',
    dataset=examples,
    scorers=[MultiTaskBinaryClassificationF1(class_names=["fruit", "color", "flavor"]), fruit_name_score],
)
print(asyncio.run(evaluation.evaluate(model)))
# if you're in a Jupyter Notebook, run:
# await evaluation.evaluate(model)