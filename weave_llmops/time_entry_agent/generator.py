import weave
from weave import Dataset

# Initialize Weave
weave.init('abe-weave-agent')

def generate_evaluation_data():
    dataset = Dataset(name='evaluation_directions', rows=[
        {'question': "He no likes ice cream."},
        {'question': "She goed to the store."},
        {'question': "They plays video games all day."}
    ])

    # Publish the dataset
    weave.publish(dataset)