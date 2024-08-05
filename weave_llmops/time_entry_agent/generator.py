import weave
from weave import Dataset

@weave.op()
def generate_evaluation_data(weave_client):
    dataset = Dataset(name='evaluation_directions', rows=[
        {'question': "Log 8 hours of work for project chameleon and entry code 23523 for today"},
        {'question': "Log 4 hours of work for project ballon with code 4444 and 2 hours of work for project chacha with code 432 for the work week"},
        {'question': "Remove 3 hours of work from project ballon with code 4444 for last tuesday and add 3 hours to project chameleon with code 23523"},
        {'question': "log 2 hours to project chameleon with code 277 and 3 hours to project mobamba with code 542"},
        {'question': "remove 5 hours from project numbnumb with code 567 for last wednesday and add 6 hours for project lolo with code 890"},
        {'question': "Log 8 hours of work with project momo with code 555 for the entire week"}
    ])

    # Publish the dataset
    weave_client.publish(dataset, "evaluation_data")