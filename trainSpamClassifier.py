from pathlib import Path
import os
import pandas
import pandas as pd

# This function takes a path, label and column definition to return a pd df from the path with [filename, data, label]
def walk_and_store (path:str, label:str, columns:list[str], existingdf = pd.DataFrame) -> pd.DataFrame:
    #Create df if not given
    if existingdf.empty:
        existingdf = pd.DataFrame(columns=columns)

    #Fill Df with path given
    for root, dirs, files in os.walk(Path(path)):
        for file in files:
            data = open(path+file, 'rb').read()
            existingdf = pd.concat([pd.DataFrame([[file, data, label]], columns=existingdf.columns), existingdf],
                                ignore_index=True)

    return existingdf


def fetch_data() -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    if Path("/Users/abrahamleal/PycharmProjects/learningML/spamassassin/spam.pkl").exists():
        print("Loading DFs from disk")
        spam = pd.read_pickle(Path("/Users/abrahamleal/PycharmProjects/learningML/spamassassin/spam.pkl"))
        easyham = pd.read_pickle(Path("/Users/abrahamleal/PycharmProjects/learningML/spamassassin/easyham.pkl"))
        hardham = pd.read_pickle(Path("/Users/abrahamleal/PycharmProjects/learningML/spamassassin/hardham.pkl"))
        return spam, easyham, hardham

    #Define Paths to go fetch
    spam="/Users/abrahamleal/PycharmProjects/learningML/spamassassin/spam/"
    spam2="/Users/abrahamleal/PycharmProjects/learningML/spamassassin/spam_2/"
    easyham = "/Users/abrahamleal/PycharmProjects/learningML/spamassassin/easy_ham/"
    easyham2 = "/Users/abrahamleal/PycharmProjects/learningML/spamassassin/easy_ham_2/"
    hardham = "/Users/abrahamleal/PycharmProjects/learningML/spamassassin/hard_ham/"

    # Define DFs
    columns = ["emailid", "text", "label"]
    allspam = walk_and_store(spam, "spam", columns)
    allspam = walk_and_store(spam2, "spam", columns, allspam)
    alleasyham = walk_and_store(easyham, "easyham", columns)
    alleasyham = walk_and_store(easyham2, "easyham", columns, alleasyham)
    allhardham = walk_and_store(hardham, "hardham", columns)

    allspam.to_pickle(Path("/Users/abrahamleal/PycharmProjects/learningML/spamassassin/spam.pkl"))
    alleasyham.to_pickle(Path("/Users/abrahamleal/PycharmProjects/learningML/spamassassin/easyham.pkl"))
    allhardham.to_pickle(Path("/Users/abrahamleal/PycharmProjects/learningML/spamassassin/hardham.pkl"))

    return allspam, alleasyham, allhardham


def explore_data():
    spam, easyham, hardham = fetch_data()

    print(spam.describe())








#def massageData:



#def trainModel:



#def testModel:




#def predict:

explore_data()