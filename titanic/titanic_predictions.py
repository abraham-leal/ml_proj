#!/usr/bin/env python
# coding: utf-8
import sys
import torch
from torch.utils.data import Dataset
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.utils.data
import CustomTitanicDataset
import torch.utils.data
import binaryModel
import evaluate
import train
import os
import logging

logger = logging.getLogger(__name__)
config = { 'batchSize': 64, 'num_epochs': 100, 'lr': 0.01}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getData () -> [pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with wandb.init(entity="aleal", job_type="preprocessing",
        project="aleal-kaggle-titanic", save_code=True,
                 group="finetune", force=True, config=config) as run:
        data_art = wandb.Artifact(name="titatinc_artifacts", type="dataset")

        if os.path.isfile("./titanic/train.csv"):
            logger.info('Using Data from Filesystem')
            train_csv = pd.read_csv("./titanic/train.csv", index_col='PassengerId')
            test_csv = pd.read_csv("./titanic/test.csv", index_col='PassengerId')
            data_art.add_file("./titanic/train.csv")
            data_art.add_file("./titanic/test.csv")
        else:
            logger.info('Using Data from Wandb')
            test_data = run.use_artifact("titatinc_artifacts:latest")
            csvs = test_data.download()
            test_csv = pd.read_csv(csvs+"/test.csv", index_col='PassengerId')
            train_csv = pd.read_csv(csvs+"/train.csv", index_col='PassengerId')

        ## Mod data to fit model
        ##### Drop "Names"
        ##### Drop "Cabin"
        ##### Drop "PassengerId"
        ##### Drop "Ticket"
        ##### Drop "Embarked"
        ##### Numericalize "Sex"
        ##### BUcketize "Age" and fillna
        ##### Generate train/valid/test splits
        logger.info('Performing Data Engineering')
        train_csv = train_csv.drop('Name', axis=1)
        train_csv = train_csv.drop('Cabin', axis=1)
        train_csv = train_csv.drop('Ticket', axis=1)
        train_csv = train_csv.drop('Embarked', axis=1)
        train_csv['Sex'] = train_csv['Sex'].astype('category')
        train_csv['Sex'] = train_csv['Sex'].cat.codes
        bins= [0,18,40,60,100]
        labels = [0,1,2,3]
        train_csv['Age'] = train_csv['Age'].fillna(train_csv['Age'].mean())
        train_csv['Age'] = pd.cut(train_csv['Age'], bins=bins, labels=labels, right=False)

        train, test = train_test_split(train_csv, test_size=0.2)
        train, valid = train_test_split(train, test_size=0.2)
        ### Now we have train, validation, and test splits

        logToWandbArt(run, "train_table", data_art, train)
        logToWandbArt(run, "valid_table", data_art, valid)
        logToWandbArt(run, "test_table", data_art, test)

        run.log_artifact(data_art)
        return train, valid, test


def predictTest(model: nn.Module, dataloader, test: pd.DataFrame):
    logger.info('Predicting Test Set')
    with wandb.init(entity="aleal", job_type="inference",
                    project="aleal-kaggle-titanic", save_code=True,
                    group="finetune", force=True, config=config) as run:
        ### Visualize test dataset
        test_output_table = wandb.Table(columns=["Id", "In_Pclass", "In_Sex", "In_SibSp", "In_Parch",
                                                 "In_Fare", "Prediction", "Ground_Truth", "Pred_Probability"])

        total_correct = 0
        total_tests = 0

        for idx, (data, label) in enumerate(dataloader):
            data.to(device)
            label.to(device)

            output = model(data)
            prediction = output.squeeze(dim=0).item()
            total_tests += 1

            num_pred = 0
            prob = 0

            if (prediction > 0.5):
                num_pred = 1
                prob = prediction
            else:
                prob = 1 - prediction

            curr_row = test.iloc[idx]
            test_output_table.add_data(idx, curr_row['Pclass'], curr_row['Sex'], curr_row['SibSp']
                                       , curr_row['Parch'], curr_row['Fare'], num_pred, curr_row['Survived'], prob)

            if (num_pred == curr_row['Survived']):
                total_correct += 1

        run.log({"test_acc": total_correct / total_tests})
        run.log({"predictions_table": test_output_table})

# Log a pandas df to wandb under the given artifact
def logToWandbArt (run, name, art, df):
    # save source data to W&B
    x_table = wandb.Table(dataframe=df)
    art.add(x_table, name)

def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger.info('Started')

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

    train_d, valid, test = getData()

    with wandb.init(entity="aleal", job_type="training",
               project="aleal-kaggle-titanic", save_code=True,
               group="finetune", force=True, config=config,
                    settings=wandb.Settings(code_dir=".")) as run:

        #Initialize Model and log to wandb
        model = binaryModel.binaryModel()
        model.to(device)
        run.watch(model)

        ### Contruct pytorch dataloaders
        train_ds = CustomTitanicDataset.CustomTitanicDataset(train_d)
        train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=run.config.batchSize, shuffle=True)
        valid_ds = CustomTitanicDataset.CustomTitanicDataset(valid)
        valid_dataloader = torch.utils.data.DataLoader(valid_ds, batch_size=run.config.batchSize, shuffle=True)
        test_ds = CustomTitanicDataset.CustomTitanicDataset(test)
        test_dataloader = torch.utils.data.DataLoader(test_ds)

        ## Training and Validation Loop

        loss_fn = nn.BCELoss()
        loss_fn = loss_fn.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=run.config.lr)

        for epoch in range(run.config.num_epochs):
            ttc = 0
            tts = 0
            trl = 0

            vtc = 0
            vts = 0
            vrl = 0

            #train
            logger.info('Training Epoch:' + str(epoch))
            ttc, tts, trl = train.trainModel(model, train_dataloader, optimizer, loss_fn, ttc, tts, trl, device)
            #evaluate
            vtc, vts, vrl = evaluate.evaluateModel(model, valid_dataloader, loss_fn, vtc, vts, vrl, device)

            train_accuracy = ttc / tts
            train_loss = trl / len(train_ds)
            valid_accuracy = vtc / vts
            valid_loss = vrl / len(valid_ds)
            run.log({"train_acc": train_accuracy, "train_loss": train_loss}, step=epoch)
            run.log({"valid_acc": valid_accuracy, "valid_loss": valid_loss}, step=epoch)

        # Log model to wandb
        model_art = wandb.Artifact(type="model", name="titanic-model")
        with model_art.new_file("titanic-model.pt", mode="wb") as file:
            torch.save(model.state_dict(), file)
        run.log_artifact(model_art)

    predictTest(model, test_dataloader, test)

if __name__ == '__main__':
    main()
    logger.info('Done')

