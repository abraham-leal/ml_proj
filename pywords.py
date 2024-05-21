import collections


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import tqdm
import wandb
import random
from fetchData import fetchAndTokenize

run = wandb.init(project="imdb-sentiment-analysis-model-creation", config="./config.yaml")

np.random.seed(wandb.config["seed"])
torch.manual_seed(wandb.config["seed"])
torch.cuda.manual_seed(wandb.config["seed"])
torch.backends.cudnn.deterministic = True


def train(data_loader, model, criterion, optimizer, device):
    model.train()
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm.tqdm(data_loader, desc="training..."):
        ids = batch["ids"].to(device)
        label = batch["label"].to(device)
        prediction = model(ids)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)


def evaluate(data_loader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="evaluating..."):
            ids = batch["ids"].to(device)
            label = batch["label"].to(device)
            prediction = model(ids)
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)


def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def predict_sentiment(text, model, tokenizer, vocab, device):
    tokens = tokenizer(text)
    ids = vocab.lookup_indices(tokens)
    tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    prediction = model(tensor).squeeze(dim=0)
    probability = torch.softmax(prediction, dim=-1)
    predicted_class = prediction.argmax(dim=-1).item()
    predicted_probability = probability[predicted_class].item()
    return predicted_class, predicted_probability


vocab, model, tokenizer, train_data_loader, valid_data_loader, test_data_loader, test_data = fetchAndTokenize(run)

print(f"The model has {count_parameters(model):,} trainable parameters")

vectors = torchtext.vocab.GloVe()
pretrained_embedding = vectors.get_vecs_by_tokens(vocab.get_itos())
model.embedding.weight.data = pretrained_embedding
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = criterion.to(device)
best_valid_loss = float("inf")
metrics = collections.defaultdict(list)

# Magic
wandb.watch(model, criterion, log="all", log_freq=10)

for epoch in range(wandb.config["epochs"]):
    train_loss, train_acc = train(
        train_data_loader, model, criterion, optimizer, device
    )
    valid_loss, valid_acc = evaluate(valid_data_loader, model, criterion, device)
    metrics["train_losses"].append(train_loss)
    metrics["train_accs"].append(train_acc)
    metrics["valid_losses"].append(valid_loss)
    metrics["valid_accs"].append(valid_acc)
    wandb.log({"accuracy": train_acc, "loss": train_loss, "valid_loss": valid_loss, "valid_acc": valid_acc}, step=epoch)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "nbow.pt")
    print(f"epoch: {epoch}")
    print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
    print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")

model.load_state_dict(torch.load("nbow.pt"))
artifact = wandb.Artifact('model', type='model')
artifact.add_file('nbow.pt')
run.log_artifact(artifact)

test_loss, test_acc = evaluate(test_data_loader, model, criterion, device)
wandb.log({"test_acc": test_acc, "test_loss": test_loss })

testing_table = wandb.Table(columns=["artifact", "text", "sentiment", "confidence"])

text = test_data[random.randrange(1,24999)]["text"]
prediction, confidence = predict_sentiment(text, model, tokenizer, vocab, device)
testing_table.add_data(run.name, text, prediction, confidence)

text = test_data[random.randrange(1,24999)]["text"]
prediction, confidence = predict_sentiment(text, model, tokenizer, vocab, device)
testing_table.add_data(run.name,text, prediction, confidence)

text = test_data[random.randrange(1,24999)]["text"]
prediction, confidence = predict_sentiment(text, model, tokenizer, vocab, device)
testing_table.add_data(run.name,text, prediction, confidence)

text = test_data[random.randrange(1,24999)]["text"]
prediction, confidence = predict_sentiment(text, model, tokenizer, vocab, device)
testing_table.add_data(run.name,text, prediction, confidence)

run.log({"predictionResults": testing_table})
run.finish()



