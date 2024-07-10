import wandb
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from dataEng import loadData
from model import imageModel
from train_test import train, test

run = wandb.init(entity="wandb-smle",
                 project="aleal-domain-img",
                 config="./config.yaml",
                 save_code=True,
                 group="debug",
                 job_type="all",
                 force=True,
                 )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

    train_dl, test_dl, validation_dl = loadData(run)

    # initialize model and watch
    model = imageModel().to(device)
    run.watch(model)
    optimizer = optim.Adadelta(model.parameters(), lr=wandb.config["learning_rate"])

    scheduler = StepLR(optimizer, step_size=1, gamma=wandb.config["gamma"])
    for epoch in range(1, wandb.config["epochs"] + 1):
        train(run, model, device, train_dl, optimizer, epoch)
        test(run, model, device, validation_dl, epoch)
        scheduler.step()

    # save the model
    model_art = wandb.Artifact(type="model", name="DomainNet-Model")
    torch.save(model.state_dict(), "dn_cnn.pt")
    model_art.add_file("dn_cnn.pt")
    run.log_artifact(model_art)


if __name__ == '__main__':
    main()
