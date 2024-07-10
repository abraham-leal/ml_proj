from datasets import load_dataset, Dataset
import torch
from torchvision.transforms import v2
import torch.utils.data
import wandb



def loadData (run: wandb.run) -> [torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    # fetch from hugging face
    full_load = load_dataset("wltjr1007/DomainNet")
    s_train_data = full_load['train']
    test_set = full_load['test']

    # split train set into train/validation
    split_train_data = s_train_data.train_test_split(test_size=wandb.config["test_size"])
    training_set = split_train_data["train"]
    validation_set = split_train_data["test"]

    # construct wandb artifact, add the datasets
    hf_data_set = wandb.Artifact(name="hf-dataset", type="dataset")
    names = ["training", "validation", "test"]
    datasets = [training_set, validation_set, test_set]

    name: str
    dataset: Dataset
    for name, dataset in zip(names, datasets):
        # 🐣 Store a new file in the artifact, and write something into its contents.
        with hf_data_set.new_file(name + ".pt", mode="wb") as file:
            torch.save(dataset.set_format("torch"), file)


    run.log_artifact(hf_data_set)

    # turn into dataloaders and return

    transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize(size=(300, 300), antialias=True),
        v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_kwargs = {'batch_size': wandb.config["trainBatchSize"]}
    test_kwargs = {'batch_size': 1000}

    torch_training_set = training_set.with_format("torch")
    torch_validation_set = validation_set.with_format("torch")
    torch_testing_set = test_set.with_format("torch")

    torch_training_set.set_transform(transforms)
    torch_validation_set.set_transform(transforms)
    torch_testing_set.set_transform(transforms)

    train_loader = torch.utils.data.DataLoader(torch_training_set, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(torch_testing_set, **test_kwargs)
    validation_loader = torch.utils.data.DataLoader(torch_validation_set, **test_kwargs)

    return train_loader, test_loader, validation_loader



