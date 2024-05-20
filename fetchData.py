import datasets
import torchtext
import wandb
import torch
import torch.nn as nn

from NBoW import NBoW

special_tokens = ["<unk>", "<pad>"]

def tokenize(example, tokenizer, max_length):
    tokens = tokenizer(example["text"])[:max_length]
    return {"tokens": tokens}

def numericalize(example, vocab):
    ids = vocab.lookup_indices(example["tokens"])
    return {"ids": ids}

def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_ids = [i["ids"] for i in batch]
        batch_ids = nn.utils.rnn.pad_sequence(
            batch_ids, padding_value=pad_index, batch_first=True
        )
        batch_label = [i["label"] for i in batch]
        batch_label = torch.stack(batch_label)
        batch = {"ids": batch_ids, "label": batch_label}
        return batch

    return collate_fn

def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader

def fetchAndTokenize(run):
    # Get data from hugging face
    train_data, test_data = datasets.load_dataset("imdb", split=["train", "test"])
    tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

    tokenized_train_data = train_data.map(
        tokenize, fn_kwargs={"tokenizer": tokenizer, "max_length": wandb.config["max_length"]}
    )
    tokenized_test_data = test_data.map(
        tokenize, fn_kwargs={"tokenizer": tokenizer, "max_length": wandb.config["max_length"]}
    )

    tokenized_train_valid_data = tokenized_train_data.train_test_split(test_size=wandb.config["test_size"])
    tokenized_valid_train_data = tokenized_train_valid_data["train"]
    tokenized_valid_valid_data = tokenized_train_valid_data["test"]

    vocab = torchtext.vocab.build_vocab_from_iterator(
        tokenized_train_valid_data["train"]["tokens"],
        min_freq=wandb.config["min_freq"],
        specials=special_tokens,
    )

    unk_index = vocab["<unk>"]
    pad_index = vocab["<pad>"]

    vocab.set_default_index(unk_index)

    numericalized_train_data = tokenized_valid_train_data.map(numericalize, fn_kwargs={"vocab": vocab})
    numericalized_valid_data = tokenized_valid_valid_data.map(numericalize, fn_kwargs={"vocab": vocab})
    numericalized_test_data = tokenized_test_data.map(numericalize, fn_kwargs={"vocab": vocab})


    torch_train_data = numericalized_train_data.with_format(type="torch", columns=["ids", "label"])
    torch_valid_data = numericalized_valid_data.with_format(type="torch", columns=["ids", "label"])
    torch_test_data = numericalized_test_data.with_format(type="torch", columns=["ids", "label"])

    train_data_loader = get_data_loader(torch_train_data, wandb.config["batch_size"], pad_index, shuffle=True)
    valid_data_loader = get_data_loader(torch_valid_data, wandb.config["batch_size"], pad_index)
    test_data_loader = get_data_loader(torch_test_data, wandb.config["batch_size"], pad_index)

    output_dim = len(torch_train_data.unique("label"))

    model = NBoW(len(vocab), wandb.config["embedding_dim"], output_dim, pad_index)

    return vocab, model, tokenizer, train_data_loader, valid_data_loader, test_data_loader


