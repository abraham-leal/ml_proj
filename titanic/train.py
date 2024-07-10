import torch
from torch import nn


def trainModel(model: nn.Module, dataloader, optimizer, loss_fn, ttc, tts, trl, device):
    model.train()
    for idx, (data, label) in enumerate(dataloader):
        data.to(device)
        label.to(device)

        output = model(data)
        loss = loss_fn(output, label.unsqueeze(1))
        optimizer.zero_grad()
        _, predicted = torch.max(output, 1)
        loss.backward()
        optimizer.step()

        ttc += (predicted == label).sum().item()
        tts += label.size(0)
        trl += loss.item()

    return ttc, tts, trl
