import torch
from torch import nn


def evaluateModel(model: nn.Module, dataloader, loss_fn, vtc, vts, vrl, device):
    model.eval()
    with torch.no_grad():
        for idx, (data, label) in enumerate(dataloader):
            data.to(device)
            label.to(device)

            output = model(data)
            loss = loss_fn(output, label.unsqueeze(1))
            _, predicted = torch.max(output, 1)

            vtc += (predicted == label).sum().item()
            vts += label.size(0)
            vrl += loss.item()

    return vtc, vts, vrl