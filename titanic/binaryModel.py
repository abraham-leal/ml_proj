from torch import nn


class binaryModel(nn.Module):
    def __init__(self):
        super(binaryModel, self).__init__()
        self.hidden = nn.Linear(6, 100)
        self.relu = nn.ReLU()
        self.hidden2 = nn.Linear(100, 100)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

        # Regularization
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.hidden(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.dropout(x)
        x = self.relu2(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x