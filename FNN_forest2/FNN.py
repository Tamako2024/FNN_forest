import torch.nn as nn


# FNN模型
class FeedForwardNN(nn.Module):
    def __init__(self, input_size=12, hidden_sizes=None, output_size=1):
        super(FeedForwardNN, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [400, 200, 100, 50, 25]
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        self.input_layer = nn.Linear(self.input_size, self.hidden_sizes[0])
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1]) for i in range(len(self.hidden_sizes) - 1)])
        self.output_layer = nn.Linear(self.hidden_sizes[-1], self.output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        x = self.output_layer(x)
        return x

