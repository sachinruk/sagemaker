from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, nodes: List[int]) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(x, y) for x, y in zip(nodes, nodes[1:])])
        self.activation = F.relu

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        return self.layers[-1](x)
