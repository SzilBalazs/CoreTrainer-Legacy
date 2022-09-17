import torch
import torch.nn as nn
import torch.quantization


L_0_IN = 768
L_1_SIZE = 256

"""
class NNUE(nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()

        self.l0 = nn.Linear(L_0_IN, L_1_SIZE)
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(L_1_SIZE, 1)

    def forward(self, x):
        x = self.l0(x)
        x = self.relu(x)
        x = torch.sigmoid(self.l1(x))
        return x
"""


class NNUE(nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()

        self.l0 = nn.Linear(L_0_IN, L_1_SIZE)
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(L_1_SIZE, 1)

    def forward(self, x):

        x = self.l0(x)
        x = self.relu(x)
        x = self.l1(x)

        return x
