import torch
import torch.nn as nn
import torch.quantization


L_0_IN = 768
L_1_SIZE = 256


class NNUE(nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()

        self.l0 = nn.Linear(L_0_IN, L_1_SIZE)
        self.l1 = nn.Linear(2 * L_1_SIZE, 1)
        self.relu = nn.ReLU()

    def forward(self, white_features, black_features, stm):

        w = self.l0(white_features)
        b = self.l0(black_features)

        x = self.relu(((1-stm) * torch.cat([w, b], dim=1)) + (stm * torch.cat([b, w], dim=1)))
        x = torch.sigmoid(self.l1(x))

        return x
