import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)

        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def get_mask(self):
        pass

    def forward(self, inputs, lengths):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]

        weights = torch.bmm(
            torch.tanh(inputs),
            self.att_weights
            .permute(1, 0)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )

        attentions = torch.softmax(F.relu(weights.squeeze(-1)), dim=-1)

        mask = torch.ones(attentions.size(), device=inputs.device, requires_grad=False)
        for i, l in enumerate(lengths):
            if l < max_len:
                mask[i, l:] = 0

        masked = attentions * mask
        _sums = masked.sum(-1).unsqueeze(-1)
        attentions = masked.div(_sums)

        if attentions.dim() == 1:
            attentions = attentions.unsqueeze(1)

        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))
        representations = weighted.sum(1).squeeze()

        return representations, attentions
