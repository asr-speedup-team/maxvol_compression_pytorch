import torch
from torch.nn.parameter import Parameter
from torch.nn import Module
import torch.nn.functional as F
import numpy as np


class LinearMaxvol(Module):
    def __init__(self, linear, idxs, V, device='cpu'):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.device = device

        self.idxs = idxs
        with torch.no_grad():
            self.weight = Parameter(linear.weight[idxs].detach()).to(device)
            if linear.bias is not None:
                self.bias = Parameter(linear.bias[idxs].detach()).to(device)
            else:
                self.register_parameter('bias', None)
            self.V = Parameter(torch.Tensor(V)).to(device)
            self.invSV = Parameter(torch.Tensor(np.linalg.pinv(V[idxs, :]))).to(device)

    def forward(self, input):
        x = F.linear(input, self.weight, self.bias)
        return (self.V @ self.invSV @ x.T).T

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, idxs_len={}'.format(
            self.in_features, self.out_features, self.bias is not None, len(self.idxs)
        )