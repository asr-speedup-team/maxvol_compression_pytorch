import torch
from torch.nn import Linear
import numpy as np
from maxvol_compression.layers import LinearMaxvol


class TestLinearMaxvol:

    def test_create(self):
        linear = Linear(20, 30)
        idxs = np.random.choice(range(20), size=5)
        V = np.random.rand(30, 10)
        linearmaxvol = LinearMaxvol(linear, idxs, V)
        assert linearmaxvol.invSV.shape == (10, 5), 'wrong shape of invSV matrix'

    def test_forward(self):
        linear = Linear(20, 30)
        idxs = np.random.choice(range(20), size=5)
        V = np.random.rand(30, 10)
        x = np.random.rand(100, 20)
        linearmaxvol = LinearMaxvol(linear, idxs, V)
        with torch.no_grad():
            output = linearmaxvol(torch.Tensor(x))
        assert output.shape == (100, 30)

    def test_backward(self):
        linear = Linear(20, 30)
        idxs = np.random.choice(range(20), size=5)
        V = np.random.rand(30, 10)
        linearmaxvol = LinearMaxvol(linear, idxs, V)
        x = np.random.rand(100, 20)
        output = linearmaxvol(torch.Tensor(x)).sum()
        output.backward()
        assert linearmaxvol.V.grad is not None
        assert linearmaxvol.invSV.grad is not None
        assert linearmaxvol.weight.grad is not None
        assert linearmaxvol.bias.grad is not None