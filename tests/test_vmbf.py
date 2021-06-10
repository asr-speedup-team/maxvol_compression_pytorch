import os
import numpy as np
import pytest
from maxvol_compression.vmbf import EVBMF


class TestVMBF:

    def test_args(self):
        A = np.random.rand(80,100)
        _, s1, V1, post1 = EVBMF(A)
        _, s2, V2, post2 = EVBMF(None, pretrained_svd=np.linalg.svd(A, full_matrices=False))
        assert np.array_equal(s1, s2) and np.array_equal(V1, V2), 'Results differ for equivalent input'
