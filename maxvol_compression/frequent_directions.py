import numpy as np


class FastFrequentDirections:

    def __init__(self, d, l):
        # number of features
        self.d = d
        # desirable rank
        self.l = l
        # doubled rank for faster computation
        self.m = 2 * self.l
        self._sketch = np.zeros((self.m, self.d), dtype=np.float32)

        self.nextZeroRow = 0
        
    def _rotate(self):
        [_, s, Vt] = np.linalg.svd(self._sketch, full_matrices=False)

        sShrunk = np.sqrt(s[:self.l]**2 - s[self.l - 1]**2)
        self._sketch[:self.l:, :] = np.dot(np.diag(sShrunk), Vt[:self.l, :])
        
        self._sketch[self.l:, :] = 0
        self.nextZeroRow = self.l

    def append(self, vector):
        if np.count_nonzero(vector) == 0:
            return

        if self.nextZeroRow >= self.m:
            self._rotate()

        self._sketch[self.nextZeroRow, :] = vector
        self.nextZeroRow += 1
        
    def update(self, batch):
        for row in batch:
            self.append(row)

    @property
    def sketchmatrix(self):
        return self._sketch[:self.l]