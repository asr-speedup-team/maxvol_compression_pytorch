from abc import ABC, abstractmethod

import numpy as np


class SketchMatrixAlg(ABC):
    """Class for sketch matrix approximation of streaming data.
    """
    def __init__(self, l, d, keep_original=False):
        # desirable rank
        self.l = l
        # number of features
        self.d = d
        # original matrix. only updated during test runs to compute error
        self._keep_original = keep_original
        self._original = np.empty((0, self.d), dtype=np.float32)

    @abstractmethod
    def append(self, vector):
        pass
        
    def update(self, batch):
        for row in batch:
            self.append(row)
        if self.keep_original:
            self._original = np.vstack((self._original, batch))

    def save(self, name='sketch_matrix'):
        np.save(f'{name}.npy', self._sketch)
        print(f'Sketch matrix was saved to {name}.npy')

    def load(self, filename):
        value = np.load(filename)
        self.sketch_matrix = value
        print(f'Sketch matrix was loaded from {filename}')
    
    @property
    def original_matrix(self):
        return self._original
    
    @property
    def keep_original(self):
        return self._keep_original
    
    def compute_error(self):
        if not self.keep_original:
            raise NotImplementedError('keep_original attribute is False. Computation of error is not possible, because original matrix was not stored.')
        assert self.sketch_matrix.shape[1] == self.original_matrix.shape[1], 'Wrong shape of sketch matrix or original matrix.'
        return np.max(self.original_matrix.T @ self.original_matrix - self.sketch_matrix.T @ self.sketch_matrix)

    
class FastFrequentDirections(SketchMatrixAlg):
    """Class for low-rank approximation of streaming data.
    Implementation follows https://arxiv.org/abs/1501.01711
    """
    def __init__(self, l, d, keep_original=False):
        super().__init__(l, d, keep_original)
        # doubled rank for faster computation
        self._sketch = np.zeros((2*self.l, self.d), dtype=np.float32)
        self.nextZeroRow = 0

    def append(self, vector):
        if np.count_nonzero(vector) == 0:
            return

        if self.nextZeroRow >= 2*self.l:
            [_, s, Vt] = np.linalg.svd(self._sketch, full_matrices=True)
            # nullifying every row with index > self.l
            s = np.maximum(s[:self.l]**2 - s[self.l - 1]**2, 0)
            self._sketch[:self.l:, :] = np.sqrt(np.diag(s)) @ Vt[:self.l, :]
            self._sketch[self.l:, :] = 0
            self.nextZeroRow = self.l

        self._sketch[self.nextZeroRow, :] = vector
        self.nextZeroRow += 1

    @property
    def sketch_matrix(self):
        return self._sketch[:self.l]

    @property
    def sketch_matrix_rotated(self):
        [_, s, Vt] = np.linalg.svd(self._sketch, full_matrices=False)
        return s[:self.l], Vt[:self.l]

    @sketch_matrix.setter
    def sketch_matrix(self, value):
        assert value.shape == self._sketch.shape, "Error trying to load sketch matrix: wrong shape"
        assert value.dtype == self._sketch.dtype, "Error trying to load sketch matrix: wrong dtype"
        self._sketch = value
        zero_rows = np.argwhere(np.all(value == 0, axis=1))
        if len(zero_rows) == 0:
            self.nextZeroRow = 2*self.l
        else:
            self.nextZeroRow = zero_rows[0][0]
    

class RandomSums(SketchMatrixAlg):

    def __init__(self, l, d, keep_original):
        super().__init__(l, d, keep_original)
        self._sketch = np.zeros((self.l, self.d), dtype=np.float32)
        self.signs = [1.0, -1.0]
        
    def append(self, vector):
        row = np.random.randint(self.l)
        sign = np.random.choice(self.signs)

        self._sketch[row, :] += np.array(sign * vector)
        
    @property
    def sketch_matrix(self):
        return self._sketch

    @property
    def sketch_matrix_rotated(self):
        [_, s, Vt] = np.linalg.svd(self.sketch_matrix, full_matrices=False)
        return s, Vt

    @sketch_matrix.setter
    def sketch_matrix(self, value):
        assert value.shape == self._sketch.shape, "Error trying to load sketch matrix: wrong shape"
        assert value.dtype == self._sketch.dtype, "Error trying to load sketch matrix: wrong dtype"
        self._sketch = value