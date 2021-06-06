import os
import numpy as np
from maxvol_compression.frequent_directions import FastFrequentDirections

class TestFastFrequentDirections:

    def test_create(self):
        fds = FastFrequentDirections(150, 400, keep_original=True)
        assert fds.sketch_matrix.shape[0] == 150 and fds.sketch_matrix.shape[1] == 400, "Wrong sketch matrix shape"
        assert fds.keep_original == True, "Wrong keep_original attribute"

    def test_save(self):
        fds = FastFrequentDirections(150, 400, keep_original=False)
        fds.save('test_save_sketch_matrix')
        assert os.path.exists('test_save_sketch_matrix.npy'), "No file was created during save method"
        os.remove('test_save_sketch_matrix.npy')

    def test_load(self):
        fds = FastFrequentDirections(150, 400, keep_original=False)
        new_matrix = np.ones((150, 400), dtype=np.float32)
        assert new_matrix.shape == fds.sketch_matrix.shape
        np.save('test_load_sketch_matrix.npy', new_matrix)
        fds.load('test_load_sketch_matrix.npy')
        assert np.array_equal(fds._sketch[:fds.l], new_matrix)
        assert np.all(fds._sketch[fds.l:, :] == 0)
        os.remove('test_load_sketch_matrix.npy')
