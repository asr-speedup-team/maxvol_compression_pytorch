import os
import numpy as np
import pytest
from maxvol_compression.sketch_matrix import FastFrequentDirections, RandomSums


class TestSketchMatrix:

    def test_create_fd(self):
        fds = FastFrequentDirections(150, 400, keep_original=True)
        assert fds.sketch_matrix.shape[0] == 150 and fds.sketch_matrix.shape[1] == 400, "Wrong sketch matrix shape"
        assert fds.keep_original == True, "Wrong keep_original attribute"

    def test_save_fd(self):
        fds = FastFrequentDirections(150, 400, keep_original=False)
        fds.save('test_save_sketch_matrix')
        assert os.path.exists('test_save_sketch_matrix.npy'), "No file was created during save method"
        os.remove('test_save_sketch_matrix.npy')

    def test_load_fd(self):
        fds = FastFrequentDirections(150, 400, keep_original=False)
        new_matrix = np.ones((150 * 2, 400), dtype=np.float32)
        assert new_matrix.shape == fds._sketch.shape
        np.save('test_load_sketch_matrix.npy', new_matrix)
        fds.load('test_load_sketch_matrix.npy')
        assert np.array_equal(fds._sketch, new_matrix)
        os.remove('test_load_sketch_matrix.npy')

    def test_keep_fd(self):
        fds = FastFrequentDirections(50, 50, keep_original=False)
        A = np.random.rand(1000, 50)
        fds.update(A)
        with pytest.raises(NotImplementedError):
            fds.compute_error()
            
            
    def test_create_rs(self):
        rs = RandomSums(150, 400, keep_original=True)
        assert rs.sketch_matrix.shape[0] == 150 and rs.sketch_matrix.shape[1] == 400, "Wrong sketch matrix shape"
        assert rs.keep_original == True, "Wrong keep_original attribute"

    def test_save_rs(self):
        rs = RandomSums(150, 400, keep_original=False)
        rs.save('test_save_sketch_matrix')
        assert os.path.exists('test_save_sketch_matrix.npy'), "No file was created during save method"
        os.remove('test_save_sketch_matrix.npy')

    def test_load_rs(self):
        rs = RandomSums(150, 400, keep_original=False)
        new_matrix = np.ones((150, 400), dtype=np.float32)
        assert new_matrix.shape == rs._sketch.shape
        np.save('test_load_sketch_matrix.npy', new_matrix)
        rs.load('test_load_sketch_matrix.npy')
        assert np.array_equal(rs._sketch, new_matrix)
        os.remove('test_load_sketch_matrix.npy')

    def test_keep_rs(self):
        rs = RandomSums(50, 50, keep_original=False)
        A = np.random.rand(1000, 50)
        rs.update(A)
        with pytest.raises(NotImplementedError):
            rs.compute_error()
