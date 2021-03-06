import glob
import numpy as np

from unittest import TestCase
from utils.batch_generator import BatchGenerator, NoDataPath

DATA_PATH = 'test_data/'


class TestBaseGenerator(TestCase):
    """Class testing a batch generator"""

    def setUp(self):
        self.batch_gen = BatchGenerator(DATA_PATH, 2)
        self.batch_gen.load_data()

    def test_check_init(self):
        self.assertEqual(self.batch_gen.data_dir, DATA_PATH)
        self.assertEqual(self.batch_gen.batch_size, 2)

    def test_no_data_path(self):
        with self.assertRaises(NoDataPath):
            self.batch_gen.data_dir = None

        with self.assertRaises(NoDataPath):
            self.batch_gen.data_dir = ''

    def test_load_files_names(self):
        f_names = np.array([
            'test_data/96.jpg',
            'test_data/97.jpg',
            'test_data/98.jpg'
        ])
        np.testing.assert_equal(self.batch_gen.images_pairs, f_names)

    def test_num_batches(self):
        self.assertEqual(self.batch_gen.num_batches, 2)

    def test_loaded_dataset_shape(self):
        x_shape = (2, 256, 256, 3)
        y_shape = (2, 256, 256, 3)
        x, y = next(self.batch_gen.train_batches)
        self.assertEqual(x.shape, x_shape)
        self.assertEqual(y.shape, y_shape)
        self.assertEqual(y.max(), 1)
        self.assertEqual(y.min(), -1)
