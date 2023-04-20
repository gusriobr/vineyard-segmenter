import os
import shutil
import tempfile
import unittest

import numpy as np

import cfg
from vsegmenter.data.dataset import Dataset


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.dts = Dataset(self.temp_dir, img_size=128)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _prepare_dts_data(self):
        extraction_folder = os.path.join(self.temp_dir, "extractions")
        os.mkdir(extraction_folder)
        # copy one extraction file
        extraction_sample = cfg.resources("extractions/PNOA_CYL_2020_25cm_OF_etrsc_rgb_hu30_h05__0425_2-1_0.tiff")
        shutil.copy(extraction_sample, extraction_folder)
        sample_file = cfg.resources('dataset/samples.sqlite')
        return sample_file

    def test_extract_samples(self):
        # prepare
        sample_file = self._prepare_dts_data()

        # act
        self.dts.extract_samples(sample_file)

        # assert
        mask_folder = self.dts.masks_folder
        files = os.listdir(mask_folder)
        assert len(
            files) == 1, f"Expected folder {mask_folder} to contain exactly one file, but found {len(files)} files"

        sample_folder = self.dts.samples_folder
        assert len(os.listdir(sample_folder)) > 0, "Folder is empty"

    def test_save_samples(self):
        # prepare
        sample_file = self._prepare_dts_data()
        self.dts.extract_samples(sample_file)

        # act
        self.dts.save_samples()

        assert os.path.exists(self.dts.dataset_file), f"Dataset file not found {self.dts.dataset_file}"
        assert os.path.exists(self.dts.samples_index), f"Dataset samples index file not found {self.dts.samples_index}"

    def test_load_dataset(self):
        # prepare
        sample_file = self._prepare_dts_data()
        self.dts.extract_samples(sample_file)
        self.dts.save_samples()

        # act
        x_train, y_train, x_test, y_test = Dataset.load_from_folder(self.temp_dir)
        assert isinstance(x_train, np.ndarray) and x_train.size > 0, "x_train es nulo o no es un ndarray de NumPy"
        assert isinstance(y_train, np.ndarray) and y_train.size > 0, "y_train es nulo o no es un ndarray de NumPy"
        assert isinstance(x_test, np.ndarray) and x_test.size > 0, "x_test es nulo o no es un ndarray de NumPy"
        assert isinstance(y_test, np.ndarray) and y_test.size > 0, "y_test es nulo o no es un ndarray de NumPy"


if __name__ == "__main__":
    unittest.main()
