import logging
import os

import cfg
from data.dataset import create_segmentation_dataset, save_dataset, create_dataset
from data.extraction import DEFAULT_NUM_SAMPLES, run_extraction
from utils.file import remake_folder

cfg.configLog()


def prepare_folders(dts_folder):
    output_mask_folder = dts_folder + "/masks"
    samples_folder = dts_folder + "/samples"
    # re-create folders
    remake_folder(output_mask_folder)
    remake_folder(samples_folder)


if __name__ == '__main__':

    version = "v4"
    dts_folder = cfg.dataset(f'{version}')

    do_extraction = True
    do_create_dts = True

    NUM_SAMPLES_PER_RASTER = {"0": 75, "1": 25, "mixed": 300}

    if do_extraction:
        sample_size = (256, 256)
        raster_base_folder = dts_folder + "/extractions"
        raster_samples = [
            ['0425_1-1', {"0": 25, "1": 0, "mixed": 100},
             os.path.join(raster_base_folder, 'PNOA_CYL_2020_25cm_OF_etrsc_rgb_hu30_h05_0425_1-1_extraction.tiff')],
            ['0425_2-1', NUM_SAMPLES_PER_RASTER,
             os.path.join(raster_base_folder, 'PNOA_CYL_2020_25cm_OF_etrsc_rgb_hu30_h05_0425_2-1_extraccion.tiff')],
            ['0398_3-1', NUM_SAMPLES_PER_RASTER,
             os.path.join(raster_base_folder, 'PNOA_CYL_2020_25cm_OF_etrsc_rgb_hu30_h05_0398_3-1_extraccion.tiff')],
            ['0398_3-1B', {"0": 25, "1": 0, "mixed": 100},
             os.path.join(raster_base_folder, 'PNOA_CYL_2020_25cm_OF_etrsc_rgb_hu30_h05_0398_3-1_extraccion_B.tiff')],
            ['0345_6-4', {"0": 75, "1": 0, "mixed": 200},
             os.path.join(raster_base_folder, 'PNOA_CYL_2020_25cm_OF_etrsc_rgb_hu30_h05_0345_6-4_extraccion.tiff')],
            ['0373_7-2', NUM_SAMPLES_PER_RASTER,
             os.path.join(raster_base_folder, 'PNOA_CYL_2020_25cm_OF_etrsc_rgb_hu30_h05_0373_7-2_extraction.tiff')],
        ]
        sample_file = cfg.resource('dataset/samples.sqlite')
        prepare_folders(dts_folder)
        run_extraction(raster_samples, sample_file, dts_folder, sample_size, remake_folders=False)

    if do_create_dts:
        image_size = 128
        dts_file = os.path.join(dts_folder, f"dataset_{image_size}.pickle")
        sample_folder = os.path.join(dts_folder, "samples")

        train_samples, test_samples = create_segmentation_dataset(sample_folder, image_size)
        dataset = create_dataset(train_samples, test_samples)
        save_dataset(dataset, dts_file)
        logging.info(f"Dataset stored in file {dts_file} ")
