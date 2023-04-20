import os

import cfg
from data.dataset import Dataset
from utils.file import remake_folder

cfg.configLog()


def prepare_folders(dts_folder):
    output_mask_folder = dts_folder + "/masks"
    samples_folder = dts_folder + "/samples"
    # re-create folders
    remake_folder(output_mask_folder)
    remake_folder(samples_folder)


if __name__ == '__main__':

    version = "v5"
    dts_folder = cfg.dataset(f'{version}')

    do_extraction = False
    do_create_dts = True
    dts = Dataset(dts_folder, img_size=128)

    NUM_SAMPLES_PER_RASTER = {"0": 75, "1": 25, "mixed": 300}

    if do_extraction:
        sample_size = 256
        raster_base_folder = dts_folder + "/extractions"
        raster_samples = [
            # custom sample number
            [{"0": 75, "1": 0, "mixed": 200},
             os.path.join(raster_base_folder, 'PNOA_CYL_2020_25cm_OF_etrsc_rgb_hu30_h05__0345_6-4_0.tiff')],
            [{"0": 30, "1": 0, "mixed": 100},
             os.path.join(raster_base_folder, 'PNOA_CYL_2020_25cm_OF_etrsc_rgb_hu30_h05__0398_3-1_1.tiff')],
            [{"0": 30, "1": 0, "mixed": 100},
             os.path.join(raster_base_folder, 'PNOA_CYL_2020_25cm_OF_etrsc_rgb_hu30_h05__0425_1-1_0.tiff')],
            [{"0": 30, "1": 0, "mixed": 0},
             os.path.join(raster_base_folder, 'PNOA_CYL_2020_25cm_OF_etrsc_rgb_hu30_h05__0373_5-1_0.tiff')],
            [{"0": 30, "1": 0, "mixed": 0},
             os.path.join(raster_base_folder, 'PNOA_CYL_2020_25cm_OF_etrsc_rgb_hu30_h05__0373_5-2_0.tiff')],
            [{"0": 30, "1": 30, "mixed": 30},
             os.path.join(raster_base_folder, 'PNOA_CYL_2020_25cm_OF_etrsc_rgb_hu30_h05__0373_7-2_1.tiff')],
            # regular sample number
            [NUM_SAMPLES_PER_RASTER,
             os.path.join(raster_base_folder, 'PNOA_CYL_2020_25cm_OF_etrsc_rgb_hu30_h05__0373_7-2_0.tiff')],
            [NUM_SAMPLES_PER_RASTER,
             os.path.join(raster_base_folder, 'PNOA_CYL_2020_25cm_OF_etrsc_rgb_hu30_h05__0398_3-1_0.tiff')],
            [NUM_SAMPLES_PER_RASTER,
             os.path.join(raster_base_folder, 'PNOA_CYL_2020_25cm_OF_etrsc_rgb_hu30_h05__0425_2-1_0.tiff')],
        ]
        sample_file = cfg.resources('dataset/samples.sqlite')
        dts.extract_samples(sample_file, raster_samples, sample_size=sample_size)

    if do_create_dts:
        dts.save_samples()
