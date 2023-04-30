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

    version = "v6"
    dts_folder = cfg.dataset(f'{version}')

    do_extraction = True
    do_create_dts = True
    do_sampling = True
    dts = Dataset(dts_folder, img_size=128)

    sample_size = 256
    sample_file = cfg.resources('dataset/samples.sqlite')
    raster_base_folder = dts_folder + "/extractions"

    if do_extraction:
        dts.extract_rasters(sample_file)

    if do_sampling:
        sampling_info = dts.get_sampling_info(sample_file)
        dts.sample_images(sample_file, sampling_info, sample_size=sample_size)

    if do_create_dts:
        dts.save_samples()
