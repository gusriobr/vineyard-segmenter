import os

import cfg
from data.dataset import create_segmentation_dataset, save_dataset, create_dataset
from data.extraction import burn_samples, extract_images

cfg.configLog()

if __name__ == '__main__':
    raster_folder = "/media/gus/data/viticola/raster"
    label = "v1"
    dataset_folder = cfg.dataset(label)
    image_size = 128
    dts_file = cfg.resource(f"dataset/{label}_{image_size}.pickle")

    samples_file = cfg.resource('dataset/samples.sqlite')
    raster_sample = cfg.resource('raster/PNOA_CYL_2020_25cm_OF_etrsc_rgb_hu30_h05_0373_7-2_extraction.tiff')
    mask_file = cfg.resource('raster/PNOA_CYL_2020_25cm_OF_etrsc_rgb_hu30_h05_0373_7-2_extraction_mask.tiff')
    burn_samples(samples_file, raster_sample, mask_file)

    sample_folder = os.path.join(dataset_folder, "samples")
    extract_images(raster_sample, mask_file, sample_folder, num_polygons=500)

    train_samples, test_samples = create_segmentation_dataset(sample_folder, image_size)
    dataset = create_dataset(train_samples, test_samples)
    save_dataset(dataset, dts_file)
