import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window


def _write_raster(img, img_filename, meta):
    with rasterio.open(img_filename, 'w', **meta, compress="JPEG") as dst:  # , photometric="YCBCR"
        for ch in range(img.shape[-1]):
            # iterate over channels and write bands
            img_channel = img[:, :, ch]
            dst.write(img_channel, ch + 1)  # rasterio bands are 1-indexed


def georeference_image(img, img_source, img_filename, scale=1, bands=3):
    """
    Creates a tiff raster to store the img param using geolocation
    information from another raster.

    :param img: Numpy array with shape [height, width, channels]
    :param img_source: raster to take the geolocation information from.
    :param img_filename: output raster filename
    :param scale: scale rate to apply to output image
    :param bands: number of bands in the output raster
    :return:
    """

    with rasterio.Env():
        # read profile info from first file
        dataset = rasterio.open(img_source)
        meta = dataset.meta.copy()
        dataset.close()

        original_transform = meta["transform"]

        meta.update({"driver": "GTiff", "count": bands, 'dtype': 'uint8'})
        meta.update({"width": img.shape[1], "height": img.shape[0]})
        new_affine = original_transform * Affine.scale(scale, scale)
        meta.update({"transform": new_affine})
        _write_raster(img, img_filename, meta)
