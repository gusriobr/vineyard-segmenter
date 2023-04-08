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


def georeference_image(img, img_source, img_filename, scale=1, bands=3, reproject_raster=False):
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

        print(meta)
        meta.update({"driver": "GTiff", "count": bands, 'dtype': 'uint8'})
        if not reproject_raster:
            meta.update({"width": img.shape[1], "height": img.shape[0]})
            new_affine = original_transform * Affine.scale(scale, scale)
            meta.update({"transform": new_affine})
            _write_raster(img, img_filename, meta)
        else:
            # meta.update({"width": img.shape[1], "height": img.shape[0]})
            dest_shape = (meta['height'], meta['width'], bands)
            dest_img = np.zeros(dest_shape, dtype=np.uint8)
            dest_transform = original_transform
            reproject(img, dest_img, src_transform=original_transform, src_crs=meta['crs'],
                      dst_transform=dest_transform, dst_crs=meta['crs'], resampling=Resampling.nearest,
                      dst_shape=dest_shape)
            _write_raster(dest_img, img_filename, meta)
