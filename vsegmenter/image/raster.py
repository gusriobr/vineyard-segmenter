import os

import numpy as np
import rasterio
from PIL import Image
from affine import Affine
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.mask import mask
from rasterio.warp import Resampling
from rasterio.warp import calculate_default_transform
from rasterio.warp import reproject
from shapely.geometry import Polygon
from shapely.geometry import mapping
from shapely.ops import transform as shapely_transform
from pyproj import Transformer
from shapely.ops import transform as shapely_transform
from rasterio.warp import reproject, Resampling
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling


def _write_raster(img, img_filename, meta):
    with rasterio.open(img_filename, 'w', **meta, compress="JPEG") as dst:  # , photometric="YCBCR"
        for ch in range(img.shape[-1]):
            # iterate over channels and write bands
            img_channel = img[:, :, ch]
            dst.write(img_channel, ch + 1)  # rasterio bands are 1-indexed


def read_rasterio_image(filename):
    with rasterio.open(filename) as src:
        # Lee los datos de la imagen como una matriz numpy
        data = src.read()
        # Transpone la matriz para que tenga la forma (bandas, filas, columnas)
        data = np.transpose(data, (1, 2, 0))
        # Crea una imagen PIL a partir de la matriz de datos
        if data.shape[2] == 1:
            data = data.squeeze()
        pil_image = Image.fromarray(data)
        return pil_image


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


def get_raster_bbox(raster_file):
    with rasterio.open(raster_file) as src:
        crs = src.crs.to_epsg()
        left, bottom, right, top = src.bounds
        return Polygon.from_bounds(left, bottom, right, top), crs


def clip_raster_with_polygon(shapely_geometry, geometry_crs, raster_path, output_path, output_srid=None):
    """
    Clips a raster with a Shapely geometry and saves the resulting image to a file.

    :param shapely_geometry: The geometry to clip the raster with.
    :type shapely_geometry: shapely.geometry

    :param raster_path: The path to the input raster file.
    :type raster_path: str

    :param output_path: The path to save the output raster file, including the .tiff extension.
    :type output_path: str

    :param output_srid: The CRS of the output raster file. Optional, defaults to the CRS of the input raster.
    :type output_srid: int, optional

    :return: None
    """
    # Open the raster file using Rasterio
    raster_csr = None
    with rasterio.open(raster_path) as src:
        # project shapely polygon to raster crs if needed
        raster_csr = src.crs.to_epsg()
        if raster_csr != geometry_crs:
            transformer = Transformer.from_crs(geometry_crs, src.crs.to_epsg(), always_xy=True)
            shapely_geometry = shapely_transform(transformer.transform, shapely_geometry)
            min_x, min_y, max_x, max_y = shapely_geometry.bounds
            rect = Polygon([(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)])

        # Convert the Shapely geometry to a GeoJSON-like dict using Shapely's `mapping()` function
        geojson = mapping(rect)

        # Use Rasterio's `mask()` function to clip the raster with the Shapely geometry
        out_image, out_transform = mask(src, [geojson], crop=True)

        # Copy the metadata from the original raster
        out_meta = src.meta.copy()

        # Update the metadata with new dimensions, transform, and CRS
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform,
                         "crs": src.crs})

        # Write the clipped raster to a new file using Rasterio's `write()` function
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

        # If output_srid is specified and different from the input raster CRS, reproject the clipped raster before saving
        if output_srid and raster_csr != output_srid:
            tmp_filename = output_path + "_tmp"
            reproject_raster(output_path, tmp_filename, output_srid)
            # rename
            os.remove(output_path)
            os.rename(tmp_filename, output_path)


def reproject_raster(input_raster_path, output_raster_path, dst_crs):
    # Read the input raster data
    with rasterio.open(input_raster_path) as src:
        # Calculate the default transform to reproject the raster
        transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)

        # Update the metadata with the new CRS, transform, width, and height
        out_meta = src.meta.copy()
        out_meta.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        # Read the input raster data into a numpy array
        src_array = src.read()

        # Create an empty numpy array with the same shape as the source array to store the reprojected data
        dst_array = np.empty_like(src_array)

        # Reproject the raster data
        reproject(src_array, dst_array, src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform,dst_crs=dst_crs,
            resampling=Resampling.nearest
        )

        # Write the reprojected data to the output raster file
        with rasterio.open(output_raster_path, 'w', **out_meta) as dst:
            dst.write(dst_array)
