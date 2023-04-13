"""
- Extract polygons
"""
import os
import random

import geopandas as gpd
import numpy as np
import rasterio
import spatialite
from PIL import Image
from rasterio import features
from rasterio.features import geometry_mask
import tensorflow as tf
import numpy as np
import os

# Must used recent master (pypi version appears broken as of 8/18/2014)
# pip install "git+https://github.com/lokkju/pyspatialite.git@94a4522e58#pyspatialite"
import cfg
from utils.file import remake_folder
import logging

cfg.configLog()


def burn_features(raster_file, geometries, output_folder):
    """

    :param raster_file:
    :param geometries:
    :param output_folder:
    :return:
    """
    # get raster metadata from source
    source = rasterio.open(raster_file)

    meta = source.meta.copy()
    meta.update(compress='lzw', count=1)  # just one band as output
    source.close()

    output_f = os.path.join(output_folder, "burned.tif")
    # burn features on the raster and create a mask image
    with rasterio.open(output_f, 'w+', **meta) as out:
        shapes = ((geom, 255) for geom in geometries)
        burned = features.rasterize(shapes=shapes, fill=0, out_shape=out.shape, transform=out.transform)
        out.write_band(1, burned)
    return output_f


def get_samples_gp(db_file, layer_name):
    with spatialite.connect(db_file) as con:
        sql = f"SELECT Hex(ST_AsBinary(GEOMETRY)) as geometry FROM {layer_name}"
        df = gpd.GeoDataFrame.from_postgis(sql, con, geom_col="geometry")
    return [geom for geom in df.geometry]


def get_samples(db_file, layer_name):
    shapes = []
    with spatialite.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT geometry FROM {layer_name}")
        rows = cursor.fetchall()
        # Crear una lista de geometrías de los polígonos
        for row in rows:
            geom = row[1]
            shapes.append(geom)

    return shapes


def burn_samples(samples_file, input_file, output_file):
    # Nombre del campo que contiene el identificador único de cada polígono

    # Abrir el archivo GeoTIFF de entrada
    with rasterio.open(input_file) as src:
        # Leer los metadatos del archivo
        profile = src.profile
        # Cargar los polígonos de la capa especificada en el archivo Spatialite
        shapes = get_samples_gp(samples_file, 'samples')

        # Crear una máscara de los polígonos en el GeoTIFF
        mask = geometry_mask(shapes, out_shape=src.shape, transform=src.transform, invert=True)
        # Crear un arreglo de ceros con las mismas dimensiones que el GeoTIFF
        data = mask.astype("uint8") * 255

        # Guardar el arreglo como un nuevo GeoTIFF
        profile.update(count=1, dtype="uint8", compress="lzw")
        with rasterio.open(output_file, "w", **profile) as dst:
            dst.write(data, 1)


def extract_images(src_file, file_id, mask_file, dst_folder, samples_per_category, rect_size=(256, 256),
                   delete_folder=True,
                   mixed_threshold=0.3):
    """
    :param src_file:
    :param mask_file:
    :param dst_folder:
    :param num_polygons:
    :param rect_size:
    :param delete_folder:
    :param per_category_threshold: min number of pixels of each category that must contain the image
    :param one_category_samples: dict indicating for each category the number of images to create that has pixels just
    for that category (all 0 or all 1)
    :return:
    """
    if delete_folder:
        remake_folder(dst_folder)

    init_counter = 0
    if "mixed" in samples_per_category:
        num_polygons = samples_per_category["mixed"]
        logging.info(f"Extracting {num_polygons} images with category representation threshold = {mixed_threshold}")

        def f_threshold(mask_data):
            return check_categories(mask_data, zero_threshold=mixed_threshold, one_threshold=mixed_threshold)

        get_random_images(dst_folder, f_threshold, mask_file, file_id, init_counter, num_polygons, rect_size, src_file)
        init_counter += num_polygons

    if "0" in samples_per_category:
        num_polygons = samples_per_category["0"]
        logging.info(f"Extracting {num_polygons} images for category 0")

        def f_category0(mask_data):
            """
            Just zero category images
            :param mask_data:
            :return:
            """
            return check_categories(mask_data, zero_threshold=1)

        get_random_images(dst_folder, f_category0, mask_file, file_id, init_counter, num_polygons, rect_size, src_file)
        init_counter += num_polygons

    if "1" in samples_per_category:
        num_polygons = samples_per_category["1"]
        logging.info(f"Extracting {num_polygons} images for category 1")

        def f_category1(mask_data):
            """
            Just One category images
            :param mask_data:
            :return:
            """
            return check_categories(mask_data, one_threshold=1)

        get_random_images(dst_folder, f_category1, mask_file, file_id, init_counter, num_polygons, rect_size, src_file)
        init_counter += num_polygons
    logging.info(f"Extracted {init_counter} images")


def get_random_images(dst_folder, image_filter, mask_file, prefix, init_counter, num_polygons, rect_size, src_file):
    """

    :param dst_folder:
    :param image_filter:
    :param mask_file:
    :param num_polygons:
    :param rect_size:
    :param src_file:
    :return:
    """
    from rasterio.windows import Window

    # Abrir el archivo GeoTIFF
    src = rasterio.open(src_file)
    mask = rasterio.open(mask_file)
    # Obtener el tamaño de la imagen
    width = src.width
    height = src.height
    # Iterar sobre el número de rectángulos
    i = 0
    while i < num_polygons:
        # Seleccionar una ubicación aleatoria para el rectángulo
        x = random.randint(0, width - rect_size[0])
        y = random.randint(0, height - rect_size[1])

        # Crear una ventana de Rasterio que representa el rectángulo
        window = Window(x, y, rect_size[0], rect_size[1])

        # read mask data and check at least each category is represented by a % of pixes
        mask_data = mask.read(window=window)
        mask_data = np.squeeze(mask_data)
        if not image_filter(mask_data):
            continue
        # logging.info(f"Got {i}")
        data = src.read(window=window)

        # save images
        image = Image.fromarray(np.moveaxis(data, 0, 2), 'RGB')
        image.save(f"{dst_folder}/{prefix}_{init_counter + i}_data.jpg")
        img_mask = Image.fromarray(mask_data)
        img_mask.save(f"{dst_folder}/{prefix}_{init_counter + i}_mask.jpg")
        i += 1


def check_categories(mask_data, zero_threshold=None, one_threshold=None):
    # mask must have at least <threshold>% of each category
    total_pxls = mask_data.shape[0] * mask_data.shape[1]
    perc_zeros = np.count_nonzero(mask_data == 0) / total_pxls
    perc_ones = np.count_nonzero(mask_data == 255) / total_pxls
    if zero_threshold is not None and one_threshold is None:
        return perc_zeros >= zero_threshold
    if one_threshold is not None and zero_threshold is None:
        return perc_ones >= one_threshold
    return perc_zeros >= zero_threshold and perc_ones >= one_threshold


def run_extraction(raster_extractions, samples_file, dts_folder, rect_size, remake_folders=True):
    logging.info(f"Staring dataset extraction on folder: {dts_folder}")
    output_mask_folder = dts_folder + "/masks"
    samples_folder = dts_folder + "/samples"
    # re-create folders
    remake_folder(output_mask_folder, remake_folders)
    remake_folder(samples_folder, remake_folders)

    for raster_id, num_samples, raster in raster_extractions:
        logging.info(f"Processing file {raster} to extract samples: {num_samples}")
        output_mask_file = os.path.join(output_mask_folder, os.path.basename(raster.replace(".tiff", "_mask.tiff")))
        burn_samples(samples_file, raster, output_mask_file)
        logging.info(f"Image mask created from polygons")

        logging.info(f"Extracted samples from mask")
        extract_images(raster, raster_id, output_mask_file, samples_folder, rect_size=rect_size,
                       samples_per_category=num_samples,
                       mixed_threshold=0.3, delete_folder=remake_folders,
                       )
    logging.info("Image extraction finished successfully!")


"""
determines the number of samples to get from each raster representing each category, 
"mixed" means both categories must be present in the image
"""
DEFAULT_NUM_SAMPLES = {"0": 300, "1": 100, "mixed": 600}

if __name__ == '__main__':
    samples_file = cfg.resources('dataset/samples.sqlite')
    sample_size = (256, 256)

    version = "v1"
    dts_folder = f'/media/gus/data/viticola/datasets/segmenter/{version}'
    raster_base_folder = dts_folder + "/extractions"
    raster_samples = [
        ["0373_7-2", DEFAULT_NUM_SAMPLES,
         os.path.join(raster_base_folder, 'PNOA_CYL_2020_25cm_OF_etrsc_rgb_hu30_h05_0373_7-2_extraction.tiff')]
    ]

    run_extraction(raster_samples, samples_file, dts_folder, sample_size)
