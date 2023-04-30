"""
- Extract polygons
"""
import logging
import os
import random

import geopandas as gpd
import numpy as np
import rasterio
import spatialite
from PIL import Image
from rasterio import features
from rasterio.features import geometry_mask
from shapely.geometry import box
from shapely.ops import transform
from shapely.affinity import rotate

# Must used recent master (pypi version appears broken as of 8/18/2014)
# pip install "git+https://github.com/lokkju/pyspatialite.git@94a4522e58#pyspatialite"
import cfg
from vsegmenter.image.raster import read_rasterio_image
from utils.file import remake_folder

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


def get_num_autosampling(width, height, rect_size):
    cols = width // rect_size[0]
    rows = height // rect_size[1]
    return cols * rows + (cols - 1) * (rows - 1)


def extract_images(src_file, file_id, mask_file, dst_folder, samples_per_category, rect_size=(256, 256),
                   delete_folder=True, mixed_threshold=0.3, auto_mixed=True):
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

    src_image = read_rasterio_image(src_file)
    mask_image = read_rasterio_image(mask_file)
    height = src_image.height
    width = src_image.width
    if rect_size[0] > width or rect_size[1] > height:
        raise ValueError(
            f"Invalid extraction size. Minimum expected size: {rect_size} got: {(width, height)} in file {src_file}")

    init_counter = 0
    num_polygons = samples_per_category.get("0", 0)
    if num_polygons > 0:
        logging.info(f"Extracting {num_polygons} images for category 0")

        def f_category0(mask_data):
            """
            Just zero category images
            :param mask_data:
            :return:
            """
            return check_categories(mask_data, zero_threshold=1)

        get_random_samples(dst_folder, f_category0, src_image, mask_image, file_id, init_counter, num_polygons,
                           rect_size)
        init_counter += num_polygons

    num_polygons = samples_per_category.get("1", 0)
    if num_polygons > 0:
        logging.info(f"Extracting {num_polygons} images for category 1")

        def f_category1(mask_data):
            """
            Just One category images
            :param mask_data:
            :return:
            """
            return check_categories(mask_data, one_threshold=1)

        get_random_samples(dst_folder, f_category1, src_image, mask_image, file_id, init_counter, num_polygons,
                           rect_size)
        init_counter += num_polygons

    num_polygons = samples_per_category.get("mixed", 0)
    if num_polygons > 0:
        if auto_mixed:
            num_polygons = get_num_autosampling(width, height, rect_size)
        logging.info(f"Extracting {num_polygons} images with category representation threshold = {mixed_threshold}")

        def f_threshold(mask_data):
            return check_categories(mask_data, zero_threshold=mixed_threshold, one_threshold=mixed_threshold)

        get_random_samples(dst_folder, f_threshold, src_image, mask_image, file_id, init_counter, num_polygons,
                           rect_size)
        init_counter += num_polygons

    logging.info(f"Extracted {init_counter} images")


def create_rotated_rectangle(x, y, width, height, angle):
    rectangle = box(x, y, x + width, y + height)
    return rotate(rectangle, angle, origin=(x + width / 2, y + height / 2))


def get_random_images2(dst_folder, image_filter, mask_file, prefix, init_counter, num_polygons, rect_size, src_file,
                       max_iterations=100_00):
    """
    Get ramdon images from dastaset folder
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
    num_iterations = 0
    while i < num_polygons:
        # Seleccionar una ubicación aleatoria para el rectángulo
        x = random.randint(0, width - rect_size[0])
        y = random.randint(0, height - rect_size[1])

        # Crear una ventana de Rasterio que representa el rectángulo
        window = Window(x, y, rect_size[0], rect_size[1])

        # read mask data and check at least each category is represented by a % of pixes
        mask_data = mask.read(window=window)
        mask_data = np.squeeze(mask_data)
        num_iterations += 1
        if num_iterations > max_iterations:
            raise ValueError(
                f"Couldn't extract enough images after {max_iterations} iterations found {i}. Check the image mask: {mask_file}")
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


def get_random_samples(dst_folder, image_filter, original_src, original_mask, prefix, init_counter, num_polygons,
                       rect_size,
                       max_iterations=10_000, rotate=True, rotate_freq=0.5):
    """
    Get ramdon images from dastaset folder
    :param dst_folder:
    :param image_filter:
    :param mask_file:
    :param num_polygons:
    :param rect_size:
    :param src_file:
    :return:
    """

    # Iterar sobre el número de rectángulos
    i = 0
    num_iterations = 0
    fill_color = 125
    while i < num_polygons:
        if rotate and random.random() <= rotate_freq:
            angle = random.randint(0, 360)
            src = original_src.rotate(angle=angle)
            mask = original_mask.rotate(angle=angle, fillcolor=fill_color)
            has_rotated = True
        else:
            src = original_src
            mask = original_mask
            has_rotated = False

        height = src.height
        width = src.width

        # Seleccionar una ubicación aleatoria para el rectángulo
        x = random.randint(0, width - rect_size[0])
        y = random.randint(0, height - rect_size[1])

        # read mask data and check at least each category is represented by a % of pixes
        image_rect = src.crop((x, y, x + rect_size[0], y + rect_size[1]))
        mask_rect = mask.crop((x, y, x + rect_size[0], y + rect_size[1]))

        num_iterations += 1
        if num_iterations > max_iterations:
            raise ValueError(
                f"Couldn't extract enough images after {max_iterations} iterations found {i}. Check the image mask: {mask_file}")
        mask_array = np.array(mask_rect)
        if has_rotated and np.count_nonzero(mask_array[mask_array == fill_color]) > 0:
            # when the image has rotated some of the info is lost, this portions of the image
            # are mark as "fill-color". Make sure non of these pixels is used as sample
            continue
        if not image_filter(mask_array):
            continue

        # save images
        image_rect.save(f"{dst_folder}/{prefix}_{init_counter + i}_data.jpg")
        mask_rect.save(f"{dst_folder}/{prefix}_{init_counter + i}_mask.jpg")
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
    logging.info(f"Starting dataset extraction on folder: {dts_folder}")
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
