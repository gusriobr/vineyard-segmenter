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


def extract_images(src_file, mask_file, dst_folder, num_polygons=10, rect_size=(256, 256), delete_folder=True):
    from rasterio.windows import Window
    if delete_folder:
        remake_folder(dst_folder)

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

        # read mask data and check at least echa category is represented by a % of pixes
        mask_data = mask.read(window=window)
        mask_data = np.squeeze(mask_data)
        if not check_categories(mask_data, threshold=0.3):
            continue
        data = src.read(window=window)

        # save images
        image = Image.fromarray(np.moveaxis(data, 0, 2), 'RGB')
        image.save(f"{dst_folder}/{i}_data.jpg")
        img_mask = Image.fromarray(mask_data)
        img_mask.save(f"{dst_folder}/{i}_mask.jpg")
        i += 1


def check_categories(mask_data, threshold=0.3):
    # mask must have at least <threshold>% of each category
    total_pxls = mask_data.shape[0] * mask_data.shape[1]
    perc_zeros = np.count_nonzero(mask_data == 0) / total_pxls
    perc_ones = np.count_nonzero(mask_data == 255) / total_pxls
    return perc_zeros >= threshold and perc_ones >= threshold


if __name__ == '__main__':
    samples_file = '/media/gus/workspace/wml/vineyard-segmenter/resources/dataset.sqlite'
    raster_sample = '/media/gus/workspace/wml/vineyard-segmenter/resources/PNOA_CYL_2020_25cm_OF_etrsc_rgb_hu30_h05_0373_7-2_extraction.tiff'
    mask_file = cfg.resource("dataset/salida.tiff")
    burn_samples(samples_file, raster_sample, mask_file)

    output_folder = cfg.dataset("samples")
    rect_size = (256, 256)
    extract_images(raster_sample, mask_file, output_folder, rect_size=rect_size, num_polygons=500)
