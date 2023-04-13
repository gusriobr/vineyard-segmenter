"""
Post processing to create feature file out from raster images
"""
import logging
import os
from collections import OrderedDict

import fiona
import rasterio
import sys
from fiona.crs import from_epsg
from rasterio import features
from shapely.geometry import shape

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)

from vsegmenter import cfg
from vsegmenter.geo import proj

cfg.configLog()


def vectorize_predictions(raster_file, shape_file, filter_value=1, feature_filter=None):
    logging.info("Started image vectorization from raster {} into shapefile {}".format(raster_file, shape_file))
    polys = None
    raster_crs = None
    with rasterio.open(raster_file) as src:
        img = src.read()
        mask = img == filter_value
        polys = features.shapes(img, mask=mask, transform=src.transform)
        raster_crs = src.crs
    logging.info("Polygons extracted")

    # if file exist append otherwise create
    feature_crs = from_epsg(4258)  # ETRS89
    if os.path.exists(shape_file):
        open_f = fiona.open(shape_file, 'a')
    else:
        # output_driver = "GeoJSON"
        output_driver = 'ESRI Shapefile'
        vineyard_schema = {
            'geometry': 'Polygon',
            'properties': OrderedDict([('FID', 'int')])
        }
        open_f = fiona.open(shape_file, 'w', driver=output_driver, crs=feature_crs, schema=vineyard_schema)

    with open_f as c:
        for p in polys:
            if feature_filter and not feature_filter(p[0]):
                # se ha definido el filtro y devuelve False
                continue
            p = list(p)
            p[0] = proj.transform(raster_crs, feature_crs, p[0])
            poly_feature = {"geometry": p[0], "properties": {"FID": 0}}
            c.write(poly_feature)

    logging.info("Vectorization finished.")


def filter_by_area(min_area):
    def area_filter(polygon):
        return shape(polygon).area > min_area

    return area_filter


def filter_features(input_file, output_file):
    with fiona.open(input_file) as source:
        source_driver = source.driver
        source_crs = source.crs
        source_schema = source.schema
        polys_filtered = list(filter(filter_by_area, source))

    with fiona.open(output_file, "w", driver=source_driver, schema=source_schema, crs=source_crs) as dest:
        for r in polys_filtered:
            dest.write(r)


if __name__ == '__main__':
    iteration = 4

    input_folder = cfg.results(f"processed/v{iteration}")
    input_images = [os.path.join(input_folder, f_img) for f_img in os.listdir(input_folder) if f_img.endswith(".tif")]
    output_file = cfg.results(f"processed/v{iteration}/polygons_v{iteration}.shp")

    total = len(input_images)
    for i, f_image in enumerate(input_images):
        logging.info("Vectorizing image {} of {}".format(i + 1, total))
        vectorize_predictions(f_image, output_file, feature_filter=filter_by_area(min_area=450))

    # filtered_output_file = output_file.replace(".shp", "_filtered.shp")
    # logging.info(f"Filtering out small polygons into {filtered_output_file}")
    # filter_features(output_file, filtered_output_file)

    logging.info("Vectorized geometries successfully written.")
