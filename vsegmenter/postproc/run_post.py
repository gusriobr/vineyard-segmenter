"""
Post processing to create feature file out from raster images
"""
import logging
import os
import sys

import fiona
import rasterio
from rasterio import features
from shapely.geometry import shape
import pyproj

import geo.spatialite as spt
import geo.vectors as geov
from geo import proj

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)

from vsegmenter import cfg

cfg.configLog()


def vectorize_predictions(raster_file, db_file, filter_value=1, feature_filter=None, db_file_srid=4258):
    logging.info("Started image vectorization from raster {} into shapefile {}".format(raster_file, db_file))

    with rasterio.open(raster_file) as src:
        img = src.read()
        mask = img == filter_value
        polys = features.shapes(img, mask=mask, transform=src.transform)
        raster_crs = src.crs

    # transform polygons to db srid
    logging.info(f"Raster file read.")
    polys = [shape(p[0]) for p in polys]
    logging.info(f"Merging overlapping vectorized {len(polys)} polygons.")
    polys = geov.merge_polygons(polys, multiparts=False)

    logging.info(f"Projecting {len(polys)} from srid={raster_crs} and into srid={db_file_srid}")
    polys = [shape(proj.transform(raster_crs, db_file_srid, p)) for p in polys]

    if feature_filter:
        polys = [p for p in polys if feature_filter(p)]
    logging.info(f"Polygons after filtering {len(polys)}")
    polys = geov.remove_interior_rings(polys)

    # create extent for current polygons
    poly_ext = geov.get_extent(polys)

    layer_name = VINEYARD_LAYER["name"]
    geometry_column = "geom"
    if not os.path.exists(db_file):
        spt.create_spatialite_table(db_file, layer_name, VINEYARD_LAYER["sql"], geomtry_col=geometry_column,
                                    srid=db_file_srid)
    # find polygons stores in the same area as current polygons
    polys_found = spt.list_by_geometry(db_file, layer_name, geometry_column, poly_ext, db_file_srid)

    # merge all polygons
    logging.info(f"Merging {len(polys_found)} polygons found in vineyard layer.")
    polys.extend(polys_found)
    polys = geov.merge_polygons(polys, multiparts=False)
    logging.info(f"Final merged polygons: {len(polys)}.")

    # remove existing polygons in the extent and insert new ones
    logging.info("Removing preexisting polygons.")
    spt.remove_by_geometry(db_file, layer_name, geometry_column, poly_ext, db_file_srid)
    logging.info(f"Inserting final {len(polys)} polygons.")
    spt.insert_polygons(db_file, layer_name, geometry_column, polys, srid=db_file_srid)

    logging.info("Vectorization finished.")


VINEYARD_LAYER = {"name": "vineyard",
                  "sql": """
        CREATE TABLE IF NOT EXISTS vineyard (
            id INTEGER PRIMARY KEY
        )
    """}


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

    ### test filtering

    iteration = 4

    input_folder = cfg.results(f"processed/v{iteration}")
    input_images = [os.path.join(input_folder, f_img) for f_img in os.listdir(input_folder) if f_img.endswith(".tif")]
    # output_file = cfg.results(f"processed/v{iteration}/polygons_v{iteration}.shp")
    output_file = cfg.results(f"processed/v{iteration}/polygons_v{iteration}.sqlite")

    total = len(input_images)
    for i, f_image in enumerate(input_images):
        logging.info("Vectorizing image {} of {}".format(i + 1, total))
        vectorize_predictions(f_image, output_file, feature_filter=filter_by_area(min_area=450), db_file_srid=25830)

    # filtered_output_file = output_file.replace(".shp", "_filtered.shp")
    # logging.info(f"Filtering out small polygons into {filtered_output_file}")
    # filter_features(output_file, filtered_output_file)

    logging.info("Vectorized geometries successfully written.")
