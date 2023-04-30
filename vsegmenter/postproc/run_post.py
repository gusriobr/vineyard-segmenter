"""
Post processing to create feature file out from raster images
"""
import argparse
import logging
import os

import fiona
import rasterio
from rasterio import features
from shapely.geometry import mapping
from shapely.geometry import shape

import geo.spatialite as spt
import geo.vectors as geov
from geo import proj
from vsegmenter import cfg

# ROOT_DIR = os.path.abspath("../../")
# sys.path.append(ROOT_DIR)

cfg.configLog()


def simplify_polygon(shapely_polygon, simply_threshold1=1.5, simply_threshold2=3, buffer_distance=1,
                     simply_threshold3=0.5):
    simplified_polygon = shapely_polygon.simplify(tolerance=simply_threshold1)
    simplified_polygon = simplified_polygon.simplify(tolerance=simply_threshold2)
    simplified_polygon = simplified_polygon.buffer(distance=buffer_distance)
    simplified_polygon = simplified_polygon.simplify(tolerance=simply_threshold3)
    return simplified_polygon


def vectorize_predictions(raster_file, db_file, filter_value=1, feature_filter=None, db_file_srid=4258):
    logging.info("Started image vectorization from raster {} into Spatialite {}".format(raster_file, db_file))

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
    if len(polys) == 0:
        logging.info(f"No polygons to process, skipping")
        return

    polys = geov.remove_interior_rings(polys)

    # create extent for current polygons
    poly_ext = geov.get_extent(polys)

    layer_name = VINEYARD_LAYER["name"]
    geometry_column = "geom"
    if not os.path.exists(db_file):
        spt.create_spatialite_table(db_file, layer_name, VINEYARD_LAYER["sql"], geometry_col=geometry_column,
                                    srid=db_file_srid)
    # find polygons stored in the same area as current polygons
    logging.info(f"Retrieving exising polygons in {db_file}")
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
                  "geometry_column": "geom",
                  "sql": """
        CREATE TABLE IF NOT EXISTS vineyard (
            id INTEGER PRIMARY KEY
        )
    """}


def filter_by_area(min_area):
    def area_filter(polygon):
        return shape(polygon).area > min_area

    return area_filter


def simplify_features(input_file, output_file):
    with fiona.open(input_file) as source:
        source_driver = source.driver
        source_crs = source.crs
        source_schema = source.schema
        polys_filtered = [simplify_polygon(shape(p["geometry"])) for p in source]

    with fiona.open(output_file, "w", driver=source_driver, schema=source_schema, crs=source_crs) as dest:
        for r in polys_filtered:
            feature = {'geometry': mapping(r), 'properties': {}}
            dest.write(feature)


if __name__ == '__main__':

    ### test filtering
    parser = argparse.ArgumentParser(description="Running inference con vsegmentation models")
    parser.add_argument("version", type=int, help="Model version")
    args = parser.parse_args()

    version = args.version

    input_folder = cfg.results(f"processed/v{version}")
    input_images = [os.path.join(input_folder, f_img) for f_img in os.listdir(input_folder) if f_img.endswith(".tif")]
    # output_file = cfg.results(f"processed/v{iteration}/polygons_v{iteration}.shp")
    output_file = cfg.results(f"processed/v{version}/polygons_v{version}.sqlite")

    total = len(input_images)
    for i, f_image in enumerate(input_images):
        logging.info("Vectorizing image {} of {}".format(i + 1, total))
        vectorize_predictions(f_image, output_file, feature_filter=filter_by_area(min_area=300), db_file_srid=25830)

    filtered_output_file = output_file.replace(".sqlite", "_filtered.sqlite")
    logging.info(f"Simplifying polygons into {filtered_output_file}")
    simplify_features(output_file, filtered_output_file)

    logging.info("Vectorized geometries successfully written.")


