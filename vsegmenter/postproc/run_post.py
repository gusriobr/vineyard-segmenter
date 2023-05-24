"""
Post processing to create feature file out from raster images
"""
import argparse
import logging
import os
import time
from pathlib import Path

import fiona
import rasterio
from rasterio import features
from shapely.geometry import mapping
from shapely.geometry import shape
from tqdm import tqdm

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
    logging.info(f"Merging {len(polys)} overlapped vectorized polygons.")
    polys = geov.merge_polygons(polys, multiparts=False)

    logging.info(f"Projecting {len(polys)} from srid={raster_crs} into srid={db_file_srid}")
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
        polys = [p for p in source]

    # Simplificar los polígonos y almacenar en una lista
    with tqdm(total=len(polys), desc="Simplifying polygons") as pbar:
        polys_filtered = []
        for p in polys:
            poly_filtered = simplify_polygon(shape(p["geometry"]))
            polys_filtered.append(poly_filtered)
            pbar.update(1)

    logging.info(f"Saving simplified features into {output_file}")

    with fiona.open(output_file, "w", driver=source_driver, schema=source_schema, crs=source_crs) as dest:
        records = [{'geometry': mapping(r), 'properties': {}} for r in polys_filtered]
        dest.writerecords(records)


def post_process_images(image_files, output_file):
    total = len(image_files)
    for i, f_image in enumerate(image_files):
        logging.info("Vectorizing image {} of {}".format(i + 1, total))
        vectorize_predictions(f_image, output_file, feature_filter=filter_by_area(min_area=300), db_file_srid=25830)

    if not os.path.exists(output_file):
        logging.info(f"NO SQLITE FILE FOUND, check image post_processing")
        return

    filtered_output_file = output_file.replace(".sqlite", "_filtered.sqlite")
    logging.info(f"Simplifying polygons into {filtered_output_file}")
    simplify_features(output_file, filtered_output_file)
    logging.info("Vectorized geometries successfully written.")


def run_process(input_folder, output_file, interactive=False, wait_time=15, max_wait=15 * 60, process_existing=True):
    # Procesa las imágenes existentes en el directorio input_folder
    input_images = [os.path.join(input_folder, f_img) for f_img in os.listdir(input_folder) if
                    f_img.endswith(".tif")]
    if process_existing:
        logging.info(f"Running post-processing actions on existing images: {len(input_images)}")
        post_process_images(input_images, output_file)
    else:
        # add all existing files as processed
        logging.info(f"Avoiding processing existing images: {input_images}")

    # Si se ejecuta en modo interactivo, espera y procesa nuevas imágenes que vayan apareciendo
    if interactive:
        logging.info(f"Interactive loop, waiting for new images on : {input_folder}")
        processed_files = set(input_images)
        no_new_images_time = 0

        while no_new_images_time < max_wait:
            input_files = set(Path(input_folder).glob("*.tif"))

            # Encuentra nuevos archivos que aún no se han procesado
            new_files = input_files.difference(processed_files)
            if new_files:
                logging.info(f"New files found : {new_files}")
                post_process_images(new_files, output_file)
                processed_files.update(new_files)
                no_new_images_time = 0
            else:
                time.sleep(wait_time)
                no_new_images_time += wait_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running inference con vsegmentation models")
    parser.add_argument("version", type=int, help="Model version")
    parser.add_argument("--interactive", help="Run in interactive mode", default=False)
    parser.add_argument("--process_existing", help="Wether to process already existing files or just new", default=True)
    args = parser.parse_args()

    version = args.version
    interactive = args.interactive
    # version = 5
    # interactive = True
    logging.info(f"Processing existing images:  {args.process_existing}")

    input_folder = cfg.results(f"processed/v{version}_ribera")
    output_file = cfg.results(f"processed/v{version}_ribera/polygons_v{version}.sqlite")

    run_process(input_folder, output_file, interactive=interactive, process_existing=args.process_existing)

    logging.info("Vectorized geometries successfully written.")
