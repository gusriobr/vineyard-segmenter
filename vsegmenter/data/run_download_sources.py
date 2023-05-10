"""
Give a spatialite database with the samples, and the vectorial database of tiles, looks for the tiles needed
to create the dataset and donwloads the files into the raster folder.

"""
import logging
import os.path

import cfg
from data.extractions import update_extraction_info
from vsegmenter.image.pnoa import download_pnoa_tile, pnoa_storage_path, create_pnoa_list_by_layer

cfg.configLog()

sample_file = cfg.resources("dataset/samples.sqlite")
pnoa_index = cfg.pnoa("pnoa_index.sqlite")
pnoa_folder = cfg.PNOA_BASE_FOLDER

logging.info(f"Intersecting extraction layer with PNOA index to find needed raster files")
pnoa_tiles = create_pnoa_list_by_layer(pnoa_index, sample_file, 'extractions', 'geometry')
logging.info(f"Found {len(pnoa_tiles)} intersecting PNOA tiles {pnoa_tiles}")

pnoa_tiles = [tile for tile in pnoa_tiles if not os.path.exists(pnoa_storage_path(tile, pnoa_folder))]
logging.info(f"{len(pnoa_tiles)} tiles not found in PNOA_FOLDER  {pnoa_folder}")
if not pnoa_tiles:
    logging.info("No tiles to download.")
else:
    logging.info("Starting download.")
    for idx, img_file in enumerate(pnoa_tiles):
        pnoa_file = pnoa_storage_path(img_file, pnoa_folder)
        logging.info(f"Downloading {idx + 1} of {len(pnoa_tiles)} images")
        download_pnoa_tile(img_file, pnoa_folder)

logging.info("PNOA tiles successfully downloaded.")
