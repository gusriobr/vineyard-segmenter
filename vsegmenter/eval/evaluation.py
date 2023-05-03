import os
import logging

import cfg
from inference.run_inf import predict_on_raster
from postproc.run_post import post_process_images


def evaluate_on_dts(model, tag, dts, version, output_folder):
    # runs model on dataset extractions to evaluation the modelo on dataset
    raster_files = dts.get_extration_files()
    total = len(raster_files)
    output_raster_list = []
    for idx, r_file in enumerate(raster_files):
        logging.info("Processing image {} of {} - {}".format(idx + 1, total, r_file))
        filename = os.path.basename(r_file)
        base, ext = os.path.splitext(filename)
        output_file = os.path.join(output_folder, "{}_{}{}".format(base, tag, ext))
        try:
            predict_on_raster(r_file, model, output_file)
            output_raster_list.append(output_file)
        except:
            logging.error(f"Error while predicting on raster {r_file}")

    logging.info(f"Prediction on dataset finished. Starting post-processing")
    output_db_file = cfg.results(f"processed/v{version}/polygons_v{version}.sqlite")
    post_process_images(output_raster_list, output_db_file)
    logging.info(f"Evaluation finished output_folder: {output_folder}")
