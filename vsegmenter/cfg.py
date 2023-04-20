import logging
import os
from pathlib import Path


def resources(filename):
    return os.path.join(PROJECT_BASE, "resources", filename)


def results(filename):
    return os.path.join(PROJECT_BASE, "results", filename)


def project_file(filename):
    return os.path.join(PROJECT_BASE, filename)


def dataset(filename):
    dts_base = os.environ.get('DATASET_FOLDER', default="/media/gus/data/viticola/datasets/segmenter")
    return os.path.join(dts_base, filename)


def cartography(filename):
    carto_base = os.environ.get('CARTOGRAPHY_BASE_FOLDER', default=CARTOGRAPHY_BASE_FOLDER)
    return os.path.join(carto_base, filename)


def pnoa(filename):
    pnoa_base = os.environ.get('PNOA_BASE_FOLDER', default=PNOA_BASE_FOLDER)
    return os.path.join(pnoa_base, filename)


PROJECT_BASE = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute()
# '/media/cartografia/01_Ortofotografia/2020/RGB'
PNOA_BASE_FOLDER = os.environ.get('PNOA_BASE_FOLDER', default='/media/gus/data/rasters/aerial/pnoa/2020')
CARTOGRAPHY_BASE_FOLDER = os.environ.get('CARTOGRAPHY_BASE_FOLDER', default="/workspaces/cartography/")

LOG_CONFIGURATION = True


def configLog(level=logging.INFO):
    global LOG_CONFIGURATION
    if LOG_CONFIGURATION:
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)-1s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S",
                            handlers=[
                                logging.FileHandler("vineyard.log"),
                                logging.StreamHandler()
                            ])
        # logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
        LOG_CONFIGURATION = True
