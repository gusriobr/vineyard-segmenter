import os.path
import shutil
from pathlib import Path


def remake_folder(folder, delete_if_exist=True):
    """
    Delete previous folders and create
    :param output_folder:
    :return:
    """
    folder_path = Path(folder)
    if delete_if_exist:
        if os.path.exists(folder_path):
            shutil.rmtree(folder, ignore_errors=True)
        # create folders
        folder_path.mkdir(parents=True, exist_ok=True)

def filesize_in_MB(file_path):
    tamano_bytes = os.path.getsize(file_path)
    tamano_mb = tamano_bytes / (1024 * 1024)
    return tamano_mb
