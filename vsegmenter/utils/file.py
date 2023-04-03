import shutil
from pathlib import Path


def remake_folder(folder):
    """
    Delete previous folders and create
    :param output_folder:
    :return:
    """
    folder_path = Path(folder)
    shutil.rmtree(folder, ignore_errors=True)
    # create folders
    folder_path.mkdir(parents=True, exist_ok=True)