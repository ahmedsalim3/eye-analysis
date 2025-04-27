import os
import bz2
import patoolib
import gdown

from src.commons.logger import Logger

logger = Logger()

def initialize_folder(home_path: str) -> None:
    """Create required directories if they don't exist."""
    paths = [
        os.path.join(home_path, "data", "raw"),
        os.path.join(home_path, "weights")
    ]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            logger.info(f"Directory {path} created")

def _download_file(source_url: str, output_path: str) -> None:
    """Download helper function with Google Drive support."""
    if "drive.google.com" in source_url or "google.com" in source_url:
        file_id = source_url.split('id=')[1].split('&')[0]
        gdown.download(id=file_id, output=output_path, quiet=False)
    else:
        gdown.download(source_url, output_path, quiet=False)

def _verify_not_html(file_path: str) -> None:
    """Check if downloaded file is not HTML."""
    with open(file_path, 'rb') as f:
        if b'<html' in f.read(1024).lower():
            os.remove(file_path)
            raise ValueError("Downloaded file is HTML, not a valid archive")

def download_dataset(file_name: str, source_url: str, home_path: str = ".") -> str:
    """Download and extract dataset file using patoolib."""
    target_dir = os.path.normpath(os.path.join(home_path, "data", "raw"))
    archive_path = os.path.join(target_dir, file_name)
    
    # Remove .rar extension if present in file_name to get target directory name
    base_name = file_name.replace('.rar', '')
    final_target = os.path.join(target_dir, base_name)
    
    if os.path.exists(final_target):
        logger.debug(f"Dataset already exists at {final_target}")
        return final_target

    try:
        logger.info(f"Downloading dataset {file_name} from {source_url}")
        _download_file(source_url, archive_path)
        _verify_not_html(archive_path)

        logger.info(f"Extracting archive to {target_dir}")
        patoolib.extract_archive(archive_path, outdir=target_dir)
        os.remove(archive_path)
        
        return final_target

    except Exception as err:
        if os.path.exists(archive_path):
            os.remove(archive_path)
        raise ValueError(f"Failed to download and extract dataset {file_name}") from err

def download_weights(file_name: str, source_url: str, home_path: str = ".") -> str:
    """Download and extract BZ2 weights file."""
    target_file = os.path.normpath(os.path.join(home_path, "weights", file_name))
    output_bz2 = f"{target_file}.bz2"

    if os.path.isfile(target_file):
        logger.debug(f"{file_name} already exists at {target_file}")
        return target_file

    try:
        logger.info(f"Downloading weights {file_name} from {source_url}")
        _download_file(source_url, output_bz2)
        _verify_not_html(output_bz2)

        with bz2.BZ2File(output_bz2) as bz2file:
            with open(target_file, "wb") as f:
                f.write(bz2file.read())
        os.remove(output_bz2)

    except Exception as err:
        if os.path.exists(output_bz2):
            os.remove(output_bz2)
        raise ValueError(f"Failed to download weights {file_name}") from err

    return target_file
