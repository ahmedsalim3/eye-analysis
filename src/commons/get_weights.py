from src.commons import folder_utils
from src.commons.constant import ROOT_DIR


def download_shape_predictor():
    """Download the shape predictor file and return its path."""
    SHAPE_PREDICTOR_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    return folder_utils.download_weights(
            file_name="shape_predictor_68_face_landmarks.dat",
            source_url=SHAPE_PREDICTOR_URL,
            home_path=ROOT_DIR
        )
