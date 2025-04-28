import cv2
import numpy as np


def get_dlib_points(img, predictor, rectangle):
    """
    Extracts dlib key points from face image
    parameters
    ----------
    img : numpy.ndarray
        Grayscale face image
    predictor : dlib.shape_predictor
        shape predictor which is used to localize key points from face image
    rectangle : dlib.rectangle
        face bounding box inside image

    Returns
    -------
    numpy.ndarray
        dlib key points of the face inside rectangle.
    """

    shape = predictor(img, rectangle)
    dlib_points = np.zeros((68, 2))
    for i, part in enumerate(shape.parts()):
        dlib_points[i] = [part.x, part.y]
    return dlib_points


def get_left_key_points(key_points):
    """
    Extract dlib key points from left eye region including eye brow region.
    Parameters
    ----------
    key_points : numpy.ndarray
        Dlib face key points

    Returns:
    ----------
        dlib key points of left eye region
    """
    output = np.zeros((11, 2))
    output[0:5] = key_points[22:27]
    output[5:11] = key_points[42:48]
    return output

def get_right_key_points(key_points):
    """
    Extract dlib key points from right eye region including eye brow region.

    Parameters
    ----------
    key_points : numpy.ndarray
        Dlib face key points

    Returns:
    ----------
        dlib key points of right eye region
    """

    output = np.zeros((11, 2))
    output[0:5] = key_points[17:22]
    output[5:11] = key_points[36:42]
    return output


def get_attributes_wrt_local_frame(face_image, key_points_11, image_shape):
    """
    Extracts eye image, key points of the eye region with respect
    face eye image, angles and distances between centroid of key point of eye  and
    other key points of the eye.

    Parameters
    ----------
    face_image : numpy.ndarray
        Image of the face
    key_points_11 : numpy.ndarray
        Eleven key points of the eye including eyebrow region.
    image_shape : tuple
        Shape of the output eye image

    Returns
    -------
    eye_image : numpy.ndarray
        Image of the eye region
    key_points_11 : numpy.ndarray
        Eleven key points translated to eye image frame
    dists : numpy.ndarray
        Distances of each 11 key points from centeroid of all 11 key points
    angles : numpy.ndarray
        Angles between each 11 key points from centeroid

    """

    face_image_shape = face_image.shape
    top_left = key_points_11.min(axis=0)
    bottom_right = key_points_11.max(axis=0)

    # bound the coordinate system inside eye image
    bottom_right[0] = min(face_image_shape[1], bottom_right[0])
    bottom_right[1] = min(face_image_shape[0], bottom_right[1] + 5)
    top_left[0] = max(0, top_left[0])
    top_left[1] = max(0, top_left[1])

    # crop the eye
    top_left = top_left.astype(np.uint8)
    bottom_right = bottom_right.astype(np.uint8)
    eye_image = face_image[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]

    # translate the eye key points from face image frame to eye image frame
    key_points_11 = key_points_11 - top_left
    key_points_11 += np.finfo(float).eps

    # horizontal scale to resize image
    scale_h = image_shape[1] / float(eye_image.shape[1])

    # vertical scale to resize image
    scale_v = image_shape[0] / float(eye_image.shape[0])

    # resize left eye image to network input size
    eye_image = cv2.resize(eye_image, (image_shape[0], image_shape[1]))

    # scale left key points proportional with respect to left eye image resize scale
    scale = np.array([[scale_h, scale_v]])
    key_points_11 = key_points_11 * scale

    # calculate centroid of left eye key points
    centroid = np.array([key_points_11.mean(axis=0)])

    # calculate distances from  centroid to each left eye key points
    dists = __distance_between(key_points_11, centroid)

    # calculate angles between centroid point vector and left eye key points vectors
    angles = __angles_between(key_points_11, centroid)
    return eye_image, key_points_11, dists, angles


def __distance_between(v1, v2):
    """
    Calculates euclidean distance between two vectors.
    If one of the arguments is matrix then the output is calculated for each row
    of that matrix.

    Parameters
    ----------
    v1 : numpy.ndarray
        First vector
    v2 : numpy.ndarray
        Second vector

    Returns:
    --------
    numpy.ndarray
        Matrix if one of the arguments is matrix and vector if both arguments are vectors.
    """

    diff = v2 - v1
    diff_squared = np.square(diff)
    dist_squared = diff_squared.sum(axis=1)
    dists = np.sqrt(dist_squared)
    return dists


def __angles_between(v1, v2):
    """
    Calculates angle between two point vectors.

    Parameters
    ----------
    v1 : numpy.ndarray
        First vector
    v2 : numpy.ndarray
        Second vector

    Returns:
    --------
    numpy.ndarray
        Vector if one of the arguments is matrix and scalar if both arguments are vectors.
    """
    dot_prod = (v1 * v2).sum(axis=1)
    v1_norm = np.linalg.norm(v1, axis=1)
    v2_norm = np.linalg.norm(v2, axis=1)

    cosine_of_angle = (dot_prod / (v1_norm * v2_norm)).reshape(11, 1)

    angles = np.arccos(np.clip(cosine_of_angle, -1, 1))

    return angles
