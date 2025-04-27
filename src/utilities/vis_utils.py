import cv2
import dlib
import numpy as np

from src.utilities.data_utils import get_dlib_points, get_left_key_points, get_right_key_points
from src.commons import logger

logger = logger.Logger()

def debug_eye_extraction(image_path, eye_coords, predictor, features):
    """
    Debug visualization for eye feature extraction.
    """
    orig_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
    
    x1, y1, x2, y2 = eye_coords
    
    # Create a rectangle around the eye region with padding
    rect = dlib.rectangle(
        left=max(0, x1-10), 
        top=max(0, y1-10), 
        right=min(gray_image.shape[1], x2+10), 
        bottom=min(gray_image.shape[0], y2+10)
    )
    
    # Create debug image with eye region rectangle
    debug_img = orig_image.copy()
    cv2.rectangle(debug_img, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)
    cv2.circle(debug_img, (x1, y1), radius=4, color=(255, 0, 0), thickness=-1)
    cv2.circle(debug_img, (x2, y2), radius=4, color=(255, 0, 0), thickness=-1)
    cv2.imshow("Eye Region", debug_img)
    if cv2.waitKey(0) & 0xFF == ord('q'): 
        cv2.destroyWindow("Eye Region")
    
    try:
        # Extract and visualize landmarks
        landmarks = get_dlib_points(gray_image, predictor, rect)
        
        landmarks_img = orig_image.copy()
        for i, point in enumerate(landmarks):
            cv2.circle(landmarks_img, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
        cv2.imshow("Landmarks", landmarks_img)
        if cv2.waitKey(0) & 0xFF == ord('q'): 
            cv2.destroyWindow("Landmarks")
        
        # Extract and visualize eye key points
        left_key_points = get_left_key_points(landmarks)
        right_key_points = get_right_key_points(landmarks)
        
        eye_points_img = orig_image.copy()
        for points, color in [(left_key_points, (0, 255, 0)), (right_key_points, (0, 255, 0))]:
            for point in points:
                cv2.circle(eye_points_img, (int(point[0]), int(point[1])), 3, color, -1)
        cv2.imshow("Eye Key Points", eye_points_img)
        if cv2.waitKey(0) & 0xFF == ord('q'): 
            cv2.destroyWindow("Eye Key Points")
        
        # Extract and visualize eye features
        if features:
            # Left eye visualization
            left_eye = features['left'][0] * 255
            left_eye = left_eye.astype(np.uint8).squeeze()
            cv2.imshow("Left Eye Features", left_eye)
            cv2.circle(orig_image, (x1, y1), radius=4, color=(255, 0, 0), thickness=-1)
            if cv2.waitKey(0) & 0xFF == ord('q'): 
                cv2.destroyWindow("Left Eye Features")

            # Right eye visualization
            right_eye = features['right'][0] * 255
            right_eye = right_eye.astype(np.uint8).squeeze()
            cv2.imshow("Right Eye Features", right_eye)
            cv2.circle(orig_image, (x2, y2), radius=4, color=(0, 0, 255), thickness=-1)
            if cv2.waitKey(0) & 0xFF == ord('q'): 
                cv2.destroyWindow("Right Eye Features")
            logger.debug("Debug visualizations displayed.")
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            return {
                'original': debug_img,
                'landmarks': landmarks_img,
                'eye_points': eye_points_img,
                'left_eye': left_eye,
                'right_eye': right_eye
            }

    except Exception as e:
        logger.debug(f"Error in _debug_eye_extraction: {e}")
        cv2.destroyAllWindows()
    
    return None
