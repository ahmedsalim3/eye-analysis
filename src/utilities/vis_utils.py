import os
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utilities.data_utils import get_dlib_points, get_left_key_points, get_right_key_points
from src.commons.logger import Logger
from src.commons import constant

logger = Logger()

def render(frame, results, draw_landmarks=False, detector_type="retinaface"):
    if len(results) == 0:
        return frame

    state_colors = {
        'open': (0, 255, 0),
        'closed': (0, 0, 255),
        'partially_open': (0, 255, 255)
    }

    panel_w = 450
    margin = 30
    h, w = frame.shape[:2]
    faces_count = len(results)
    panel_height = max(h, 40 + (faces_count * 80) + margin)
    canvas = np.full((panel_height, w + panel_w + margin, 3), 50, dtype=np.uint8)
    canvas[:h, :w] = frame

    for i in range(faces_count):
        bbox = results.bboxes[i]
        landmarks = results.landmarks[i]
        l_state, r_state = results.left_states[i], results.right_states[i]

        if bbox is None:
            continue

        # handle landmarks based on detector type
        # for retinaface, we have actual landmark points
        # for dlib, we don't have reliable landmark points, so we'll estimate eye positions
        # based on the bounding box
        if detector_type == "retinaface" and landmarks is not None and len(landmarks) >= 2:
            l_eye = tuple(map(int, landmarks[0]))
            r_eye = tuple(map(int, landmarks[1]))
            cv2.circle(canvas, l_eye, 5, state_colors.get(l_state, (128, 128, 128)), -1)
            cv2.circle(canvas, r_eye, 5, state_colors.get(r_state, (128, 128, 128)), -1)
        elif detector_type == "dlib":
            x1, y1, x2, y2 = map(int, bbox)
            face_width = x2 - x1
            face_height = y2 - y1
            
            left_eye_x = x1 + int(face_width * 0.3)
            left_eye_y = y1 + int(face_height * 0.4)
            right_eye_x = x1 + int(face_width * 0.7)
            right_eye_y = y1 + int(face_height * 0.4)
            
            cv2.circle(canvas, (left_eye_x, left_eye_y), 5, state_colors.get(l_state, (128, 128, 128)), -1)
            cv2.circle(canvas, (right_eye_x, right_eye_y), 5, state_colors.get(r_state, (128, 128, 128)), -1)

        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.putText(canvas, f"Face {i+1}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if draw_landmarks and detector_type == "retinaface" and landmarks is not None:
            for pt in landmarks:
                if len(pt) >= 2:
                    cv2.circle(canvas, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)

    ax = w + margin // 2
    cv2.rectangle(canvas, (ax - 10, 0), (canvas.shape[1], canvas.shape[0]), (30, 30, 30), -1)
    cv2.line(canvas, (w + margin // 4, 0), (w + margin // 4, canvas.shape[0]), (100, 100, 100), 2)
    cv2.putText(canvas, "EYE ANALYSIS", (ax, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    y_cursor = 70
    for i in range(faces_count):
        l_state = results.left_states[i]
        r_state = results.right_states[i]
        l_conf = results.left_confidences[i]
        r_conf = results.right_confidences[i]

        cv2.putText(canvas, f"Face {i+1}:", (ax, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 2)
        y_cursor += 20

        for label, state, conf in [("Left Eye", l_state, l_conf), ("Right Eye", r_state, r_conf)]:
            cv2.putText(canvas, f"{label}:", (ax + 10, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.circle(canvas, (ax + 80, y_cursor - 5), 4, state_colors.get(state, (128, 128, 128)), -1)
            cv2.putText(canvas, f"{state.upper()}", (ax + 90, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            cv2.putText(canvas, f"Conf: {conf:.2f}", (ax + 220, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            y_cursor += 18

        if i < faces_count - 1:
            cv2.line(canvas, (ax, y_cursor), (ax + 400, y_cursor), (80, 80, 80), 1)
            y_cursor += 15
        else:
            y_cursor += 25

    legend_y = canvas.shape[0] - 60
    cv2.putText(canvas, "Legend:", (ax, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    for idx, (label, color) in enumerate([("Open", (0, 255, 0)), ("Closed", (0, 0, 255))]):
        cx = ax + 10 + (idx * 80)
        cv2.circle(canvas, (cx, legend_y + 15), 4, color, -1)
        cv2.putText(canvas, label, (cx + 10, legend_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    return canvas

def plot_training_history(history, show=False):
    sns.set_style("darkgrid")
    colors = {
        'train': '#13034d',
        'val': '#084d02'
    }

    epochs = range(1, len(history.history['loss']) + 1)

    metrics = {
        "Loss": ('loss', 'val_loss'),
        "Accuracy": ('accuracy', 'val_accuracy')
    }

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    def plot_metric(ax, train_data, val_data, title, ylabel, ylim=None):
        ax.plot(epochs, train_data, marker='o', markersize=6, linewidth=2, color=colors['train'], label="Training")
        ax.plot(epochs, val_data, marker='s', markersize=6, linewidth=2, color=colors['val'], label="Validation")
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        if ylim:
            ax.set_ylim(ylim)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper left', frameon=True, fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.6)

    for ax, (title, (train_key, val_key)) in zip(axs.flat, metrics.items()):
        plot_metric(ax, history.history[train_key], history.history[val_key], title, title, ylim=(0, 1) if 'Accuracy' in title else None)
    
    os.makedirs(os.path.join(constant.ROOT_DIR, "results"), exist_ok=True)
    output_path = "results/training_history.png"
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()


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
