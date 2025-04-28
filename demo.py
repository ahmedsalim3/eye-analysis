import os
import cv2
import pathlib


from src.modules.pipeline import Pipeline
from src.utilities.vis_utils import render
from src.commons.logger import Logger
from src.commons.get_weights import download_shape_predictor

logger = Logger()

def test_on_images(model_weights, test_dir, shape_predictor_path):

    pipeline = Pipeline(
        weights=pathlib.Path(model_weights),
        shape_predictor=pathlib.Path(shape_predictor_path),
        detector="retinaface",  # or "dlib"
        device="cpu",
        confidence_threshold=0.5
    )
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(test_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_dir, filename)
            
            frame = cv2.imread(image_path)
            if frame is None:
                logger.info(f"Failed to read image: {image_path}")
                continue
            
            results = pipeline.step(frame)            
            output_frame = render(frame, results, draw_landmarks=True)            
            output_path = os.path.join(output_dir, f"result_{filename}")
            cv2.imwrite(output_path, output_frame)
            logger.info(f"Processed {filename} -> {output_path}")

def test_webcam(model_weights, shape_predictor_path, camera_id=0):

    pipeline = Pipeline(
        weights=pathlib.Path(model_weights),
        shape_predictor=pathlib.Path(shape_predictor_path),
        detector="retinaface",  # or "dlib"
        device="cpu",
        confidence_threshold=0.5
    )
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.info(f"Failed to open camera with ID {camera_id}")
        return
    
    logger.info("Press 'q' to quit")
    logger.info("Press 'd' to toggle debug mode (show landmarks)")
    
    draw_landmarks = False

    while True:

        ret, frame = cap.read()
        if not ret:
            logger.info("Failed to read frame from camera")
            break
        
        results = pipeline.step(frame)        
        output_frame = render(frame, results, draw_landmarks=draw_landmarks)        
        cv2.imshow("Eye State Detection", output_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            draw_landmarks = not draw_landmarks
            logger.info(f"Landmarks: {'ON' if draw_landmarks else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()

def main():

    model_weights = "./weights/eye_state_model.weights.h5"
    shape_predictor_path = download_shape_predictor()
    
    logger.info("Select test mode:")
    logger.info("1. Test on image directory")
    logger.info("2. Test using webcam")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        test_dir = input("Enter path to test images directory: ")
        if not os.path.exists(test_dir):
            logger.info(f"Error: Test directory '{test_dir}' not found.")
            return
        test_on_images(model_weights, test_dir, shape_predictor_path)
    elif choice == "2":
        camera_id = int(input("Enter camera ID (default: 0): ") or "0")
        test_webcam(model_weights, shape_predictor_path, camera_id)
    else:
        logger.info("Invalid choice. Please run the script again and select 1 or 2.")

if __name__ == "__main__":
    main()