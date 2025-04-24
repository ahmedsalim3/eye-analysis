import configparser
import os
import numpy as np
import cv2
import dlib
from tqdm import tqdm

from ..commons.logger import Logger
from ..utilities.data_utils import get_dlib_points, get_left_key_points, get_right_key_points, get_attributes_wrt_local_frame
from ..commons.constant import ROOT_DIR
from ..commons import folder_utils


class FacialImageDataset:
    """
    A class for processing facial image datasets to extract eye features for open/closed eye classification.
    
    This processor:
    1. Loads facial images from two categories (open and closed eyes)
    2. Extracts eye regions using provided eye coordinates in the txt files
    3. Detects facial landmarks within those regions
    4. Extracts key points, distances, and angles for each eye
    5. Processes and normalizes the data
    
    The processed features include:
    - Eye images: Normalized grayscale eye region images
    - Keypoints: Detected landmark points for each eye
    - Distances: Distance measurements between keypoints
    - Angles: Angular measurements between keypoints
    - Labels: Binary classification (0=closed, 1=open)

    The dataset is mainly structured as follows:

    dataset_B_FacialImages
    │
    ├── EyeCoordinatesInfo_ClosedFace.txt
    ├── EyeCoordinatesInfo_OpenFace.txt
    │
    ├── ClosedFace
    │   ├── closed_eye_0001.jpg_face_1.jpg
    │   ├── closed_eye_0002.jpg_face_2.jpg
    │   └──closed_eye_0003.jpg_face_2.jpg
    │
    └── OpenFace
        ├── Aaron_Guiel_0001.jpg
        ├── Abdel_Madi_Shabneh_0001.jpg
        └── Abid_Hamid_Mahmud_Al-Tikriti_0001.jpg

    
    To run the class, you can run as follows:
    ```python
    dataset = FacialImageDataset()
    dataset.process(debug_mode=False)
    ```
    """
    
    # URLs for dataset and model resources
    SHAPE_PREDICTOR_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    DATASET_URL = "https://drive.usercontent.google.com/u/0/uc?id=1FcRw251WxGKwT6Dt9ncP-1TbW6h445W5&export=download"
    
    def _init_(self):
        """Initialize the processor with configuration settings and paths."""
        self.logger = Logger()
        folder_utils.initialize_folder(ROOT_DIR)
        
        # Load configuration
        self.config = configparser.ConfigParser()
        self.config.read(os.path.join(ROOT_DIR, 'configs', 'model.conf'))
        
        self.dataset_name = self.config['facial_image']['dataset_name']
        self.dataset_path = self.config['facial_image']['folder_path']
        self.image_width = int(self.config['facial_image']['image_width'])
        self.image_height = int(self.config['facial_image']['image_height'])
        
        if self.image_width != self.image_height:
            self.logger.warning("Non-square target shape may affect model performance")
        
        # Set dataset paths
        self.closed_path = os.path.join(self.dataset_path, "ClosedFace")
        self.open_path = os.path.join(self.dataset_path, "OpenFace")
        self.closed_coords = os.path.join(self.dataset_path, "EyeCoordinatesInfo_ClosedFace.txt")
        self.open_coords = os.path.join(self.dataset_path, "EyeCoordinatesInfo_OpenFace.txt")
        
        # Initialize resources, and download dataset if not exists
        self._initialize_resources()

        self.process_data_dir = os.path.join(ROOT_DIR, "data", "processed")
        os.makedirs(self.process_data_dir, exist_ok=True)  
    
    def _initialize_resources(self):
        """
        Initialize necessary resources including
        dataset download and facial landmark predictor.
        """
        # Check if dataset exists, prompt download if not
        if not os.path.exists(self.dataset_path):
            self.logger.error(f"Dataset path does not exist: {self.dataset_path}\n do you want to download it? (y/n)")
            if input() == "y":
                folder_utils.download_dataset(
                    file_name="dataset_B_Facial_Images.rar",
                    source_url=self.DATASET_URL,
                    home_path=ROOT_DIR
                )
        
        self.logger.info(f"Dataset: {self.dataset_name}")
        
        # Download and load the facial landmark predictor
        shape_predictor_path = folder_utils.download_weights(
            file_name="shape_predictor_68_face_landmarks.dat",
            source_url=self.SHAPE_PREDICTOR_URL,
            home_path=ROOT_DIR
        )
        
        self.predictor = dlib.shape_predictor(shape_predictor_path)
    
    @staticmethod
    def parse_label_file(file_path, logger):
        """
        Parse a text file containing eye coordinate labels.
        """
        coordinates = {}
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    filename = parts[0]
                    x1, y1, x2, y2 = map(int, parts[1:5])
                    coordinates[filename] = (x1, y1, x2, y2)
        
        logger.info(f"Parsed {len(coordinates)} eye coordinates from {file_path}")
        return coordinates
    
    def extract_eye_features(self, image_path, eye_coords):
        """
        Extract eye features from an image using the specified eye coordinates.
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            self.logger.error(f"Failed to read image: {image_path}")
            return None
        
        x1, y1, x2, y2 = eye_coords
        
        # Create a rectangle around the eye region with padding
        rect = dlib.rectangle(
            left=max(0, x1-10), 
            top=max(0, y1-10), 
            right=min(image.shape[1], x2+10), 
            bottom=min(image.shape[0], y2+10)
        )
        
        try:
            # Extract landmarks and features
            landmarks = get_dlib_points(image, self.predictor, rect)
            
            left_key_points = get_left_key_points(landmarks)
            right_key_points = get_right_key_points(landmarks)
            
            # Get attributes for left eye
            left_eye_img, left_kp, left_dist, left_angles = get_attributes_wrt_local_frame(
                image, left_key_points, image_shape=(self.image_width, self.image_height, 1)
            )
            
            # Get attributes for right eye
            right_eye_img, right_kp, right_dist, right_angles = get_attributes_wrt_local_frame(
                image, right_key_points, image_shape=(self.image_width, self.image_height, 1)
            )
            
            return {
                'left': (left_eye_img, left_kp, left_dist, left_angles),
                'right': (right_eye_img, right_kp, right_dist, right_angles)
            }
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {e}")
            return None
    
    def _debug_eye_extraction(self, image_path, eye_coords):
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
            landmarks = get_dlib_points(gray_image, self.predictor, rect)
            
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
            features = self.extract_eye_features(image_path, eye_coords)
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
                self.logger.info("Debug visualizations displayed.")
                
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
            self.logger.error(f"Error in _debug_eye_extraction: {e}")
            cv2.destroyAllWindows()
        
        return None
    
    def _process_data(self, debug_mode=False):
        """
        Process the dataset to extract features for both open and closed eyes.
        """
        self.logger.info("Reading eye coordinates...")
        closed_coords = self.parse_label_file(self.closed_coords, self.logger)
        open_coords = self.parse_label_file(self.open_coords, self.logger)
        
        eye_images = []
        keypoints = []
        distances = []
        angles = []
        labels = []  # 0 for closed, 1 for open
        
        # Process closed eye images
        self.logger.info("Processing closed eye images...")
        for filename in tqdm(os.listdir(self.closed_path)):
            if filename in closed_coords:
                image_path = os.path.join(self.closed_path, filename)
                features = self.extract_eye_features(image_path, closed_coords[filename])
                
                # Debug the first closed eye image if debug mode is enabled
                if debug_mode and len(eye_images) == 0:
                    _ = self._debug_eye_extraction(image_path, closed_coords[filename])
                
                if features:
                    # Add left eye features
                    left_features = features['left']
                    eye_images.append(left_features[0])
                    keypoints.append(left_features[1])
                    distances.append(left_features[2])
                    angles.append(left_features[3])
                    labels.append(0)  # 0 for closed
                    
                    # Add right eye features
                    right_features = features['right']
                    eye_images.append(right_features[0])
                    keypoints.append(right_features[1])
                    distances.append(right_features[2])
                    angles.append(right_features[3])
                    labels.append(0)  # 0 for closed
                else:
                    self.logger.error(f"Failed to extract features for {image_path}")
        
        # Process open eye images
        self.logger.info("Processing open eye images...")
        for filename in tqdm(os.listdir(self.open_path)):
            if filename in open_coords:
                image_path = os.path.join(self.open_path, filename)
                features = self.extract_eye_features(image_path, open_coords[filename])
                
                # Debug the first open eye image if debug mode is enabled
                if debug_mode and len(labels) > 0 and labels.count(1) == 0:
                    _ = self._debug_eye_extraction(image_path, open_coords[filename])
                
                if features:
                    # Add left eye features
                    left_features = features['left']
                    eye_images.append(left_features[0])
                    keypoints.append(left_features[1])
                    distances.append(left_features[2])
                    angles.append(left_features[3])
                    labels.append(1)  # 1 for open
                    
                    # Add right eye features
                    right_features = features['right']
                    eye_images.append(right_features[0])
                    keypoints.append(right_features[1])
                    distances.append(right_features[2])
                    angles.append(right_features[3])
                    labels.append(1)  # 1 for open
                else:
                    self.logger.error(f"Failed to extract features for {image_path}")
        
        # Convert lists to numpy arrays and normalize
        X_eye_images = np.array(eye_images).astype(np.float32) / 255.0  # Normalize to [0,1]
        X_keypoints = np.array([kp.reshape(1, 11, 2) for kp in keypoints]).astype(np.float32) / 24.0  # Normalize by image size
        X_distances = np.array([d.reshape(1, 11, 1) for d in distances]).astype(np.float32) / 24.0
        X_angles = np.array([a.reshape(1, 11, 1) for a in angles]).astype(np.float32) / np.pi  # Normalize by pi
        y = np.array(labels)
        
        return X_eye_images, X_keypoints, X_distances, X_angles, y
    
    def _save_data(self, X_eye_images, X_keypoints, X_distances, X_angles, y):
        """
        Save processed dataset features to disk.
        
        Args:
            X_eye_images: Array of normalized eye images
            X_keypoints: Array of normalized keypoints
            X_distances: Array of normalized distances
            X_angles: Array of normalized angles
            y: Array of labels (0=closed, 1=open)
        """
              
        np.save(os.path.join(self.process_data_dir, "X_eye_images.npy"), X_eye_images)
        np.save(os.path.join(self.process_data_dir, "X_keypoints.npy"), X_keypoints)
        np.save(os.path.join(self.process_data_dir, "X_distances.npy"), X_distances)
        np.save(os.path.join(self.process_data_dir, "X_angles.npy"), X_angles)
        np.save(os.path.join(self.process_data_dir, "y.npy"), y)
        
        self.logger.info(f"Data saved to {self.process_data_dir}")
    
    def _load_data(self):
        """
        Load the processed data from disk.
        """
        X_eye_images = np.load(os.path.join(self.process_data_dir, "X_eye_images.npy"))
        X_keypoints = np.load(os.path.join(self.process_data_dir, "X_keypoints.npy"))
        X_distances = np.load(os.path.join(self.process_data_dir, "X_distances.npy"))
        X_angles = np.load(os.path.join(self.process_data_dir, "X_angles.npy"))
        y = np.load(os.path.join(self.process_data_dir, "y.npy"))
            
        return X_eye_images, X_keypoints, X_distances, X_angles, y
    
    def process(self, debug_mode=False, save_data=True):
        """
        Process the dataset and save the resulting features.
        
        Args:
            debug_mode: Whether to show debug visualizations
        """
        self.logger.info("Starting dataset processing...")
        
        # Process the dataset
        X_eye_images, X_keypoints, X_distances, X_angles, y = self._process_data(debug_mode=debug_mode)
        
        # Log dataset statistics
        self.logger.info("Dataset statistics:")
        self.logger.info(f"Total samples: {len(y)}")
        self.logger.info(f"Open eyes: {np.sum(y == 1)}")
        self.logger.info(f"Closed eyes: {np.sum(y == 0)}")
        
        # Save the processed data
        if save_data:
            self._save_data(X_eye_images, X_keypoints, X_distances, X_angles, y)

        return X_eye_images, X_keypoints, X_distances, X_angles, y
    
    def load_data(self):
        """
        Load the processed data from disk.
        """
        return self._load_data()

def main():
    """
    Main entry point for the eye dataset processor.
    """
    processor = FacialImageDataset()
    processor.process(debug_mode=False)


if __name__ == "_main_":
    main()