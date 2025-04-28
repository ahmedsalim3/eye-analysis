import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class EyeStateResultContainer:

    left_states: list
    right_states: list
    left_confidences: list
    right_confidences: list
    bboxes: np.ndarray
    landmarks: np.ndarray
    scores: np.ndarray
    
    def __len__(self):
        return len(self.left_states)
    
    def get_combined_states(self):
        combined_states = []
        for l_state, r_state in zip(self.left_states, self.right_states):
            if l_state == "open" and r_state == "open":
                combined_states.append("open")
            elif l_state == "closed" and r_state == "closed":
                combined_states.append("closed")
            else:
                combined_states.append("partially_open")
        return combined_states
    
    def get_blink_status(self, threshold=0.8):
        blink_statuses = []
        for l_state, r_state, l_conf, r_conf in zip(
            self.left_states, self.right_states, 
            self.left_confidences, self.right_confidences
        ):
            is_blinking = (l_state == "closed" and l_conf > threshold) or \
                          (r_state == "closed" and r_conf > threshold)
            blink_statuses.append(is_blinking)
        return blink_statuses
    
    def to_dataframe(self):
        data = {
            'face_bboxes': [bbox if isinstance(bbox, (list, tuple)) else [bbox] for bbox in self.bboxes],
            'landmarks': [landmark if isinstance(landmark, (list, tuple)) else [landmark] for landmark in self.landmarks],
            'left_state': self.left_states,
            'right_state': self.right_states,
            'left_confidence': self.left_confidences,
            'right_confidence': self.right_confidences,
            'combined_state': self.get_combined_states(),
            'is_blinking': self.get_blink_status(),
            'scores': [score for score in self.scores]
        }
        
        return pd.DataFrame(data)
