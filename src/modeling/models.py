import os
# disable tensorflow warnings logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate # type: ignore
import tensorflow as tf

from src.commons.logger import Logger

logger = Logger()

# Configure GPU to avoid OOM
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    logger.warn(f"Could not configure GPU memory growth: {e}")

class FusionNet:
    
    def __init__(self, compile=True):
        """
        FusionNet: Multi-input fusion-based architecture
        
        Args:
            compile (bool): If True, the model will be compiled immediately after creation.
        """
        self.model = self.classifier()
        if compile:
            self.compile_model()
    
    
    @staticmethod
    def classifier():
        """
        This model uses a multi-branch convolutional neural network architecture to classify
        eye states as either open or closed.

        Architecture:
            - The main branch processes eye images through convolutional layers and pooling
            - Three auxiliary branches process additional feature inputs
            - Features from all branches are concatenated and passed through fully connected layers
            - Final output is a binary classification (open/closed)

        Returns:
            A compiled Keras Model instance ready for training or inference.
        """

        # inputs
        input_1 = Input(shape=(24, 24, 1), name='input_1')
        input_2 = Input(shape=(1, 11, 2), name='input_2')
        input_3 = Input(shape=(1, 11, 1), name='input_3')
        input_4 = Input(shape=(1, 11, 1), name='input_4')
        
        # 1 - image processing
        x1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_1')(input_1)
        x1 = MaxPooling2D((2, 2), name='max_pooling2d_1')(x1)
        x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_2')(x1)
        x1 = MaxPooling2D((2, 2), name='max_pooling2d_2')(x1)
        x1 = Flatten(name='flatten_1')(x1)
        
        # 2
        x2 = Conv2D(8, (1, 3), activation='relu', padding='same', name='conv2d_3')(input_2)
        x2 = Conv2D(16, (1, 3), activation='relu', padding='same', name='conv2d_4')(x2)
        x2 = Flatten(name='flatten_2')(x2)
        
        # 3
        x3 = Conv2D(8, (1, 3), activation='relu', padding='same', name='conv2d_5')(input_3)
        x3 = Conv2D(16, (1, 3), activation='relu', padding='same', name='conv2d_6')(x3)
        x3 = Flatten(name='flatten_3')(x3)
        
        # 4
        x4 = Conv2D(8, (1, 3), activation='relu', padding='same', name='conv2d_7')(input_4)
        x4 = Conv2D(16, (1, 3), activation='relu', padding='same', name='conv2d_8')(x4)
        x4 = Flatten(name='flatten_4')(x4)
        
        # Concatenate all branches
        merged = Concatenate(axis=-1, name='concatenate_1')([x1, x2, x3, x4])
        
        # FC layers
        x = Dense(128, activation='relu', name='dense_1')(merged)
        x = Dropout(0.2, name='dropout_1')(x)
        x = Dense(256, activation='relu', name='dense_2')(x)
        x = Dropout(0.2, name='dropout_2')(x)
        
        # output
        output = Dense(2, activation='softmax', name='dense_3')(x)
        
        model = Model(inputs=[input_1, input_2, input_3, input_4], outputs=output, name='model_1')
        
        return model

    def compile_model(self):
        """Compile the model with the specified optimizer, loss, and metrics."""
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
