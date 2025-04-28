import os
import argparse
import configparser

# disable tensorflow warnings logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical  # type: ignore
from sklearn.model_selection import train_test_split

from src.dataset.facial_images import FacialImageDataset
from src.commons.constant import ROOT_DIR
from src.commons.logger import Logger
from src.modeling.models import FusionNet
from src.utilities.vis_utils import plot_training_history

logger = Logger()

def train_model(data, epochs=50, batch_size=32,
                val_split=0.2, lr=0.001, early_stop_patience=5,
                lr_reduce_patience=3, random_state=16, model_name="eye_state_model"):
    
    X_eye_images = data["X_eye_images"]
    X_keypoints = data["X_keypoints"]
    X_distances = data["X_distances"]
    X_angles = data["X_angles"]
    y = data["y"]

    # Convert to one-hot encoding
    y_categorical = to_categorical(y, num_classes=2)
    
    # =========== train-test split ===========
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=val_split, stratify=y, random_state=random_state)
    
    logger.info("Creating model...")
    model = FusionNet(compile=False).model
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    os.makedirs(os.path.join(ROOT_DIR, "weights"), exist_ok=True)
    checkpoint_path = os.path.join(ROOT_DIR, "weights", f"{model_name}.weights.h5")
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor='val_accuracy', verbose=1,
            save_best_only=True, mode='max', save_weights_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=early_stop_patience, verbose=1, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=lr_reduce_patience, verbose=1
        ),
    ]
    
    # =========== Train ===========
    logger.info("Training model...")
    history = model.fit(
        [
            X_eye_images[train_idx],
            X_keypoints[train_idx],
            X_distances[train_idx],
            X_angles[train_idx]
        ],
        y_categorical[train_idx],
        validation_data=(
            [
                X_eye_images[test_idx],
                X_keypoints[test_idx],
                X_distances[test_idx],
                X_angles[test_idx]
            ],
            y_categorical[test_idx]
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    # =========== Evaluate ===========
    test_loss, test_acc = model.evaluate(
        [
            X_eye_images[test_idx],
            X_keypoints[test_idx],
            X_distances[test_idx],
            X_angles[test_idx]
        ],
        y_categorical[test_idx]
    )
    logger.info(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")
    logger.info(f"Best model saved to {checkpoint_path}")
    
    return history, model

def load_config():
    """Load configuration from the config file, excluding default values"""
    config = configparser.ConfigParser()
    config_path = os.path.join(ROOT_DIR, 'configs', 'model.conf')
    
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found at {config_path}. Using default values.")
        return {}
    
    config.read(config_path)
    
    # Prepare a dictionary to store config values
    config_values = {}

    config_values['model_name'] = config['model']['model_name']
    config_values['network'] = config['model']['network']
    config_values["epochs"] = config.getint("training", "epochs")
    config_values["batch_size"] = config.getint("training", "batch_size")
    config_values["val_split"] = config.getfloat("training", "val_split")
    config_values["learning_rate"] = config.getfloat("training", "learning_rate")
    config_values["early_stop_patience"] = config.getint("training", "early_stop_patience")
    config_values["lr_reduce_patience"] = config.getint("training", "lr_reduce_patience")
    config_values["random_state"] = config.getint("training", "random_state")

    return config_values

if __name__ == "__main__":
    config_values = load_config()
    print(config_values)
    
    # create argument parser with defaults from config
    parser = argparse.ArgumentParser(description="Train eye state classifier model")
    parser.add_argument("--network", type=str, default=config_values["network"],
                        choices=["mi", "si"],
                        help=f"Network type to train the model, either multi-input or single input (default: {config_values['network']})")
    parser.add_argument("--epochs", type=int, default=config_values["epochs"], 
                        help=f"Number of training epochs (default: {config_values['epochs']})")
    parser.add_argument("--batch-size", type=int, default=config_values["batch_size"], 
                        help=f"Training batch size (default: {config_values['batch_size']})")
    parser.add_argument("--val-split", type=float, default=config_values["val_split"], 
                        help=f"Validation split ratio (default: {config_values['val_split']})")
    parser.add_argument("--lr", type=float, default=config_values["learning_rate"], 
                        help=f"Initial learning rate (default: {config_values['learning_rate']})")
    parser.add_argument("--early-stop", type=int, default=config_values["early_stop_patience"], 
                        help=f"Early stopping patience (default: {config_values['early_stop_patience']})")
    parser.add_argument("--lr-patience", type=int, default=config_values["lr_reduce_patience"], 
                        help=f"LR reduction patience (default: {config_values['lr_reduce_patience']})")
    parser.add_argument("--random-state", type=int, default=config_values["random_state"], 
                        help=f"Random state for reproducibility (default: {config_values['random_state']})")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # get the dataset
    if args.network == "mi":
        data_processor = FacialImageDataset()
        try:
            data = data_processor.load_data()
        except Exception:
            data = data_processor.process(debug_mode=args.debug)
    
        history, model = train_model(
            data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_split=args.val_split,
            lr=args.lr,
            early_stop_patience=args.early_stop,
            lr_reduce_patience=args.lr_patience,
            random_state=args.random_state,
            model_name=config_values["model_name"]
        )
        
        plot_training_history(history, args.debug)
    
    elif args.network == "si":
        logger.error("Single input network is not implemented yet.")
        exit(1)
