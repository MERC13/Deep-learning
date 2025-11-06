import os
import tensorflow as tf
import json
import logging
import random
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Input, LSTM, LayerNormalization, GaussianNoise
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from dataprocessing import dataprocessing
import joblib

# Optional: accelerator selection and GPU memory behavior
def configure_accelerator():
    """Configure TensorFlow to use GPU or CPU based on env var USE_GPU.

    USE_GPU values:
      - auto (default): prefer GPU if available, else CPU
      - gpu: require GPU (raise if none)
      - cpu: force CPU only

    Also enables memory growth on all detected GPUs and can enable mixed
    precision when MIXED_PRECISION is set to '1'/'true'.
    """
    mode = os.getenv("USE_GPU", "cpu").strip().lower()

    gpus = tf.config.list_physical_devices('GPU')
    if mode == 'cpu':
        tf.config.set_visible_devices([], 'GPU')
        logging.info("USE_GPU=cpu -> Forcing CPU. GPUs hidden from TF.")
    elif mode in ('gpu', 'auto'):
        if gpus:
            # Enable memory growth on all GPUs
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.set_visible_devices(gpus, 'GPU')
                logical_gpus = tf.config.list_logical_devices('GPU')
                logging.info(f"Using GPU(s): {[g.name for g in logical_gpus]}")
            except Exception as e:
                logging.warning(f"Could not set GPU memory growth: {e}")
        else:
            if mode == 'gpu':
                raise RuntimeError("USE_GPU=gpu but no GPU is available to TensorFlow.")
            logging.info("No GPU detected. Falling back to CPU (USE_GPU=auto).")

    # Optional: Mixed precision
    mp = os.getenv("MIXED_PRECISION", "0").strip().lower() in ("1", "true", "yes")
    if mp:
        try:
            from tensorflow.keras import mixed_precision as mp_policy
            policy = mp_policy.Policy('mixed_float16')
            mp_policy.set_global_policy(policy)
            logging.info("Enabled mixed precision policy: mixed_float16")
        except Exception as e:
            logging.warning(f"Could not enable mixed precision: {e}")

############################
# Setup & Utilities
############################

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )


def save_artifacts(model, params, scalers, metadata, model_file, params_file, scalers_file, metadata_file):
    model.save(model_file)
    joblib.dump(params, params_file)
    joblib.dump(scalers, scalers_file)
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)


def load_model_and_params(model_filename, params_filename):
    model = load_model(model_filename)
    params = joblib.load(params_filename)
    return model, params


def create_model(input_shape, units=64, learning_rate=0.0003, dropout_rate=0.3, l2_reg=0.01):
    inputs = Input(shape=input_shape)
    x = GaussianNoise(0.01)(inputs)
    x = Bidirectional(LSTM(
        units,
        return_sequences=True,
        dropout=0.2,
        recurrent_dropout=0.2,
        kernel_regularizer=l2(l2_reg),
        recurrent_regularizer=l2(l2_reg)
    ))(x)
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Bidirectional(LSTM(
        units // 2,
        return_sequences=False,
        dropout=0.2,
        recurrent_dropout=0.2,
        kernel_regularizer=l2(l2_reg),
        recurrent_regularizer=l2(l2_reg)
    ))(x)
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(units // 4, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)

    outputs = Dense(1, kernel_regularizer=l2(l2_reg))(x)
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model


def create_or_load_model(MODEL_FILE, PARAMS_FILE, input_shape):
    force_new = os.getenv("FORCE_NEW_MODEL", "1").lower() in ("1", "true", "yes")
    if not force_new and os.path.exists(MODEL_FILE) and os.path.exists(PARAMS_FILE):
        logging.info("Loading existing model and parameters...")
        best_model, best_params = load_model_and_params(MODEL_FILE, PARAMS_FILE)
        return best_model, best_params
    if force_new:
        logging.info("FORCE_NEW_MODEL set - creating a fresh model with default parameters")
    # Defaults when no saved model/params are present
    default_params = {
        'units': 64,
        'learning_rate': 0.0003,
        'dropout_rate': 0.3,
        'l2_reg': 0.01
    }

    logging.info("No existing model found. Using default parameters...")
    best_model = create_model(input_shape,
                              units=default_params['units'],
                              learning_rate=default_params['learning_rate'],
                              dropout_rate=default_params['dropout_rate'],
                              l2_reg=default_params['l2_reg'])
    return best_model, default_params


def plot_predictions(y_true, y_pred, name, rmse):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', linewidth=2)
    plt.plot(y_pred, label='Predicted', linewidth=2, alpha=0.7)
    plt.title(f'{name} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    
    plt.text(0.05, 0.95, f'RMSE: {rmse:.2f}', transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'static/{name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_learning_curves(history, stock):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Learning Curves for {stock}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'static/learning_curves_{stock}.png')
    plt.close()
    
def inverse_transform_close(data, scaler, target_index):
    """Inverse-transform only the target feature from standardized values.

    Accepts arrays shaped as:
      - (N,)                        -> already 1D target series
      - (N, 1)                      -> single feature column
      - (N, T) or (N, T, 1)         -> sequence outputs; takes last timestep
    """
    arr = np.asarray(data)
    # Reduce sequence outputs to last timestep
    if arr.ndim == 3:  # (N, T, C)
        # If last dim is 1, squeeze it; then take last timestep
        if arr.shape[-1] == 1:
            arr = arr[:, -1, 0]
        else:
            arr = arr[:, -1]
    if arr.ndim == 2:  # (N, K) -> pick last column if not singleton
        if arr.shape[1] == 1:
            arr = arr[:, 0]
        else:
            arr = arr[:, -1]
    # Now arr should be (N,)
    arr = arr.reshape(-1)
    # StandardScaler inverse for a single feature: x*scale + mean
    mean = scaler.mean_[target_index]
    scale = scaler.scale_[target_index]
    return arr * scale + mean

    
    
    

############################
# Training script (main)
############################

if __name__ == "__main__":
    setup_logging()
    set_seeds(42)
    configure_accelerator()

    # Data processing
    tech_list = ['DJIA']  # extend as needed
    logging.info(f"Preparing data for: {tech_list}")
    x_train, y_train, x_test, y_test, metadata, scalers = dataprocessing(tech_list)
    input_shape = metadata['shape']

    # Filenames
    MODEL_FILE = 'best_DJIA_model.h5'
    PARAMS_FILE = 'best_model_params.joblib'
    SCALERS_FILE = 'scalers.joblib'
    METADATA_FILE = 'preprocess_metadata.json'

    # Build or load model
    best_model, best_params = create_or_load_model(MODEL_FILE, PARAMS_FILE, input_shape)

    early_stop = EarlyStopping(monitor='val_loss', patience=15, min_delta=1e-4, restore_best_weights=True)

    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

    # Train per stock (single-task)
    for stock in tech_list:
        logging.info(f"Training for {stock} with {x_train[stock].shape[0]} samples (val/test: {x_test[stock].shape[0]})")
        # Use a validation split from the training data instead of test to avoid leaking test into training decisions
        # Callbacks to reduce overfitting / improve generalization
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        checkpoint_path = f"best_{stock}_model.h5"
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=1)

        history = best_model.fit(
            x_train[stock], y_train[stock],
            batch_size=64,
            epochs=100,
            validation_split=0.2,
            shuffle=False,
            callbacks=[early_stop, reduce_lr, checkpoint, tensorboard_callback],
            verbose=1
        )

        # Plots
        plot_learning_curves(history, stock)

        # Predictions and inverse-transform using stock-specific scaler
        # Use metadata-derived target index to avoid coupling to feature order
        target_idx = metadata['features'].index('Close')

        # Use best checkpointed model if available
        try:
            best_model = load_model(f"best_{stock}_model.h5")
            logging.info(f"Loaded best checkpointed model for {stock}.")
        except Exception:
            logging.info(f"No checkpoint found for {stock}, using current model weights.")

        scaler = scalers[stock]
        train_pred = best_model.predict(x_train[stock])
        test_pred = best_model.predict(x_test[stock])

        y_train_inv = inverse_transform_close(y_train[stock], scaler, target_idx)
        y_test_inv = inverse_transform_close(y_test[stock], scaler, target_idx)
        train_pred_inv = inverse_transform_close(train_pred, scaler, target_idx)
        test_pred_inv = inverse_transform_close(test_pred, scaler, target_idx)

        # Align lengths if needed (should already match, but guard just in case)
        if train_pred_inv.shape[0] != y_train_inv.shape[0]:
            m = min(train_pred_inv.shape[0], y_train_inv.shape[0])
            logging.warning(f"Train length mismatch (pred={train_pred_inv.shape[0]}, true={y_train_inv.shape[0]}). Trimming to {m}.")
            train_pred_inv = train_pred_inv[:m]
            y_train_inv = y_train_inv[:m]
        if test_pred_inv.shape[0] != y_test_inv.shape[0]:
            m = min(test_pred_inv.shape[0], y_test_inv.shape[0])
            logging.warning(f"Test length mismatch (pred={test_pred_inv.shape[0]}, true={y_test_inv.shape[0]}). Trimming to {m}.")
            test_pred_inv = test_pred_inv[:m]
            y_test_inv = y_test_inv[:m]

        train_rmse = float(np.sqrt(mean_squared_error(y_train_inv, train_pred_inv)))
        test_rmse = float(np.sqrt(mean_squared_error(y_test_inv, test_pred_inv)))
        logging.info(f"{stock} Train RMSE: {train_rmse:.2f} | Test RMSE: {test_rmse:.2f}")

        # Save prediction plots with consistent names
        safe = stock.replace('^', '').replace('.', '_')
        plot_predictions(y_train_inv, train_pred_inv, f"{safe}_train_prediction", train_rmse)
        plot_predictions(y_test_inv, test_pred_inv, f"{safe}_test_prediction", test_rmse)

    # Save artifacts at the end
    save_artifacts(best_model, best_params, scalers, metadata, MODEL_FILE, PARAMS_FILE, SCALERS_FILE, METADATA_FILE)
    logging.info("Training completed and artifacts saved.")
    