import os
import json
import logging
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Input, LSTM, LayerNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2

# Optional: hyperparameter tuning (kept but disabled by default)
try:
    from scikeras.wrappers import KerasRegressor
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    HAS_SKOPT = True
except Exception:
    HAS_SKOPT = False

from dataprocessing import dataprocessing
import joblib

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






def bayesian_hyperparameter_tuning(x_train, y_train, x_val, y_val):
    # Define the search space
    search_spaces = {
        'model__units': Integer(32, 128),
        'model__learning_rate': Real(1e-4, 1e-2, prior='log-uniform'),
        'model__dropout_rate': Real(0.1, 0.5),
        'model__l2_reg': Real(1e-6, 1e-3, prior='log-uniform'),
    }

    # Create a custom scoring function that uses the validation set
    def custom_scorer(estimator, X, y):
        y_pred = estimator.predict(x_val)
        return -mean_squared_error(y_val, y_pred)

    # Create the KerasRegressor
    model = KerasRegressor(
        model=create_model,
        loss='mean_squared_error',
        optimizer='adam',
        metrics=['mse'],
        verbose=0
    )

    # Create the BayesSearchCV object
    bayes_search = BayesSearchCV(
        estimator=model,
        search_spaces=search_spaces,
        n_iter=50,  # Number of parameter settings that are sampled
        cv=[(slice(None), slice(None))],  # Use all training data
        n_jobs=-1,  # Use all available cores
        verbose=1,
        scoring=custom_scorer,
        random_state=42
    )

    # Fit the BayesSearchCV object to the data
    bayes_search.fit(x_train, y_train)

    # Print the best parameters and score
    print("Best parameters found: ", bayes_search.best_params_)
    print("Best validation MSE: ", -bayes_search.best_score_)

    return bayes_search.best_params_





def create_multi_task_model(input_shape, stock_outputs, units=64, learning_rate=0.001, dropout_rate=0.2, l2_reg=0.01):
    # Keeping this function integrated, but not used by default
    inputs = Input(shape=input_shape)
    x = LSTM(units, return_sequences=True,
             kernel_regularizer=l2(l2_reg),
             recurrent_regularizer=l2(l2_reg))(inputs)
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)
    # Global average pooling for stability
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dense(units // 2, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)
    outputs = {s: Dense(1, name=f'output_{s.replace("^", "").replace(".", "_")}',
                        kernel_regularizer=l2(l2_reg))(x) for s in stock_outputs}
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss={k: 'mean_squared_error' for k in outputs.keys()})
    return model




def create_model(input_shape, units=64, learning_rate=0.001, dropout_rate=0.2, l2_reg=0.01):
    inputs = Input(shape=input_shape)
    
    # Bidirectional LSTM layers
    x = Bidirectional(LSTM(units, return_sequences=True, 
                           kernel_regularizer=l2(l2_reg), 
                           recurrent_regularizer=l2(l2_reg)))(inputs)
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    x = Bidirectional(LSTM(units // 2, return_sequences=True, 
                           kernel_regularizer=l2(l2_reg), 
                           recurrent_regularizer=l2(l2_reg)))(x)
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = Dense(units // 2, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(units // 4, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = Dense(1, kernel_regularizer=l2(l2_reg))(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model





def create_or_load_model(MODEL_FILE, PARAMS_FILE, input_shape, tech_list, bayesian=False, multi_task=False):
    if os.path.exists(MODEL_FILE) and os.path.exists(PARAMS_FILE):
        logging.info("Loading existing model and parameters...")
        best_model, best_params = load_model_and_params(MODEL_FILE, PARAMS_FILE)
    elif bayesian:
        if not HAS_SKOPT:
            logging.warning("Bayesian tuning requested but skopt/scikeras not available. Falling back to defaults.")
            best_params = None
        else:
            logging.info("No existing model found. Running Bayesian hyperparameter tuning...")
            # Very basic split for tuning on first stock
            from sklearn.model_selection import train_test_split
            x_train_tune, x_val, y_train_tune, y_val = train_test_split(
                x_train[tech_list[0]], y_train[tech_list[0]], test_size=0.2, random_state=42, shuffle=False
            )
            best_params = bayesian_hyperparameter_tuning(x_train_tune, y_train_tune, x_val, y_val)
        if multi_task:
            p = best_params or {'units': 64, 'learning_rate': 0.005, 'dropout_rate': 0.5, 'l2_reg': 0.2}
            best_model = create_multi_task_model(input_shape, tech_list,
                                                 units=p['units'], learning_rate=p['learning_rate'],
                                                 dropout_rate=p['dropout_rate'], l2_reg=p['l2_reg'])
        else:
            p = best_params or {'units': 64, 'learning_rate': 0.005, 'dropout_rate': 0.5, 'l2_reg': 0.2}
            best_model = create_model(input_shape,
                                      units=p['units'], learning_rate=p['learning_rate'],
                                      dropout_rate=p['dropout_rate'], l2_reg=p['l2_reg'])
        save_artifacts(best_model, p, scalers, metadata, MODEL_FILE, PARAMS_FILE, SCALERS_FILE, METADATA_FILE)
    else:
        default_params = {
            'units': 64,
            'learning_rate': 0.005,
            'dropout_rate': 0.5,
            'l2_reg': 0.2
        }
        logging.info("No existing model found. Using default parameters...")
        p = default_params
        if multi_task:
            best_model = create_multi_task_model(input_shape, tech_list,
                                                 units=p['units'], learning_rate=p['learning_rate'],
                                                 dropout_rate=p['dropout_rate'], l2_reg=p['l2_reg'])
        else:
            best_model = create_model(input_shape,
                                      units=p['units'], learning_rate=p['learning_rate'],
                                      dropout_rate=p['dropout_rate'], l2_reg=p['l2_reg'])
        save_artifacts(best_model, p, scalers, metadata, MODEL_FILE, PARAMS_FILE, SCALERS_FILE, METADATA_FILE)
    return best_model, p





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
    
    
    
    

############################
# Training script (main)
############################
























if __name__ == "__main__":
    setup_logging()
    set_seeds(42)

    # Data processing
    tech_list = ['^DJI']  # extend as needed
    logging.info(f"Preparing data for: {tech_list}")
    x_train, y_train, x_test, y_test, metadata, scalers = dataprocessing(tech_list)
    input_shape = metadata['shape']

    # Filenames
    MODEL_FILE = 'best_stock_model.h5'
    PARAMS_FILE = 'best_model_params.joblib'
    SCALERS_FILE = 'scalers.joblib'
    METADATA_FILE = 'preprocess_metadata.json'
    bayesian = False
    multi_task = False

    # Build or load model
    best_model, best_params = create_or_load_model(MODEL_FILE, PARAMS_FILE, input_shape, tech_list, bayesian, multi_task)

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

    # Train per stock (single-task)
    for stock in tech_list:
        logging.info(f"Training for {stock} with {x_train[stock].shape[0]} samples (val/test: {x_test[stock].shape[0]})")
        history = best_model.fit(
            x_train[stock], y_train[stock],
            batch_size=64,
            epochs=100,
            validation_data=(x_test[stock], y_test[stock]),
            callbacks=[early_stop, tensorboard_callback],
            verbose=1
        )

        # Plots
        plot_learning_curves(history, stock)

        # Predictions and inverse-transform using stock-specific scaler
        def inverse_transform_close(data_1d, scaler, n_features):
            tmp = np.zeros((len(data_1d), n_features))
            tmp[:, 3] = data_1d.flatten()
            return scaler.inverse_transform(tmp)[:, 3]

        scaler = scalers[stock]
        n_features = input_shape[1]
        train_pred = best_model.predict(x_train[stock])
        test_pred = best_model.predict(x_test[stock])

        y_train_inv = inverse_transform_close(y_train[stock], scaler, n_features)
        y_test_inv = inverse_transform_close(y_test[stock], scaler, n_features)
        train_pred_inv = inverse_transform_close(train_pred, scaler, n_features)
        test_pred_inv = inverse_transform_close(test_pred, scaler, n_features)

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