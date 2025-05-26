from src.data_preproc.preprocessing import DataPreprocessor
from src.reconstruction.dense_nn import DNN
import yaml
from src.evaluation.evaluation import calculate_results
from src.evaluation.calculate_mae import results
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import LogCosh
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner.tuners import BayesianOptimization
from tensorflow.keras import backend as K

class ClearSessionCallback(tf.keras.callbacks.Callback):
    def on_train_end(self, logs=None):
        K.clear_session()

config = yaml.safe_load(open('config.yaml'))

data_preprocessor_detector_sim = DataPreprocessor(data_path="/mnt/c/Users/Lennart/Desktop/Studium/MLUnfoldingForBellTests/data/hww_simulated_final.csv", cuts=False)

X_detector_sim_train, X_detector_sim_val, X_detector_sim_test, y_detector_sim_train, y_detector_sim_val, y_detector_sim_test, types = data_preprocessor_detector_sim.run_preprocessing()

def create_model(hp):
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2)
    weight_decay = hp.Float('weight_decay', min_value=1e-5, max_value=1e-2)
    batch_normalization = hp.Choice('batch_normalization', values=[True, False])
    dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.05)
    num_dense_layers = hp.Int('num_dense_layers', min_value=7, max_value=20)
    dense_nodes = hp.Int('dense_nodes', min_value=1000, max_value=3500, step=100)
    activation_functions = hp.Choice('activation_functions', values=['relu', 'elu'])

    # Overwrite the config file
    config['learning_rate'] = learning_rate
    config['l2_regularization'] = weight_decay
    config['batch_normalization'] = batch_normalization
    config['dropout_rate'] = dropout
    config['num_dense_layers'] = num_dense_layers
    config['dense_nodes'] = dense_nodes
    config['activation_function'] = activation_functions

    optimizer = AdamW(learning_rate=config['learning_rate'], weight_decay=config['l2_regularization'])
    loss = LogCosh()
    metrics = [MeanAbsoluteError()]

    dense_net = DNN(config=config)

    dense_net.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return dense_net.model

bayes_tuner = BayesianOptimization(
    create_model,
    objective='val_loss',
    max_trials=70,
    executions_per_trial=1,
    directory='hyperparameter_opt_detector',
    project_name='higgs_detector_sim'
)

callbacks = [EarlyStopping(monitor='val_loss', patience=config['patience']), ClearSessionCallback()]

bayes_tuner.search(X_detector_sim_test, y_detector_sim_test, validation_data=(X_detector_sim_val, y_detector_sim_val), epochs=20, callbacks=callbacks, batch_size=config['batch_size'])

print(bayes_tuner.results_summary())

best_params = bayes_tuner.get_best_hyperparameters(num_trials=3)[0]

print(best_params.values)