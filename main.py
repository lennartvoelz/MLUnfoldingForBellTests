from src.data_preproc.preprocessing import DataPreprocessor
from src.reconstruction.dense_nn import DNN
import yaml
from src.evaluation.evaluation import calculate_results
from src.evaluation.calculate_mae import results
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import LogCosh
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping

config = yaml.safe_load(open('config.yaml'))

data_preprocessor_no_cuts = DataPreprocessor(data_path=config['data_path'], cuts=False)
data_preprocessor_cuts = DataPreprocessor(data_path=config['data_path'], cuts=True)

X_no_cuts_train, X_no_cuts_val, X_no_cuts_test, y_no_cuts_train, y_no_cuts_val, y_no_cuts_test = data_preprocessor_no_cuts.run_preprocessing()
X_cuts_train, X_cuts_val, X_cuts_test, y_cuts_train, y_cuts_val, y_cuts_test = data_preprocessor_cuts.run_preprocessing()

dense_net = DNN(config)

optimizer = Adam(learning_rate=config['learning_rate'])
loss = LogCosh()
metrics = [MeanAbsoluteError()]
callbacks = [EarlyStopping(monitor='val_loss', patience=config['patience'])]

# Train DNN without cuts
dense_net = DNN(config)
compiled_dnn = dense_net.compile(optimizer=optimizer, loss=loss, metrics=metrics)

history_no_cuts = dense_net.fit(X_no_cuts_train, y_no_cuts_train, X_no_cuts_val, y_no_cuts_val, callbacks=callbacks)

# Evaluate DNN without cuts
y_no_cuts_pred = dense_net.predict(X_no_cuts_test)
final_state_no_cuts = np.concatenate((X_no_cuts_test, y_no_cuts_pred), axis=1)
final_state_no_cuts_truth = np.concatenate((X_no_cuts_test, y_no_cuts_test), axis=1)

results_no_cuts = calculate_results(truth=final_state_no_cuts_truth, neural_network_reconstruction=final_state_no_cuts)
results_no_cuts.run("reports/figures/")

results_no_cuts = results(truth=final_state_no_cuts_truth, neural_network_reconstruction=final_state_no_cuts)
results_no_cuts.run("reports/regression_results/")

# Train DNN with cuts
dense_net = DNN(config)
compiled_dnn = dense_net.compile(optimizer=optimizer, loss=loss, metrics=metrics)

history_cuts = dense_net.fit(X_cuts_train, y_cuts_train, X_cuts_val, y_cuts_val, callbacks=callbacks)

# Evaluate DNN with cuts
y_cuts_pred = dense_net.predict(X_cuts_test)
final_state_cuts = np.concatenate((X_cuts_test, y_cuts_pred), axis=1)
final_state_cuts_truth = np.concatenate((X_cuts_test, y_cuts_test), axis=1)

results_cuts = calculate_results(truth=final_state_cuts_truth, neural_network_reconstruction=final_state_cuts)
results_cuts.run("reports/figures/")
results_cuts = results(truth=final_state_cuts_truth, neural_network_reconstruction=final_state_cuts)
results_cuts.run("reports/regression_results/")