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

data_preprocessor_detector_sim = DataPreprocessor(data_path=config["data_path"], cuts=False)

X_detector_sim_train, X_detector_sim_val, X_detector_sim_test, y_detector_sim_train, y_detector_sim_val, y_detector_sim_test = data_preprocessor_detector_sim.run_preprocessing()

optimizer = Adam(learning_rate=config['learning_rate'])
loss = LogCosh()
metrics = [MeanAbsoluteError()]
callbacks = [EarlyStopping(monitor='val_loss', patience=config['patience'])]

# Train DNN with cuts
dense_net = DNN(config)
dense_net.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

history_detector_sim = dense_net.model.fit(X_detector_sim_train, y_detector_sim_train, validation_data=(X_detector_sim_val, y_detector_sim_val), callbacks=callbacks, epochs=config['epochs'], batch_size=config['batch_size'])

# Evaluate DNN with cuts
y_detector_sim_pred = dense_net.model.predict(X_detector_sim_test)
final_state_detector_sim = np.concatenate((X_detector_sim_test[:,:8], y_detector_sim_pred), axis=1)
final_state_detector_sim_truth = np.concatenate((X_detector_sim_test[:,:8], y_detector_sim_test), axis=1)

# Save the model
dense_net.model.save('models/dnn_detector_sim.keras')

# Save final state to csv
np.savetxt("data/dnn_final_state_ww.csv", final_state_detector_sim)
np.savetxt("data/dnn_final_state_ww_truth.csv", final_state_detector_sim_truth)