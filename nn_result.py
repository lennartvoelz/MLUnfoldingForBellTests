import numpy as np
import yaml
import tensorflow as tf
from src.evaluation.evaluation import calculate_results
from src.evaluation.calculate_mae import results
from src.data_preproc.preprocessing import DataPreprocessor

# final_state = np.loadtxt("/mnt/c/Users/Lennart/Desktop/Studium/MLUnfoldingForBellTests/data/dnn_detector_sim_final_state.csv")
# final_state_truth = np.loadtxt("/mnt/c/Users/Lennart/Desktop/Studium/MLUnfoldingForBellTests/data/dnn_detector_sim_final_state_truth.csv")

# config = yaml.safe_load(open('config.yaml'))
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

data = DataPreprocessor(data_path=config['data_path'], raw_data_path=config['raw_data_path'], cuts=True, splits=True)

X_train, X_val, X_test, y_train, y_val, y_test, types = data.run_preprocessing()

# Load DNN with cuts
dense_net = tf.keras.models.load_model('models\dnn_detector_sim.keras')

# Show the model architecture
dense_net.summary()

# Evaluate DNN with cuts
y_pred = dense_net.predict(X_test)
final_state = np.concatenate((X_test[:,:8], y_pred), axis=1)
final_state_truth = np.concatenate((X_test[:,:8], y_test), axis=1)

results_nn = calculate_results([final_state, final_state_truth], ["NN Regression", "Truth"], "Detector Simulation", types)
results_nn.run("reports/nn_detector_cuts/")

results_nn_mae = results(final_state_truth[:,8:], final_state[:,8:])
results_nn_mae.run("reports/nn_detector_cuts/")