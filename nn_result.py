from src.evaluation.evaluation import calculate_results
from src.evaluation.calculate_mae import results
import numpy as np

final_state = np.loadtxt("/mnt/c/Users/Lennart/Desktop/Studium/MLUnfoldingForBellTests/data/dnn_detector_sim_final_state.csv")
final_state_truth = np.loadtxt("/mnt/c/Users/Lennart/Desktop/Studium/MLUnfoldingForBellTests/data/dnn_detector_sim_final_state_truth.csv")

results_nn = calculate_results([final_state, final_state_truth], ["NN Regression", "Truth"], "Detector Simulation")
results_nn.run("reports/nn_detector_cuts/")

results_nn_mae = results(final_state_truth[:,8:], final_state[:,8:])
results_nn_mae.run("reports/nn_detector_cuts/")