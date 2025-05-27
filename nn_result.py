from src.evaluation.evaluation import calculate_results
from src.evaluation.calculate_mae import results
import numpy as np

final_state = np.loadtxt("/mnt/c/Users/Lennart/Desktop/Studium/MLUnfoldingForBellTests/data/dnn_final_state_ww.csv")
final_state_truth = np.loadtxt("/mnt/c/Users/Lennart/Desktop/Studium/MLUnfoldingForBellTests/data/dnn_final_state_ww_truth.csv")

results_nn = calculate_results([final_state, final_state_truth], ["NN Regression", "Truth"], "DNN WW No Cuts")
results_nn.run("reports/nn_ww/")

# results_nn_mae = results(final_state_truth[:,8:], final_state[:,8:])
# results_nn_mae.run("reports/nn_cuts_ww/")