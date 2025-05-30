from src.evaluation.evaluation import calculate_results_diff_analysis
from src.reconstruction.analytical_reconstruction import Baseline
from src.utils.lorentz_vector import LorentzVector
from src.data_preproc.preprocessing import DataPreprocessor
import yaml
import numpy as np

config = yaml.safe_load(open('config.yaml'))

data = DataPreprocessor(data_path=config['data_path'], raw_data_path=config['raw_data_path'], cuts=False, splits=False)

X, y, types = data.run_preprocessing()

X = X[:,:8]
X_cuts = X_cuts[:,:8]

final_state = np.concatenate((X, y), axis=1)
final_state_cuts = np.concatenate((X_cuts, y_cuts), axis=1)

results_truth = calculate_results([final_state], ["Truth"], "Truth Detector Simulation", types)
results_truth.run("reports/truth_detector/")