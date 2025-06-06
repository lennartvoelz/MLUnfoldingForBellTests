from src.evaluation.evaluation import calculate_results, calculate_results_diff_analysis
from src.reconstruction.analytical_reconstruction import Baseline
from src.utils.lorentz_vector import LorentzVector
from src.data_preproc.preprocessing import DataPreprocessor
import yaml
import numpy as np

config = yaml.safe_load(open('config.yaml'))

data = DataPreprocessor(data_path=config['truth_path'], raw_data_path=config['raw_data_path'],
                        cuts=False, splits=False, drop_zeroes=True)

X, y, types = data.run_preprocessing()

X = X[:,:8]

final_state = np.concatenate((X, y), axis=1)

results_truth = calculate_results([final_state], ["Truth"], "Truth", types)
results_truth.run("reports/truth_detector/")