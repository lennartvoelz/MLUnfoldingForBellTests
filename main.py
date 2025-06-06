from src.evaluation.evaluation import calculate_results, calculate_results_diff_analysis
from src.reconstruction.analytical_reconstruction import Baseline
from src.utils.lorentz_vector import LorentzVector
from src.data_preproc.preprocessing import DataPreprocessor
import yaml
import numpy as np

config = yaml.safe_load(open('config.yaml'))

data = DataPreprocessor(data_path=config['data_path'], raw_data_path=config['raw_data_path'], truth_path=config['truth_path'],
                        cuts=True, splits=False, drop_zeroes=True)

X, y, types = data.run_preprocessing()