from src.evaluation.evaluation import calculate_results
from src.reconstruction.analytical_reconstruction import Baseline
from src.utils.lorentz_vector import LorentzVector
from src.data_preproc.preprocessing import DataPreprocessor
import yaml
import numpy as np

config = yaml.safe_load(open('config.yaml'))

data = DataPreprocessor(data_path=config['data_path'], cuts=False, splits=False)
data_cuts = DataPreprocessor(data_path=config['data_path'], cuts=True, splits=False)

X, y = data.run_preprocessing()
X_cuts, y_cuts = data_cuts.run_preprocessing()

X = X[:,:8]
X_cuts = X_cuts[:,:8]

final_state = np.concatenate((X, y), axis=1)
final_state_cuts = np.concatenate((X_cuts, y_cuts), axis=1)

results = calculate_results([final_state], ["Truth"], "HWW Truth")
results.run("reports/truth_ww/")