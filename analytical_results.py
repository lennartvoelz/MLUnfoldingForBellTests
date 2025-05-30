from src.evaluation.evaluation import calculate_results
from src.evaluation.calculate_mae import results
from src.reconstruction.analytical_reconstruction import Baseline
from src.utils.lorentz_vector import LorentzVector
from src.data_preproc.preprocessing import DataPreprocessor
import yaml
import numpy as np

config = yaml.safe_load(open('config.yaml'))

data = DataPreprocessor(data_path=config['data_path'], raw_data_path=config['raw_data_path'], cuts=False, splits=False, drop_zeroes=True)

X, y, types = data.run_preprocessing()

# lep0 = X[:100000,:4]
# lep1 = X[:100000,4:8]
# mpx = X[:100000,8]
# mpy = X[:100000,9]

# y = y[:100000]

lep0 = X[:,:4]
lep1 = X[:,4:8]
mpx = X[:,8]
mpy = X[:,9]


nu_1 = np.zeros((len(lep0), 4))
nu_2 = np.zeros((len(lep0), 4))

for i in range(len(lep0)):
    lep0_event = LorentzVector(lep0[i])
    lep1_event = LorentzVector(lep1[i])

    analytical_results = Baseline(lep0_event, lep1_event, mpx[i], mpy[i])

    nu_1[i], nu_2[i] = analytical_results.restrictions()

final_state = np.concatenate((lep0, lep1, nu_1, nu_2), axis=1)
print(final_state.shape)
y_pred = np.concatenate((nu_1, nu_2), axis=1)
print(y_pred.shape)

final_state_truth = np.concatenate((lep0, lep1, y), axis=1)

mae = results(y, y_pred)

print(mae.run("reports/analytic_ww_cuts/"))

results_analytic = calculate_results([final_state, final_state_truth], ["Analytic Reconstruction", "Truth"], "Analytic Reconstruction Detector Simulation", types)
results_analytic.run("reports/analytic/")