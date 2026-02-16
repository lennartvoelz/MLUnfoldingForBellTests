from src.evaluation.evaluation import calculate_results
from src.evaluation.calculate_mae import results
from src.reconstruction.analytical_reconstruction import Baseline
from src.utils.lorentz_vector import LorentzVector
import pandas as pd
import yaml
import numpy as np

config = yaml.safe_load(open('config.yaml'))

data = pd.read_csv("data/hww_sherpa_1M_MG_final_truth_cuts.csv")
data_mp = pd.read_csv("data/hww_sherpa_1M_MG_final_detector_sim_cuts.csv")

lep0 = data[
    [
        "p_l_1_E_truth",
        "p_l_1_x_truth",
        "p_l_1_y_truth",
        "p_l_1_z_truth",
    ]
].to_numpy()
lep1 = data[
    [
        "p_l_2_E_truth",
        "p_l_2_x_truth",
        "p_l_2_y_truth",
        "p_l_2_z_truth",
    ]
].to_numpy()
y = data[
    [
        "p_v_1_E_truth",
        "p_v_1_x_truth",
        "p_v_1_y_truth",
        "p_v_1_z_truth",
        "p_v_2_E_truth",
        "p_v_2_x_truth",
        "p_v_2_y_truth",
        "p_v_2_z_truth",
    ]
].to_numpy()
mpx = data_mp["mpx"].to_numpy()
mpy = data_mp["mpy"].to_numpy()
types = data["Event.Type"]

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

results_analytic = calculate_results([final_state, final_state_truth], ["Analytic Reconstruction", "Truth"], "Analytic Reconstruction Detector Simulation", [types, types])
results_analytic.run("reports/analytic/")