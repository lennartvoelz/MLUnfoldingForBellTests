from src.evaluation.evaluation import calculate_results, calculate_results_diff_analysis
import numpy as np
import pandas as pd

data_neutrino = pd.read_csv(
    "outputs/diffusion_pt_all_unfold_mangled/unfold_diffusion_lepqua_CT14lo.csv"
)
data_leptons = pd.read_csv("data/hww_sherpa_1M_MG_final_truth_cuts.csv")

data = pd.concat([data_neutrino, data_leptons], axis=1)

X = data[
    [
        "p_l_1_E_truth",
        "p_l_1_x_truth",
        "p_l_1_y_truth",
        "p_l_1_z_truth",
        "p_l_2_E_truth",
        "p_l_2_x_truth",
        "p_l_2_y_truth",
        "p_l_2_z_truth",
    ]
]

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
]

types = data["Event.Type"]

final_truth = np.concatenate((X, y), axis=1)

y_unfolded_ww_tt_hwwMG = data[
    [
        "p_v_1_E_diffusion",
        "p_v_1_x_diffusion",
        "p_v_1_y_diffusion",
        "p_v_1_z_diffusion",
        "p_v_2_E_diffusion",
        "p_v_2_x_diffusion",
        "p_v_2_y_diffusion",
        "p_v_2_z_diffusion",
    ]
]

final_unfolded_ww_tt_hwwMG = np.concatenate((X, y_unfolded_ww_tt_hwwMG), axis=1)

result = calculate_results(
    arrays=[final_truth, final_unfolded_ww_tt_hwwMG],
    labels=["Truth", "Diffusion"],
    title="HWW",
    types=[types, types],
)

result_ = calculate_results_diff_analysis(
    arrays=[final_truth, final_unfolded_ww_tt_hwwMG],
    labels=["Truth", "Diffusion"],
    title="HWW",
    types=[types, types],
)

result.run("plots/analysis")
result_.run("plots/analysis")