from src.evaluation.evaluation import calculate_results, calculate_results_diff_analysis
import numpy as np
import pandas as pd

data = pd.read_csv("data/mangled_hww_1M_MG_final_truth_cuts.csv")
data_cuts = pd.read_csv("data/hww_1M_MG_final_truth_cuts.csv")

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
final = np.concatenate((X, y), axis=1)

X_cuts = data_cuts[
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

y_cuts = data_cuts[
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

types_cuts = data_cuts["Event.Type"]
final_cuts = np.concatenate((X_cuts, y_cuts), axis=1)

result = calculate_results(
    arrays=[final, final_cuts],
    labels=["No Cuts", "Cuts"],
    title="HWW",
    types=[types, types_cuts],
)

result_ = calculate_results_diff_analysis(
    arrays=[final, final_cuts],
    labels=["No Cuts", "Cuts"],
    title="HWW",
    types=[types, types_cuts],
)

result.run("plots/analysis")
result_.run("plots/analysis")