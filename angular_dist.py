from src.evaluation.evaluation import calculate_results, calculate_results_diff_analysis
import numpy as np
import pandas as pd

data = pd.read_csv("data/hww_1M_MG5_final_truth.csv")
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
].copy()
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
].copy()
types = data["Event.Type"].copy()

final_state = np.concatenate((X, y), axis=1)

data_cuts = pd.read_csv("data/hww_1M_MG5_final_truth_cuts.csv")
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
].copy()
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
].copy()
types_cuts = data_cuts["Event.Type"].copy()

final_state_cuts = np.concatenate((X_cuts, y_cuts), axis=1)

result = calculate_results(
    arrays=[final_state, final_state_cuts],
    labels=["Truth", "Cuts"],
    title="Cuts Analysis",
    types=[types, types_cuts],
)
result.run("reports/")
diff_result = calculate_results_diff_analysis(
    arrays=[final_state, final_state_cuts],
    labels=["Truth", "Cuts"],
    title="Diff Analysis",
    types=[types, types_cuts],
)
diff_result.run("reports/")