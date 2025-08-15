from src.evaluation.evaluation import calculate_results, calculate_results_diff_analysis
import numpy as np
import pandas as pd

data = pd.read_csv("data/hww_1M_MG_final_truth.csv")
length = len(data)
subset_size = length
efficiency_map_data = data[: subset_size // 2]

X_eff = efficiency_map_data[
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
y_eff = efficiency_map_data[
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
types_eff = efficiency_map_data["Event.Type"].copy()
final_state_eff = np.concatenate((X_eff, y_eff), axis=1)

# Load cuts data and split
data_cuts = pd.read_csv("data/hww_1M_MG_final_truth_cuts.csv")
length_cuts = len(data_cuts)
subset_size_cuts = length_cuts
efficiency_map_cuts = data_cuts[: subset_size_cuts // 2]
measurement_cuts = data_cuts[subset_size_cuts // 2 : subset_size_cuts]

# Process cuts data for efficiency map
X_cuts_eff = efficiency_map_cuts[
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
y_cuts_eff = efficiency_map_cuts[
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
types_cuts_eff = efficiency_map_cuts["Event.Type"].copy()
final_state_cuts_eff = np.concatenate((X_cuts_eff, y_cuts_eff), axis=1)

# Process cuts data for measurement
X_cuts_meas = measurement_cuts[
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
y_cuts_meas = measurement_cuts[
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
types_cuts_meas = measurement_cuts["Event.Type"].copy()
final_state_cuts_meas = np.concatenate((X_cuts_meas, y_cuts_meas), axis=1)

result = calculate_results_diff_analysis(
    arrays=[final_state_eff, final_state_cuts_eff],
    labels=["Truth", "Cuts"],
    title="HWW",
    types=[types_eff, types_cuts_eff],
)

result_ = calculate_results(
    arrays=[final_state_eff, final_state_cuts_eff],
    labels=["Truth", "Cuts"],
    title="HWW",
    types=[types_eff, types_cuts_eff],
)

result_.run("plots")

result.initialize_datasets()

efficiency_map, cos_edges, phi_edges = result.efficiency_map()

# data_ww = pd.read_csv("data/ww_1M_MG_final_truth_cuts.csv")
# length_cuts = len(data_ww)
# subset_size_cuts = length_cuts
# measurement_cuts = data_ww[subset_size_cuts // 2 : subset_size_cuts]

# X_cuts_meas = measurement_cuts[
#     [
#         "p_l_1_E_truth",
#         "p_l_1_x_truth",
#         "p_l_1_y_truth",
#         "p_l_1_z_truth",
#         "p_l_2_E_truth",
#         "p_l_2_x_truth",
#         "p_l_2_y_truth",
#         "p_l_2_z_truth",
#     ]
# ].copy()
# y_cuts_meas = measurement_cuts[
#     [
#         "p_v_1_E_truth",
#         "p_v_1_x_truth",
#         "p_v_1_y_truth",
#         "p_v_1_z_truth",
#         "p_v_2_E_truth",
#         "p_v_2_x_truth",
#         "p_v_2_y_truth",
#         "p_v_2_z_truth",
#     ]
# ].copy()
# types_cuts_meas = measurement_cuts["Event.Type"].copy()
# final_state_cuts_meas = np.concatenate((X_cuts_meas, y_cuts_meas), axis=1)

# Get weights for measurement data
mea_res = calculate_results_diff_analysis(
    arrays=[final_state_cuts_meas],
    labels=["Cuts"],
    title="WW",
    types=[types_cuts_meas],
)

mea_res.initialize_datasets()

weights, _ = mea_res.get_efficiency_weights_4d(
    mea_res.angles_[0][:, 2],
    mea_res.angles_[0][:, 0],
    mea_res.angles_[0][:, 3],
    mea_res.angles_[0][:, 1],
    efficiency_map,
    [cos_edges, phi_edges, cos_edges, phi_edges],
)

# Apply weights to measurement data
cov_weighted = mea_res.compute_weighted_covariance(mea_res.pW1[0], mea_res.pW2[0], weights)

# Plot weighted Gell-Mann matrix
result.plot_gellmann_weighted(cov_weighted, "plots")


# Create a csv file with final state, types and weights
# df = pd.DataFrame(
#     final_state_cuts_meas,
#     columns=[
#         "p_l_1_E_truth",
#         "p_l_1_x_truth",
#         "p_l_1_y_truth",
#         "p_l_1_z_truth",
#         "p_l_2_E_truth",
#         "p_l_2_x_truth",
#         "p_l_2_y_truth",
#         "p_l_2_z_truth",
#         "p_v_1_E_truth",
#         "p_v_1_x_truth",
#         "p_v_1_y_truth",
#         "p_v_1_z_truth",
#         "p_v_2_E_truth",
#         "p_v_2_x_truth",
#         "p_v_2_y_truth",
#         "p_v_2_z_truth",
#     ],
# )
# df["type"] = types_cuts_meas
# df["weight"] = weights_minus
# df.to_csv("data/hww_cuts_reweighted.csv", index=False)
