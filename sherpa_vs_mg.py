from src.evaluation.evaluation import calculate_results
import pandas as pd

data_mg = pd.read_csv("data/hww_1M_MG_final_truth.csv")
data_sherpa = pd.read_csv("data/hww_sherpa_1M_MG_final_truth.csv")
data_mg_mangled = pd.read_csv("data/mangled_hww_1M_MG_final_truth.csv")

columns = [
    "p_l_1_E_truth",
    "p_l_1_x_truth",
    "p_l_1_y_truth",
    "p_l_1_z_truth",
    "p_l_2_E_truth",
    "p_l_2_x_truth",
    "p_l_2_y_truth",
    "p_l_2_z_truth",
    "p_v_1_E_truth",
    "p_v_1_x_truth",
    "p_v_1_y_truth",
    "p_v_1_z_truth",
    "p_v_2_E_truth",
    "p_v_2_x_truth",
    "p_v_2_y_truth",
    "p_v_2_z_truth",
]

# Filter out rows with NaN values for MG
mg = data_mg[columns + ["Event.Type"]]
mg_valid_mask = ~mg[columns].isna().any(axis=1)
mg = mg[mg_valid_mask]
print(
    f"MG: {mg_valid_mask.sum()} valid rows out of {len(data_mg)} ({100 * mg_valid_mask.sum() / len(data_mg):.1f}%)"
)
types_mg = mg["Event.Type"]
mg = mg[columns].to_numpy()

# Filter out rows with NaN values for Sherpa
sherpa = data_sherpa[columns + ["Event.Type"]]
sherpa_valid_mask = ~sherpa[columns].isna().any(axis=1)
sherpa = sherpa[sherpa_valid_mask]
print(
    f"Sherpa: {sherpa_valid_mask.sum()} valid rows out of {len(data_sherpa)} ({100 * sherpa_valid_mask.sum() / len(data_sherpa):.1f}%)"
)
types_sherpa = sherpa["Event.Type"]
sherpa = sherpa[columns].to_numpy()

# Filter out rows with NaN values for MG Mangled
mg_mangled = data_mg_mangled[columns + ["Event.Type"]]
mg_mangled_valid_mask = ~mg_mangled[columns].isna().any(axis=1)
mg_mangled = mg_mangled[mg_mangled_valid_mask]
print(
    f"MG Mangled: {mg_mangled_valid_mask.sum()} valid rows out of {len(data_mg_mangled)} ({100 * mg_mangled_valid_mask.sum() / len(data_mg_mangled):.1f}%)"
)
types_mg_mangled = mg_mangled["Event.Type"]
mg_mangled = mg_mangled[columns].to_numpy()

result = calculate_results(
    arrays=[mg, sherpa, mg_mangled],
    labels=["MG", "Sherpa", "MG Mangled"],
    title="HWW",
    types=[types_mg, types_sherpa, types_mg_mangled],
)

result.run("plots")
