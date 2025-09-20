#!/usr/bin/env python3
"""
Preprocess CSV files for diffusion model training.

This script implements the three-part preprocessing system:
1. Process individual CSV files to add precomputed moments and conditioning features
2. Combine all processed files into a single shuffled dataset
3. Prepare for training without on-the-fly computation
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append("src")
from reconstruction.diffusion.unified_config import load_unified_config


def calculate_moments(values, n_moments=4, eps=1e-8):
    """Calculate the first n moments of pT distributions."""
    values = np.asarray(values, dtype=np.float64)
    mu = np.mean(values)
    xc = values - mu
    sigma = np.sqrt(np.mean(xc**2) + eps)
    outs = [mu, sigma]
    for i in range(2, n_moments):
        outs += [np.mean(xc / (sigma + eps)) ** i]
    return np.array(outs, dtype=np.float64)


def circ_moments(alpha, k_max=2):
    """Calculate circular moments."""
    a = np.asarray(alpha, dtype=np.float64)
    a = (a + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.zeros(2 * k_max, dtype=np.float64)
    out = []
    for k in range(1, k_max + 1):
        out += [np.mean(np.cos(k * a)), np.mean(np.sin(k * a))]
    return np.array(out, dtype=np.float64)


def convert_to_four_vectors(data):
    """Convert detector-level data to four-vector format."""
    # Extract lepton four-vectors (already in E, px, py, pz format)
    lep1_4vec = data[["p_l_1_E", "p_l_1_x", "p_l_1_y", "p_l_1_z"]].values
    lep2_4vec = data[["p_l_2_E", "p_l_2_x", "p_l_2_y", "p_l_2_z"]].values

    # Missing momentum (assume pz = 0 for missing momentum, calculate E from pT)
    mpx = data["mpx"].values
    mpy = data["mpy"].values
    mpt = np.sqrt(mpx**2 + mpy**2)

    # For missing momentum, we set E = pT and pz = 0 as initial approximation
    missing_4vec = np.column_stack([mpt, mpx, mpy, np.zeros_like(mpx)])

    return lep1_4vec, lep2_4vec, missing_4vec


def compute_conditioning_features_for_file(df, config):
    """Compute conditioning features for an entire file (process)."""
    # Check if moment conditioning is enabled (all moment values are zero)
    if not getattr(config, "moment_conditioning_enabled", True):
        return {}
    # Convert to four-vectors
    lep1_4vec, lep2_4vec, missing_4vec = convert_to_four_vectors(df)

    # Calculate pT for leptons
    px1, py1 = lep1_4vec[:, 1], lep1_4vec[:, 2]
    px2, py2 = lep2_4vec[:, 1], lep2_4vec[:, 2]
    lep1_pt = np.hypot(px1, py1)
    lep2_pt = np.hypot(px2, py2)

    # Compute pT moments for the entire file using original function
    lep1_pt_moments = calculate_moments(lep1_pt, config.pt_conditioning_moments)
    lep2_pt_moments = calculate_moments(lep2_pt, config.pt_conditioning_moments)

    # Compute angle features
    pz1, pz2 = lep1_4vec[:, 3], lep2_4vec[:, 3]
    phi1 = np.arctan2(py1, px1)
    phi2 = np.arctan2(py2, px2)
    dphi = np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))

    eta1 = 0.5 * np.log(
        (np.sqrt(px1**2 + py1**2 + pz1**2) + pz1)
        / (np.sqrt(px1**2 + py1**2 + pz1**2) - pz1 + 1e-8)
    )
    eta2 = 0.5 * np.log(
        (np.sqrt(px2**2 + py2**2 + pz2**2) + pz2)
        / (np.sqrt(px2**2 + py2**2 + pz2**2) - pz2 + 1e-8)
    )
    deta = eta1 - eta2
    deta = (deta + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi)

    # Compute circular moments for angles
    eta_features = circ_moments(deta, config.eta_conditioning_moments)
    phi_features = circ_moments(dphi, config.phi_conditioning_moments)

    # Build conditioning feature dictionary
    conditioning_dict = {}

    # Add pT moments with proper naming (original format returns array)
    for i, value in enumerate(lep1_pt_moments):
        conditioning_dict[f"mom_pT_lep1_{i}"] = value

    for i, value in enumerate(lep2_pt_moments):
        conditioning_dict[f"mom_pT_lep2_{i}"] = value

    # Add circular moment features
    for i, val in enumerate(eta_features):
        conditioning_dict[f"eta_moment_{i}"] = val

    for i, val in enumerate(phi_features):
        conditioning_dict[f"phi_moment_{i}"] = val

    return conditioning_dict


def filter_zero_rows(detector_df, truth_df, max_zero_columns=2):
    """
    Remove rows from both detector and truth dataframes where more than max_zero_columns contain 0.0.
    Maintains row correspondence between detector and truth data.
    """
    # Count zeros in each row for detector data
    detector_zero_counts = (detector_df == 0.0).sum(axis=1)

    # Create mask for rows with max_zero_columns or fewer zeros
    valid_rows_mask = detector_zero_counts <= max_zero_columns

    # Apply the same mask to both dataframes
    filtered_detector = detector_df[valid_rows_mask].reset_index(drop=True)
    filtered_truth = truth_df[valid_rows_mask].reset_index(drop=True)

    removed_count = len(detector_df) - len(filtered_detector)
    print(f"  Filtered out {removed_count} rows with >{max_zero_columns} zero values")

    return filtered_detector, filtered_truth


def preprocess_csv_file_pair(
    detector_path, truth_path, config, moment_feature="pT", extra_conditioners=None
):
    """Process a pair of detector and truth CSV files in a coordinated manner."""

    # Load both CSV files
    detector_df = pd.read_csv(detector_path)
    truth_df = pd.read_csv(truth_path)

    print(f"Loaded detector: {len(detector_df)} rows, truth: {len(truth_df)} rows")

    # Ensure both files have the same number of rows
    if len(detector_df) != len(truth_df):
        raise ValueError(
            f"Detector and truth files have different row counts: {len(detector_df)} vs {len(truth_df)}"
        )

    # Apply coordinated filtering using configuration parameter
    data_config = config.get_data_config()
    detector_df, truth_df = filter_zero_rows(
        detector_df, truth_df, data_config["max_zero_columns"]
    )

    # Infer process name from detector filename
    process_name = Path(detector_path).stem
    if "_final_detector_sim" in process_name:
        process_name = process_name.replace("_final_detector_sim", "")
    if "_detector_sim_cuts" in process_name:
        process_name = process_name.replace("_detector_sim_cuts", "")
    if "_detector_sim" in process_name:
        process_name = process_name.replace("_detector_sim", "")
    if "_1M_MG5" in process_name:
        process_name = process_name.replace("_1M_MG5", "")
    if "_1M_MG" in process_name:
        process_name = process_name.replace("_1M_MG", "")

    # Add process column to both dataframes
    detector_df["process"] = process_name
    truth_df["process"] = process_name

    # Compute conditioning features for the detector file
    conditioning_dict = compute_conditioning_features_for_file(detector_df, config)

    # Add conditioning features as constant columns to detector data
    for feature_name, feature_value in conditioning_dict.items():
        detector_df[feature_name] = feature_value

    # Apply extra conditioners if provided
    if extra_conditioners is not None:
        extra_features = extra_conditioners(detector_df)
        for feature_name, feature_series in extra_features.items():
            detector_df[feature_name] = feature_series

    # Generate output filenames
    detector_output_path = detector_path.replace(".csv", "_diffusion_input.csv")
    truth_output_path = truth_path.replace(".csv", "_diffusion_target.csv")

    # Save processed files
    detector_df.to_csv(detector_output_path, index=False)
    truth_df.to_csv(truth_output_path, index=False)

    print(
        f"Processed {process_name}: {len(detector_df)} rows -> {detector_output_path}, {truth_output_path}"
    )

    return detector_output_path, truth_output_path


def process_csv_file_pairs(
    detector_csv_files,
    truth_csv_files,
    config,
    moment_feature="pT",
    extra_conditioners=None,
):
    """
    Process multiple pairs of detector and truth CSV files in a coordinated manner.

    Parameters:
        detector_csv_files: List of detector CSV file paths
        truth_csv_files: List of corresponding truth CSV file paths
        config: DiffusionConfig object
        moment_feature: Feature to compute moments for (default: 'pT')
        extra_conditioners: Optional callable for additional conditioning features

    Returns:
        Tuple of (detector_output_files, truth_output_files)
    """
    detector_output_files = []
    truth_output_files = []

    if len(detector_csv_files) != len(truth_csv_files):
        raise ValueError(
            f"Number of detector files ({len(detector_csv_files)}) must match number of truth files ({len(truth_csv_files)})"
        )

    print(f"Processing {len(detector_csv_files)} CSV file pairs...")

    for detector_file, truth_file in zip(detector_csv_files, truth_csv_files):
        if not os.path.exists(detector_file):
            print(f"Warning: Detector file {detector_file} not found, skipping pair...")
            continue
        if not os.path.exists(truth_file):
            print(f"Warning: Truth file {truth_file} not found, skipping pair...")
            continue

        try:
            detector_output, truth_output = preprocess_csv_file_pair(
                detector_file,
                truth_file,
                config,
                moment_feature=moment_feature,
                extra_conditioners=extra_conditioners,
            )
            detector_output_files.append(detector_output)
            truth_output_files.append(truth_output)

        except Exception as e:
            print(f"Error processing pair {detector_file}, {truth_file}: {e}")
            continue

    print(f"Successfully processed {len(detector_output_files)} file pairs")
    return detector_output_files, truth_output_files


def combine_and_shuffle_datasets(
    detector_input_files=None,
    truth_input_files=None,
    detector_output_path="diffusion_training_dataset.csv",
    truth_output_path="diffusion_training_targets.csv",
    max_rows=None,
    random_state=42,
):
    """
    Combine all processed detector and truth CSV files into shuffled training datasets.
    Maintains correspondence between detector and truth data through coordinated shuffling.

    Parameters:
        detector_input_files: List of detector input CSV files (if None, auto-discover *_diffusion_input.csv)
        truth_input_files: List of truth input CSV files (if None, auto-discover *_diffusion_target.csv)
        detector_output_path: Path for the combined detector dataset
        truth_output_path: Path for the combined truth dataset
        max_rows: Optional maximum number of rows (with stratified sampling)
        random_state: Random seed for shuffling

    Returns:
        Tuple of (detector_dataset_path, truth_dataset_path)
    """
    print("Combining and shuffling detector and truth datasets...")

    # Auto-discover files if not provided
    if detector_input_files is None:
        detector_input_files = list(Path(".").glob("**/*_diffusion_input.csv"))
        detector_input_files = [str(f) for f in detector_input_files]

    if truth_input_files is None:
        truth_input_files = list(Path(".").glob("**/*_diffusion_target.csv"))
        truth_input_files = [str(f) for f in truth_input_files]

    if not detector_input_files:
        raise ValueError("No *_diffusion_input.csv files found")

    if not truth_input_files:
        raise ValueError("No *_diffusion_target.csv files found")

    if len(detector_input_files) != len(truth_input_files):
        raise ValueError(
            f"Number of detector files ({len(detector_input_files)}) must match number of truth files ({len(truth_input_files)})"
        )

    print(
        f"Found {len(detector_input_files)} detector files and {len(truth_input_files)} truth files to combine"
    )

    # Load and combine all files
    detector_dataframes = []
    truth_dataframes = []
    process_counts = {}

    for detector_path, truth_path in zip(detector_input_files, truth_input_files):
        try:
            detector_df = pd.read_csv(detector_path)
            truth_df = pd.read_csv(truth_path)

            # Ensure same number of rows
            if len(detector_df) != len(truth_df):
                print(
                    f"  Warning: Row count mismatch in {detector_path} ({len(detector_df)}) vs {truth_path} ({len(truth_df)})"
                )
                continue

            detector_dataframes.append(detector_df)
            truth_dataframes.append(truth_df)

            # Count rows per process
            if "process" in detector_df.columns:
                process_name = detector_df["process"].iloc[0]
                process_counts[process_name] = len(detector_df)

        except Exception as e:
            print(f"  Error loading {detector_path}, {truth_path}: {e}")
            continue

    if not detector_dataframes or not truth_dataframes:
        raise ValueError("No valid dataframes loaded")

    # Combine all dataframes
    combined_detector_df = pd.concat(detector_dataframes, axis=0, ignore_index=True)
    combined_truth_df = pd.concat(truth_dataframes, axis=0, ignore_index=True)
    print(f"\nCombined datasets: {len(combined_detector_df)} total rows each")

    # Apply stratified downsampling if requested
    if max_rows is not None and len(combined_detector_df) > max_rows:
        print(f"Downsampling to {max_rows} rows with stratified sampling...")

        if "process" in combined_detector_df.columns:
            # Stratified sampling by process
            sampled_detector_dfs = []
            sampled_truth_dfs = []
            processes = combined_detector_df["process"].unique()
            rows_per_process = max_rows // len(processes)

            for process in processes:
                detector_process_df = combined_detector_df[
                    combined_detector_df["process"] == process
                ]
                truth_process_df = combined_truth_df[
                    combined_truth_df["process"] == process
                ]

                if len(detector_process_df) > rows_per_process:
                    # Sample the same indices for both detector and truth
                    sample_indices = detector_process_df.sample(
                        n=rows_per_process, random_state=random_state
                    ).index
                    detector_sample = detector_process_df.loc[sample_indices]
                    truth_sample = truth_process_df.loc[sample_indices]
                else:
                    detector_sample = detector_process_df
                    truth_sample = truth_process_df

                sampled_detector_dfs.append(detector_sample)
                sampled_truth_dfs.append(truth_sample)

            combined_detector_df = pd.concat(
                sampled_detector_dfs, axis=0, ignore_index=True
            )
            combined_truth_df = pd.concat(sampled_truth_dfs, axis=0, ignore_index=True)
        else:
            # Simple random sampling with same indices
            sample_indices = combined_detector_df.sample(
                n=max_rows, random_state=random_state
            ).index
            combined_detector_df = combined_detector_df.loc[sample_indices].reset_index(
                drop=True
            )
            combined_truth_df = combined_truth_df.loc[sample_indices].reset_index(
                drop=True
            )

        print(f"After downsampling: {len(combined_detector_df)} rows each")

    # Generate coordinated shuffle indices
    shuffle_indices = np.arange(len(combined_detector_df))
    np.random.seed(random_state)
    np.random.shuffle(shuffle_indices)

    # Apply the same shuffle to both datasets
    combined_detector_df = combined_detector_df.iloc[shuffle_indices].reset_index(
        drop=True
    )
    combined_truth_df = combined_truth_df.iloc[shuffle_indices].reset_index(drop=True)

    # Save the final datasets
    combined_detector_df.to_csv(detector_output_path, index=False)
    combined_truth_df.to_csv(truth_output_path, index=False)

    # Print summary report
    print(f"Combined detector dataset saved: {detector_output_path}")
    print(f"Combined truth dataset saved: {truth_output_path}")
    print(
        f"Total rows: {len(combined_detector_df)}, Detector columns: {len(combined_detector_df.columns)}, Truth columns: {len(combined_truth_df.columns)}"
    )

    if "process" in combined_detector_df.columns:
        final_process_counts = (
            combined_detector_df["process"].value_counts().sort_index()
        )
        print(f"Rows per process: {dict(final_process_counts)}")

    return detector_output_path, truth_output_path


def main(config_path="diffusion_config.yaml"):
    """Main function to run the preprocessing pipeline."""

    # Load unified configuration
    config = load_unified_config(config_path)

    print("DIFFUSION DATA PREPROCESSING PIPELINE")
    print("Using unified configuration system")
    config.print_summary()

    # Get file paths from configuration
    file_paths = config.get_file_paths()
    detector_csv_files = file_paths["detector_files"]
    truth_csv_files = file_paths["truth_files"]

    if not detector_csv_files or not truth_csv_files:
        print("Error: No input files specified in configuration")
        return

    print(f"Processing {len(detector_csv_files)} detector-truth file pairs")
    print(
        "Processing detector-truth file pairs with coordinated filtering and shuffling"
    )

    # Step 1: Process individual CSV file pairs
    detector_output_files, truth_output_files = process_csv_file_pairs(
        detector_csv_files, truth_csv_files, config
    )

    if not detector_output_files or not truth_output_files:
        print("No file pairs were successfully processed. Exiting.")
        return

    # Get output paths from configuration
    detector_output_path = file_paths["detector_dataset_path"]
    truth_output_path = file_paths["truth_dataset_path"]

    # Get data processing parameters
    data_config = config.get_data_config()

    # Step 2: Combine and shuffle datasets with coordinated shuffling
    detector_dataset, truth_dataset = combine_and_shuffle_datasets(
        detector_input_files=detector_output_files,
        truth_input_files=truth_output_files,
        detector_output_path=detector_output_path,
        truth_output_path=truth_output_path,
        max_rows=config.data_processing.get("max_rows"),
        random_state=data_config["random_state"],
    )

    print(f"Preprocessing complete!")
    print(f"Final detector dataset: {detector_dataset}")
    print(f"Final truth dataset: {truth_dataset}")
    print(
        "Detector and truth datasets maintain perfect row correspondence for diffusion training."
    )


if __name__ == "__main__":
    main()
