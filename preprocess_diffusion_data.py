#!/usr/bin/env python3
"""Preprocess CSV files for diffusion model training."""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append("src")
from reconstruction.diffusion.unified_config import load_unified_config

logger = logging.getLogger(__name__)


def calculate_moments(values, n_moments=4, eps=1e-8):
    """
    Calculate the first n moments of distributions.

    Parameters:
        values: Array of values
        n_moments: Number of moments to calculate (0 = none, 1 = mean only, 2 = mean+std, etc.)
        eps: Small value for numerical stability

    Returns:
        Array of moment values (empty array if n_moments=0)
    """
    # Return empty array if no moments requested
    if n_moments == 0:
        return np.array([], dtype=np.float64)

    values = np.asarray(values, dtype=np.float64)
    mu = np.mean(values)

    # Return just mean if n_moments=1
    if n_moments == 1:
        return np.array([mu], dtype=np.float64)

    # Calculate std and higher moments
    xc = values - mu
    sigma = np.sqrt(np.mean(xc**2) + eps)
    outs = [mu, sigma]
    for i in range(2, n_moments):
        outs += [np.mean(xc / (sigma + eps)) ** i]
    return np.array(outs, dtype=np.float64)


def calculate_total_transverse_mass(lep1_4vec, lep2_4vec, missing_4vec):
    """
    Calculate total transverse mass of the system as sum of individual transverse masses.

    The total transverse mass is computed as:
        m_T_total = m_T_lep1 + m_T_lep2 + m_T_missing

    where each individual transverse mass is:
        m_T_i = sqrt(E_i^2 - p_z_i^2)

    Parameters:
        lep1_4vec: Four-vector of first lepton [E, px, py, pz]
        lep2_4vec: Four-vector of second lepton [E, px, py, pz]
        missing_4vec: Four-vector of missing momentum [E, px, py, pz]

    Returns:
        Total transverse mass of the system (sum of individual transverse masses)
    """
    # Extract energy and pz components for each particle
    E1, pz1 = lep1_4vec[:, 0], lep1_4vec[:, 3]
    E2, pz2 = lep2_4vec[:, 0], lep2_4vec[:, 3]
    E_miss, pz_miss = missing_4vec[:, 0], missing_4vec[:, 3]

    # Calculate individual transverse masses: m_T_i = sqrt(E_i^2 - p_z_i^2)
    mt_lep1_squared = E1**2 - pz1**2
    mt_lep2_squared = E2**2 - pz2**2
    mt_miss_squared = E_miss**2 - pz_miss**2

    # Ensure non-negative values before taking square root
    mt_lep1 = np.sqrt(np.maximum(mt_lep1_squared, 0.0))
    mt_lep2 = np.sqrt(np.maximum(mt_lep2_squared, 0.0))
    mt_miss = np.sqrt(np.maximum(mt_miss_squared, 0.0))

    # Sum individual transverse masses
    mt_total = mt_lep1 + mt_lep2 + mt_miss

    return mt_total


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

    # Get epsilon from config with fallback
    epsilon = getattr(config, "epsilon", 1e-8)

    eta1 = 0.5 * np.log(
        (np.sqrt(px1**2 + py1**2 + pz1**2) + pz1)
        / (np.sqrt(px1**2 + py1**2 + pz1**2) - pz1 + epsilon)
    )
    eta2 = 0.5 * np.log(
        (np.sqrt(px2**2 + py2**2 + pz2**2) + pz2)
        / (np.sqrt(px2**2 + py2**2 + pz2**2) - pz2 + epsilon)
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

    # Add optional m_t conditioning moments if enabled
    mt_moments_count = getattr(config, "mt_conditioning_moments", 0)
    if mt_moments_count > 0:
        mt_total = calculate_total_transverse_mass(lep1_4vec, lep2_4vec, missing_4vec)
        mt_moments = calculate_moments(mt_total, mt_moments_count)
        for i, value in enumerate(mt_moments):
            conditioning_dict[f"mom_mt_{i}"] = value

    # Add optional px conditioning moments if enabled
    px_moments_count = getattr(config, "px_conditioning_moments", 0)
    if px_moments_count > 0:
        px_1_moments = calculate_moments(px1, px_moments_count)
        px_2_moments = calculate_moments(px2, px_moments_count)
        for i, value in enumerate(px_1_moments):
            conditioning_dict[f"mom_px_lep1_{i}"] = value
        for i, value in enumerate(px_2_moments):
            conditioning_dict[f"mom_px_lep2_{i}"] = value

    # Add optional py conditioning moments if enabled
    py_moments_count = getattr(config, "py_conditioning_moments", 0)
    if py_moments_count > 0:
        py_1_moments = calculate_moments(py1, py_moments_count)
        py_2_moments = calculate_moments(py2, py_moments_count)
        for i, value in enumerate(py_1_moments):
            conditioning_dict[f"mom_py_lep1_{i}"] = value
        for i, value in enumerate(py_2_moments):
            conditioning_dict[f"mom_py_lep2_{i}"] = value

    return conditioning_dict


def filter_zero_rows(detector_df, truth_df, max_zero_columns=2):
    """Remove rows from both detector and truth dataframes where more than max_zero_columns contain 0.0."""
    detector_zero_counts = (detector_df == 0.0).sum(axis=1)
    valid_rows_mask = detector_zero_counts <= max_zero_columns

    filtered_detector = detector_df[valid_rows_mask].reset_index(drop=True)
    filtered_truth = truth_df[valid_rows_mask].reset_index(drop=True)

    removed_count = len(detector_df) - len(filtered_detector)
    logger.info(f"Filtered out {removed_count} rows with >{max_zero_columns} zero values")

    return filtered_detector, filtered_truth


def preprocess_csv_file_pair(
    detector_path, truth_path, config, moment_feature="pT", extra_conditioners=None
):
    """Process a pair of detector and truth CSV files in a coordinated manner."""

    detector_df = pd.read_csv(detector_path)
    truth_df = pd.read_csv(truth_path)

    if len(detector_df) != len(truth_df):
        raise ValueError(
            f"Detector and truth files have different row counts: {len(detector_df)} vs {len(truth_df)}"
        )

    data_config = config.get_data_config()
    detector_df, truth_df = filter_zero_rows(
        detector_df, truth_df, data_config["max_zero_columns"]
    )

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

    detector_df["process"] = process_name
    truth_df["process"] = process_name

    conditioning_dict = compute_conditioning_features_for_file(detector_df, config)

    for feature_name, feature_value in conditioning_dict.items():
        detector_df[feature_name] = feature_value

    if extra_conditioners is not None:
        extra_features = extra_conditioners(detector_df)
        for feature_name, feature_series in extra_features.items():
            detector_df[feature_name] = feature_series

    detector_output_path = detector_path.replace(".csv", "_diffusion_input.csv")
    truth_output_path = truth_path.replace(".csv", "_diffusion_target.csv")

    detector_df.to_csv(detector_output_path, index=False)
    truth_df.to_csv(truth_output_path, index=False)

    logger.info(f"Processed {process_name}: {len(detector_df)} rows -> {detector_output_path}, {truth_output_path}")

    return detector_output_path, truth_output_path


def process_csv_file_pairs(
    detector_csv_files,
    truth_csv_files,
    config,
    moment_feature="pT",
    extra_conditioners=None,
):
    """Process multiple pairs of detector and truth CSV files in a coordinated manner."""
    detector_output_files = []
    truth_output_files = []

    if len(detector_csv_files) != len(truth_csv_files):
        raise ValueError(
            f"Number of detector files ({len(detector_csv_files)}) must match number of truth files ({len(truth_csv_files)})"
        )

    logger.info(f"Processing {len(detector_csv_files)} CSV file pairs...")

    for detector_file, truth_file in zip(detector_csv_files, truth_csv_files):
        if not os.path.exists(detector_file):
            logger.warning(f"Detector file {detector_file} not found, skipping pair...")
            continue
        if not os.path.exists(truth_file):
            logger.warning(f"Truth file {truth_file} not found, skipping pair...")
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
            logger.error(f"Error processing pair {detector_file}, {truth_file}: {e}")
            continue

    logger.info(f"Successfully processed {len(detector_output_files)} file pairs")
    return detector_output_files, truth_output_files


def combine_and_shuffle_datasets(
    detector_input_files=None,
    truth_input_files=None,
    detector_output_path="diffusion_training_dataset.csv",
    truth_output_path="diffusion_training_targets.csv",
    max_rows=None,
    random_state=42,
):
    """Combine all processed detector and truth CSV files into shuffled training datasets."""
    logger.info("Combining and shuffling detector and truth datasets...")

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

    logger.info(f"Found {len(detector_input_files)} detector files and {len(truth_input_files)} truth files to combine")

    detector_dataframes = []
    truth_dataframes = []
    process_counts = {}

    for detector_path, truth_path in zip(detector_input_files, truth_input_files):
        try:
            detector_df = pd.read_csv(detector_path)
            truth_df = pd.read_csv(truth_path)

            if len(detector_df) != len(truth_df):
                logger.warning(f"Row count mismatch in {detector_path} ({len(detector_df)}) vs {truth_path} ({len(truth_df)})")
                continue

            detector_dataframes.append(detector_df)
            truth_dataframes.append(truth_df)

            if "process" in detector_df.columns:
                process_name = detector_df["process"].iloc[0]
                process_counts[process_name] = len(detector_df)

        except Exception as e:
            logger.error(f"Error loading {detector_path}, {truth_path}: {e}")
            continue

    if not detector_dataframes or not truth_dataframes:
        raise ValueError("No valid dataframes loaded")

    combined_detector_df = pd.concat(detector_dataframes, axis=0, ignore_index=True)
    combined_truth_df = pd.concat(truth_dataframes, axis=0, ignore_index=True)
    logger.info(f"Combined datasets: {len(combined_detector_df)} total rows each")

    if max_rows is not None and len(combined_detector_df) > max_rows:
        logger.info(f"Downsampling to {max_rows} rows with stratified sampling...")

        if "process" in combined_detector_df.columns:
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
            sample_indices = combined_detector_df.sample(
                n=max_rows, random_state=random_state
            ).index
            combined_detector_df = combined_detector_df.loc[sample_indices].reset_index(
                drop=True
            )
            combined_truth_df = combined_truth_df.loc[sample_indices].reset_index(
                drop=True
            )

        logger.info(f"After downsampling: {len(combined_detector_df)} rows each")

    shuffle_indices = np.arange(len(combined_detector_df))
    np.random.seed(random_state)
    np.random.shuffle(shuffle_indices)

    combined_detector_df = combined_detector_df.iloc[shuffle_indices].reset_index(
        drop=True
    )
    combined_truth_df = combined_truth_df.iloc[shuffle_indices].reset_index(drop=True)

    combined_detector_df.to_csv(detector_output_path, index=False)
    combined_truth_df.to_csv(truth_output_path, index=False)

    logger.info(f"Combined detector dataset saved: {detector_output_path}")
    logger.info(f"Combined truth dataset saved: {truth_output_path}")
    logger.info(f"Total rows: {len(combined_detector_df)}, Detector columns: {len(combined_detector_df.columns)}, Truth columns: {len(combined_truth_df.columns)}")

    if "process" in combined_detector_df.columns:
        final_process_counts = (
            combined_detector_df["process"].value_counts().sort_index()
        )
        logger.info(f"Rows per process: {dict(final_process_counts)}")

    return detector_output_path, truth_output_path


def main(config_path="diffusion_config.yaml"):
    """Main function to run the preprocessing pipeline."""

    config = load_unified_config(config_path)

    logger.info("Starting diffusion data preprocessing pipeline")
    config.print_summary()

    file_paths = config.get_file_paths()

    detector_csv_files = file_paths["detector_files"]
    truth_csv_files = file_paths["truth_files"]

    if not detector_csv_files or not truth_csv_files:
        logger.error("No input files specified in configuration for training data")
        return

    logger.info(f"Processing {len(detector_csv_files)} detector-truth training file pairs")

    detector_output_files, truth_output_files = process_csv_file_pairs(
        detector_csv_files, truth_csv_files, config
    )

    if not detector_output_files or not truth_output_files:
        logger.error("No training file pairs were successfully processed. Exiting.")
        return

    detector_output_path = file_paths["detector_dataset_path"]
    truth_output_path = file_paths["truth_dataset_path"]

    data_config = config.get_data_config()

    detector_dataset, truth_dataset = combine_and_shuffle_datasets(
        detector_input_files=detector_output_files,
        truth_input_files=truth_output_files,
        detector_output_path=detector_output_path,
        truth_output_path=truth_output_path,
        max_rows=config.data_processing.get("max_rows"),
        random_state=data_config["random_state"],
    )

    logger.info("Training preprocessing complete!")
    logger.info(f"Final training detector dataset: {detector_dataset}")
    logger.info(f"Final training truth dataset: {truth_dataset}")

    val_detector_files = file_paths.get("validation_detector_files", [])
    val_truth_files = file_paths.get("validation_truth_files", [])

    if val_detector_files and val_truth_files:
        logger.info(f"Processing {len(val_detector_files)} detector-truth validation file pairs")

        val_detector_output_files, val_truth_output_files = process_csv_file_pairs(
            val_detector_files, val_truth_files, config
        )

        if not val_detector_output_files or not val_truth_output_files:
            logger.warning("No validation file pairs were successfully processed. Skipping.")
        else:
            val_detector_output_path = file_paths.get(
                "detector_val_dataset_path", "diffusion_validation_dataset.csv"
            )
            val_truth_output_path = file_paths.get(
                "truth_val_dataset_path", "diffusion_validation_targets.csv"
            )

            val_detector_dataset, val_truth_dataset = combine_and_shuffle_datasets(
                detector_input_files=val_detector_output_files,
                truth_input_files=val_truth_output_files,
                detector_output_path=val_detector_output_path,
                truth_output_path=val_truth_output_path,
                max_rows=config.data_processing.get("max_rows"),
                random_state=data_config["random_state"],
            )

            logger.info("Validation preprocessing complete!")
            logger.info(f"Final validation detector dataset: {val_detector_dataset}")
            logger.info(f"Final validation truth dataset: {val_truth_dataset}")
    else:
        logger.info("No validation_detector_files / validation_truth_files configured; skipping separate validation dataset preprocessing.")


if __name__ == "__main__":
    main()
