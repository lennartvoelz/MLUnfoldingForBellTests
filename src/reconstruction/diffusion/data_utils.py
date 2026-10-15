import numpy as np
import pandas as pd
import os
from pathlib import Path


def calculate_moments(values, n_moments=4, eps=1e-8):
    """
    Calculate the first n moments of pT distributions.

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


def circ_moments(alpha, k_max=2):
    """Calculate circular moments."""
    a = np.asarray(alpha, dtype=np.float64)
    a = (a + np.pi) % (2 * np.pi) - np.pi
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.zeros(2 * k_max, dtype=np.float64)
    out = []
    for k in range(1, k_max + 1):
        out += [np.mean(np.cos(k * a)), np.mean(np.sin(k * a))]
    return np.array(out, dtype=np.float64)


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
    E1, pz1 = lep1_4vec[:, 0], lep1_4vec[:, 3]
    E2, pz2 = lep2_4vec[:, 0], lep2_4vec[:, 3]
    E_miss, pz_miss = missing_4vec[:, 0], missing_4vec[:, 3]

    mt_lep1_squared = E1**2 - pz1**2
    mt_lep2_squared = E2**2 - pz2**2
    mt_miss_squared = E_miss**2 - pz_miss**2

    mt_lep1 = np.sqrt(np.maximum(mt_lep1_squared, 0.0))
    mt_lep2 = np.sqrt(np.maximum(mt_lep2_squared, 0.0))
    mt_miss = np.sqrt(np.maximum(mt_miss_squared, 0.0))

    mt_total = mt_lep1 + mt_lep2 + mt_miss
    return mt_total


def calculate_mode(data, bins=50):
    """
    Calculate the mode of a distribution using histogram.

    Parameters:
        data: Input data array
        bins: Number of bins for histogram

    Returns:
        Mode value
    """
    hist, bin_edges = np.histogram(data, bins=bins)
    max_bin_idx = np.argmax(hist)
    mode = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2
    return mode


def calculate_kinematic_modes(four_vectors, epsilon=1e-8, bins=50):
    """
    Calculate modes for kinematic variables from four-vectors.

    Parameters:
        four_vectors: Array of four-vectors [E, px, py, pz]
        epsilon: Small value to prevent division by zero
        bins: Number of bins for histogram

    Returns:
        Dictionary of mode values for different variables
    """
    E, px, py, pz = (
        four_vectors[:, 0],
        four_vectors[:, 1],
        four_vectors[:, 2],
        four_vectors[:, 3],
    )
    pt = np.sqrt(px**2 + py**2)

    modes = {
        "E": calculate_mode(E, bins=bins),
        "px": calculate_mode(px, bins=bins),
        "py": calculate_mode(py, bins=bins),
        "pz": calculate_mode(pz, bins=bins),
        "pt": calculate_mode(pt, bins=bins),
        "eta": calculate_mode(
            0.5
            * np.log(
                (np.sqrt(px**2 + py**2 + pz**2) + pz)
                / (np.sqrt(px**2 + py**2 + pz**2) - pz + epsilon)
            ),
            bins=bins,
        ),
        "phi": calculate_mode(np.arctan2(py, px), bins=bins),
    }

    return modes


class DiffusionDataPreprocessor:
    """Data preprocessor specifically for diffusion model training."""

    def __init__(self, config):
        """
        Initialize with diffusion configuration.

        Parameters:
            config: DiffusionConfig object
        """
        self.config = config

    def convert_to_four_vectors(self, data):
        """Convert detector-level data to four-vector format."""
        lep1_4vec = data[["p_l_1_E", "p_l_1_x", "p_l_1_y", "p_l_1_z"]].values
        lep2_4vec = data[["p_l_2_E", "p_l_2_x", "p_l_2_y", "p_l_2_z"]].values

        mpx = data["mpx"].values
        mpy = data["mpy"].values
        mpt = np.sqrt(mpx**2 + mpy**2)

        missing_4vec = np.column_stack([mpt, mpx, mpy, np.zeros_like(mpx)])

        return lep1_4vec, lep2_4vec, missing_4vec

    def calculate_conditioning_features(self, lep1_4vec, lep2_4vec, missing_4vec):
        """
        Calculate conditioning features including pT moments and angle information.

        Parameters:
            lep1_4vec, lep2_4vec, missing_4vec: Four-vector arrays

        Returns:
            Conditioning feature array (empty if moment conditioning disabled)
        """
        if not getattr(self.config, "moment_conditioning_enabled", True):
            return np.zeros((lep1_4vec.shape[0], 0))

        px1, py1, pz1 = lep1_4vec[:, 1], lep1_4vec[:, 2], lep1_4vec[:, 3]
        px2, py2, pz2 = lep2_4vec[:, 1], lep2_4vec[:, 2], lep2_4vec[:, 3]
        lep1_pt = np.hypot(px1, py1)
        lep2_pt = np.hypot(px2, py2)

        pt_1_moments = calculate_moments(lep1_pt, self.config.pt_conditioning_moments)
        pt_2_moments = calculate_moments(lep2_pt, self.config.pt_conditioning_moments)

        phi1 = np.arctan2(py1, px1)
        phi2 = np.arctan2(py2, px2)
        dphi = np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))

        epsilon = getattr(self.config, "epsilon", 1e-8)

        eta1 = 0.5 * np.log(
            (np.sqrt(px1**2 + py1**2 + pz1**2) + pz1)
            / (np.sqrt(px1**2 + py1**2 + pz1**2) - pz1 + epsilon)
        )
        eta2 = 0.5 * np.log(
            (np.sqrt(px2**2 + py2**2 + pz2**2) + pz2)
            / (np.sqrt(px2**2 + py2**2 + pz2**2) - pz2 + epsilon)
        )
        deta = eta1 - eta2
        deta = (deta + np.pi) % (2 * np.pi) - np.pi

        eta_features = circ_moments(deta, self.config.eta_conditioning_moments)
        phi_features = circ_moments(dphi, self.config.phi_conditioning_moments)

        feature_list = [pt_1_moments, pt_2_moments, eta_features, phi_features]

        mt_moments_count = getattr(self.config, "mt_conditioning_moments", 0)
        if mt_moments_count > 0:
            mt_total = calculate_total_transverse_mass(
                lep1_4vec, lep2_4vec, missing_4vec
            )
            mt_moments = calculate_moments(mt_total, mt_moments_count)
            feature_list.append(mt_moments)

        px_moments_count = getattr(self.config, "px_conditioning_moments", 0)
        if px_moments_count > 0:
            px_1_moments = calculate_moments(px1, px_moments_count)
            px_2_moments = calculate_moments(px2, px_moments_count)
            feature_list.extend([px_1_moments, px_2_moments])

        py_moments_count = getattr(self.config, "py_conditioning_moments", 0)
        if py_moments_count > 0:
            py_1_moments = calculate_moments(py1, py_moments_count)
            py_2_moments = calculate_moments(py2, py_moments_count)
            feature_list.extend([py_1_moments, py_2_moments])

        conditioning_features = np.concatenate(feature_list, axis=0)

        conditioning_features = np.repeat(
            conditioning_features[np.newaxis, :], lep1_4vec.shape[0], axis=0
        )

        return np.array(conditioning_features)

    def prepare_training_data(self, detector_data, truth_data):
        """
        Prepare training data for diffusion model.

        Parameters:
            detector_data: DataFrame with detector-level information
            truth_data: DataFrame with truth-level information

        Returns:
            X (conditioning data), y (target truth-level four-vectors)
        """
        lep1_4vec, lep2_4vec, missing_4vec = self.convert_to_four_vectors(detector_data)

        conditioning_features = self.calculate_conditioning_features(
            lep1_4vec, lep2_4vec, missing_4vec
        )

        detector_features = np.concatenate(
            [lep1_4vec, lep2_4vec, missing_4vec],
            axis=1,
        )

        X = np.concatenate([detector_features, conditioning_features], axis=1)

        neutrino1_truth = truth_data[
            ["p_v_1_E_truth", "p_v_1_x_truth", "p_v_1_y_truth", "p_v_1_z_truth"]
        ].values
        neutrino2_truth = truth_data[
            ["p_v_2_E_truth", "p_v_2_x_truth", "p_v_2_y_truth", "p_v_2_z_truth"]
        ].values

        y = np.concatenate([neutrino1_truth, neutrino2_truth], axis=1)

        if getattr(self.config, "predict_conditioning_features", False):
            y = np.concatenate([y, conditioning_features], axis=1)

        return X, y

    def normalize_data(self, X, y):
        """
        Normalize data using configuration ranges.

        Parameters:
            X: Input features
            y: Target values

        Returns:
            Normalized X and y
        """
        X_normalized = self.normalize_inputs(X)

        target_norm_vec = self._build_target_norm_vec(y.shape[1])
        y_normalized = y / target_norm_vec

        return X_normalized, y_normalized

    def denormalize_output(self, y_normalized):
        """
        Denormalize model output back to physical units.

        Parameters:
            y_normalized: Normalized model output

        Returns:
            Denormalized output in physical units
        """
        target_norm_vec = self._build_target_norm_vec(y_normalized.shape[1])
        return y_normalized * target_norm_vec

    # ------------------------------------------------------------------
    # Inference-time helpers
    # ------------------------------------------------------------------

    def load_normalization_params(self):
        """Compatibility stub for loading normalization parameters."""
        self.detector_mean = None
        self.detector_std = None
        self.target_mean = None
        self.target_std = None

    def normalize_inputs(self, X: np.ndarray) -> np.ndarray:
        """Normalize input features using the same scheme as training."""
        X = np.asarray(X, dtype=np.float64)
        X_normalized = X.copy()

        detector_norm = np.tile(
            [
                self.config.E_range,
                self.config.pT_range,
                self.config.pT_range,
                self.config.pT_range,
            ],
            3,
        )
        X_normalized[:, :12] = X[:, :12] / detector_norm

        if X.shape[1] > 12:
            for i in range(self.config.pt_conditioning_moments):
                idx = 12 + i
                if idx < X.shape[1]:
                    X_normalized[:, idx] = X[:, idx] / (
                        self.config.pT_range ** (i + 1)
                    )

            offset = 12 + self.config.pt_conditioning_moments
            for i in range(self.config.pt_conditioning_moments):
                idx = offset + i
                if idx < X.shape[1]:
                    X_normalized[:, idx] = X[:, idx] / (
                        self.config.pT_range ** (i + 1)
                    )

            angle_start = offset + self.config.pt_conditioning_moments
            if angle_start < X.shape[1]:
                X_normalized[:, angle_start:] = X[:, angle_start:]

        return X_normalized

    def _build_target_norm_vec(self, y_dim: int) -> np.ndarray:
        """Construct a normalisation vector for targets of dimension y_dim."""
        base_vec = np.asarray(self.config.norm_vec, dtype=np.float64)
        if y_dim <= base_vec.shape[0]:
            return base_vec[:y_dim]

        cond_dim = y_dim - base_vec.shape[0]

        extra_scales = []
        pt_moments = getattr(self.config, "pt_conditioning_moments", 0)

        for i in range(pt_moments):
            extra_scales.append(self.config.pT_range ** (i + 1))

        for i in range(pt_moments):
            extra_scales.append(self.config.pT_range ** (i + 1))

        if len(extra_scales) < cond_dim:
            extra_scales.extend([1.0] * (cond_dim - len(extra_scales)))
        elif len(extra_scales) > cond_dim:
            extra_scales = extra_scales[:cond_dim]

        cond_vec = np.asarray(extra_scales, dtype=np.float64)
        return np.concatenate([base_vec, cond_vec], axis=0)


def compute_conditioning_features_for_file(df, config):
    """Compute conditioning features for an entire file (process)."""
    if not getattr(config, "moment_conditioning_enabled", True):
        return {}

    preprocessor = DiffusionDataPreprocessor(config)
    lep1_4vec, lep2_4vec, missing_4vec = preprocessor.convert_to_four_vectors(df)

    px1, py1 = lep1_4vec[:, 1], lep1_4vec[:, 2]
    px2, py2 = lep2_4vec[:, 1], lep2_4vec[:, 2]
    lep1_pt = np.hypot(px1, py1)
    lep2_pt = np.hypot(px2, py2)

    lep1_pt_moments = calculate_moments(lep1_pt, config.pt_conditioning_moments)
    lep2_pt_moments = calculate_moments(lep2_pt, config.pt_conditioning_moments)

    pz1, pz2 = lep1_4vec[:, 3], lep2_4vec[:, 3]
    phi1 = np.arctan2(py1, px1)
    phi2 = np.arctan2(py2, px2)
    dphi = np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))

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
    deta = (deta + np.pi) % (2 * np.pi) - np.pi

    eta_features = circ_moments(deta, config.eta_conditioning_moments)
    phi_features = circ_moments(dphi, config.phi_conditioning_moments)

    conditioning_dict = {}

    for i, value in enumerate(lep1_pt_moments):
        conditioning_dict[f"mom_pT_lep1_{i}"] = value

    for i, value in enumerate(lep2_pt_moments):
        conditioning_dict[f"mom_pT_lep2_{i}"] = value

    for i, val in enumerate(eta_features):
        conditioning_dict[f"eta_moment_{i}"] = val

    for i, val in enumerate(phi_features):
        conditioning_dict[f"phi_moment_{i}"] = val

    mt_moments_count = getattr(config, "mt_conditioning_moments", 0)
    if mt_moments_count > 0:
        mt_total = calculate_total_transverse_mass(lep1_4vec, lep2_4vec, missing_4vec)
        mt_moments = calculate_moments(mt_total, mt_moments_count)
        for i, value in enumerate(mt_moments):
            conditioning_dict[f"mom_mt_{i}"] = value

    return conditioning_dict


def preprocess_csv_file(csv_path, config, moment_feature="pT", extra_conditioners=None):
    """Process a single CSV file to add precomputed moments and conditioning features."""
    import logging
    logger = logging.getLogger(__name__)

    df = pd.read_csv(csv_path)

    process_name = Path(csv_path).stem
    if "_final_truth" in process_name:
        process_name = process_name.replace("_final_truth", "")
    if "_1M_MG5" in process_name:
        process_name = process_name.replace("_1M_MG5", "")
    if "_1M_MG" in process_name:
        process_name = process_name.replace("_1M_MG", "")

    df["process"] = process_name

    conditioning_dict = compute_conditioning_features_for_file(df, config)

    for feature_name, feature_value in conditioning_dict.items():
        df[feature_name] = feature_value

    if extra_conditioners is not None:
        extra_features = extra_conditioners(df)
        for feature_name, feature_series in extra_features.items():
            df[feature_name] = feature_series

    output_path = csv_path.replace(".csv", "_diffusion_input.csv")

    df.to_csv(output_path, index=False)

    logger.info(f"Processed {process_name}: {len(df)} rows -> {output_path}")

    return output_path


class PrecomputedDiffusionDataLoader:
    """
    Data loader for precomputed diffusion training dataset.
    Replaces on-the-fly moment computation with precomputed values.
    """

    def __init__(self, config, dataset_path="diffusion_training_dataset.csv"):
        self.config = config
        self.dataset_path = dataset_path
        self.data = None
        self.conditioning_columns = None

    def load_dataset(self):
        """Load the precomputed dataset."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"Precomputed dataset not found: {self.dataset_path}"
            )

        self.data = pd.read_csv(self.dataset_path)
        self._identify_conditioning_columns()

        return self.data

    def _identify_conditioning_columns(self):
        """Identify which columns contain precomputed conditioning features."""
        conditioning_cols = []

        moment_cols = [col for col in self.data.columns if col.startswith("mom_pT_")]
        conditioning_cols.extend(sorted(moment_cols))

        angle_cols = [
            col
            for col in self.data.columns
            if col.startswith(("eta_moment_", "phi_moment_"))
        ]
        conditioning_cols.extend(angle_cols)

        mt_cols = [col for col in self.data.columns if col.startswith("mom_mt_")]
        conditioning_cols.extend(sorted(mt_cols))

        self.conditioning_columns = conditioning_cols

    def prepare_training_data(self):
        """
        Prepare training data using precomputed features.

        Returns:
            X (detector features + precomputed conditioning), y (target neutrinos)
        """
        if self.data is None:
            self.load_dataset()

        detector_cols = [
            "p_l_1_E",
            "p_l_1_x",
            "p_l_1_y",
            "p_l_1_z",
            "p_l_2_E",
            "p_l_2_x",
            "p_l_2_y",
            "p_l_2_z",
            "mpx",
            "mpy",
        ]

        missing_cols = [col for col in detector_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing detector columns: {missing_cols}")

        detector_data = self.data[detector_cols]
        lep1_4vec, lep2_4vec, missing_4vec = self._convert_to_four_vectors(
            detector_data
        )

        detector_features = np.concatenate([lep1_4vec, lep2_4vec, missing_4vec], axis=1)

        conditioning_features = self.data[self.conditioning_columns].values

        X = np.concatenate([detector_features, conditioning_features], axis=1)

        truth_cols = [
            "p_v_1_E_truth",
            "p_v_1_x_truth",
            "p_v_1_y_truth",
            "p_v_1_z_truth",
            "p_v_2_E_truth",
            "p_v_2_x_truth",
            "p_v_2_y_truth",
            "p_v_2_z_truth",
        ]

        missing_truth_cols = [col for col in truth_cols if col not in self.data.columns]
        if missing_truth_cols:
            raise ValueError(f"Missing truth columns: {missing_truth_cols}")

        y = self.data[truth_cols].values

        if getattr(self.config, "predict_conditioning_features", False):
            y = np.concatenate([y, conditioning_features], axis=1)

        return X, y

    def _convert_to_four_vectors(self, detector_data):
        """Convert detector data to four-vectors."""
        lep1_4vec = detector_data[["p_l_1_E", "p_l_1_x", "p_l_1_y", "p_l_1_z"]].values
        lep2_4vec = detector_data[["p_l_2_E", "p_l_2_x", "p_l_2_y", "p_l_2_z"]].values

        mpx = detector_data["mpx"].values
        mpy = detector_data["mpy"].values
        mpt = np.sqrt(mpx**2 + mpy**2)
        missing_4vec = np.column_stack([mpt, mpx, mpy, np.zeros_like(mpx)])

        return lep1_4vec, lep2_4vec, missing_4vec
