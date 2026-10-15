#!/usr/bin/env python3
"""
Unified Configuration System for Diffusion Model Pipeline

This module provides a centralized configuration system that replaces
the scattered configuration logic across preprocess_diffusion_data.py,
train_diffusion.py, and unfold_diffusion.py.
"""

import os
import logging
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Optional torch import for preprocessing-only usage
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class UnifiedDiffusionConfig:
    """
    Unified configuration class for the entire diffusion model pipeline.

    This class loads configuration from a YAML file and provides a consistent
    interface for all pipeline components.
    """

    def __init__(self, config_path: str = "diffusion_config.yaml"):
        """
        Initialize configuration from YAML file.

        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = config_path
        self._load_config()
        self._validate_config()
        self._compute_derived_values()

    def _load_config(self):
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Extract main sections for easier access
        self.data = self.config.get("data", {})
        self.data_processing = self.config.get("data_processing", {})
        self.model = self.config.get("model", {})
        self.training = self.config.get("training", {})
        self.unfolding = self.config.get("unfolding", {})
        self.system = self.config.get("system", {})
        self.evaluation = self.config.get("evaluation", {})
        self.compatibility = self.config.get("compatibility", {})

    def _validate_config(self):
        """Validate configuration parameters."""
        required_sections = ["data", "data_processing", "model", "training", "system"]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(
                    f"Required configuration section '{section}' not found"
                )

        # Validate critical parameters
        if self.training.get("batch_size", 0) <= 0:
            raise ValueError("batch_size must be positive")

        if self.training.get("epochs", 0) <= 0:
            raise ValueError("epochs must be positive")

        if self.data_processing.get("pt_conditioning_moments", 0) < 0:
            raise ValueError("pt_conditioning_moments must be positive")

    def _compute_derived_values(self):
        """Compute derived configuration values."""
        # Check if moment conditioning is enabled
        self._moment_conditioning_enabled = self._is_moment_conditioning_enabled()

        # Work out conditioning dimensionality from moment configuration
        detector_4vec_dim = 12  # 3 particles × 4 components each
        if self._moment_conditioning_enabled:
            pt_moments = int(self.data_processing.get("pt_conditioning_moments", 0))
            eta_moments = int(
                self.data_processing.get("eta_conditioning_moments", 0)
            )
            phi_moments = int(
                self.data_processing.get("phi_conditioning_moments", 0)
            )
            # Allow both "mt_conditioning_moments" and legacy
            # "mass_conditioning_moments" keys.
            mt_moments = int(
                self.data_processing.get(
                    "mt_conditioning_moments",
                    self.data_processing.get("mass_conditioning_moments", 0),
                )
            )
            px_moments = int(self.data_processing.get("px_conditioning_moments", 0))
            py_moments = int(self.data_processing.get("py_conditioning_moments", 0))

            conditioning_dim = (
                2 * pt_moments
                + 2 * eta_moments
                + 2 * phi_moments
                + mt_moments
                + 2 * px_moments
                + 2 * py_moments
            )
        else:
            conditioning_dim = 0  # No conditioning features when all moments are zero

        # Calculate input dimension if not specified
        if self.model.get("input_dim") is None:
            self.model["input_dim"] = detector_4vec_dim + conditioning_dim

        # Set up device
        self.device = self._get_device()

        # Create state name for checkpoints
        self.state_name = f"{self.training['train_type']}_b{self.training['batch_size']}_it{self.training['epochs']}"

        # Set up normalization vector for target data (neutrino four-vectors)
        self.norm_vec = np.array(
            [
                self.data_processing["E_range"],
                self.data_processing["pT_range"],
                self.data_processing["pT_range"],
                self.data_processing["pT_range"],  # neutrino 1
                self.data_processing["E_range"],
                self.data_processing["pT_range"],
                self.data_processing["pT_range"],
                self.data_processing["pT_range"],  # neutrino 2
            ]
        )

        # Optionally expand output dimension when predicting conditioning features
        base_output_dim = int(self.data_processing.get("n_dims", 8))
        if self.data_processing.get("predict_conditioning_features", False):
            self.model["output_dim"] = base_output_dim + conditioning_dim
        else:
            # Preserve existing setting, or fall back to base_output_dim
            if self.model.get("output_dim") is None:
                self.model["output_dim"] = base_output_dim

        # Set up paths
        self._setup_paths()

    def _is_moment_conditioning_enabled(self):
        """
        Check if moment conditioning is enabled.

        Returns True if any moment conditioning parameter is greater than 0,
        False if all are 0 (indicating no moment conditioning should be used).
        """
        pt_moments = self.data_processing.get("pt_conditioning_moments", 0)
        eta_moments = self.data_processing.get("eta_conditioning_moments", 0)
        phi_moments = self.data_processing.get("phi_conditioning_moments", 0)
        mt_moments = self.data_processing.get("mt_conditioning_moments", 0)
        px_moments = self.data_processing.get("px_conditioning_moments", 0)
        py_moments = self.data_processing.get("py_conditioning_moments", 0)

        return (
            pt_moments > 0
            or eta_moments > 0
            or phi_moments > 0
            or mt_moments > 0
            or px_moments > 0
            or py_moments > 0
        )

    def _get_device(self):
        """Determine the appropriate device (CUDA/CPU)."""
        requested_device = self.system.get("device", "cuda")
        if not TORCH_AVAILABLE:
            return requested_device  # Return string when torch not available
        if "cuda" in requested_device.lower() and torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    def _setup_paths(self):
        """Set up output paths."""
        # Ensure paths end with /
        for path_key in ["output_path", "plots_path", "ckpt_path"]:
            path = self.system.get(path_key, "")
            if path and not path.endswith("/"):
                self.system[path_key] = path + "/"

    # =============================================================================
    # PROPERTY ACCESSORS FOR BACKWARD COMPATIBILITY
    # =============================================================================

    @property
    def seed(self):
        return self.system.get("seed", 42)

    @property
    def random_seed(self):
        return self.system.get("random_seed", False)

    @property
    def batch_size(self):
        return int(self.training["batch_size"])

    @property
    def epochs(self):
        return int(self.training["epochs"])

    @property
    def lr(self):
        return float(self.training["learning_rate"])

    @property
    def weight_decay(self):
        """L2 regularization strength for optimizer (Adam weight_decay).

        Defaults to 0.0 when not specified in the training section.
        """

        return float(self.training.get("weight_decay", 0.0))

    @property
    def beta_1(self):
        return float(self.model["beta_1"])

    @property
    def beta_T(self):
        return float(self.model["beta_T"])

    @property
    def T(self):
        return int(self.model["T"])

    @property
    def input_dim(self):
        return int(self.model["input_dim"])

    @property
    def output_dim(self):
        return int(self.model["output_dim"])

    @property
    def hidden_dim(self):
        return int(self.model["hidden_dim"])

    @property
    def num_layers(self):
        return int(self.model["num_layers"])

    @property
    def time_dim(self):
        return int(self.model["time_dim"])

    @property
    def pt_conditioning_moments(self):
        return int(self.data_processing["pt_conditioning_moments"])

    @property
    def eta_conditioning_moments(self):
        return int(self.data_processing["eta_conditioning_moments"])

    @property
    def phi_conditioning_moments(self):
        return int(self.data_processing["phi_conditioning_moments"])

    @property
    def mt_conditioning_moments(self):
        # Support both new and legacy config keys
        return int(
            self.data_processing.get(
                "mt_conditioning_moments",
                self.data_processing.get("mass_conditioning_moments", 0),
            )
        )

    @property
    def px_conditioning_moments(self):
        return int(self.data_processing.get("px_conditioning_moments", 0))

    @property
    def py_conditioning_moments(self):
        return int(self.data_processing.get("py_conditioning_moments", 0))

    @property
    def pT_range(self):
        return float(self.data_processing["pT_range"])

    @property
    def E_range(self):
        return float(self.data_processing["E_range"])

    @property
    def eta_range(self):
        return float(self.data_processing["eta_range"])

    @property
    def phi_range(self):
        return float(self.data_processing["phi_range"])

    @property
    def output_path(self):
        return self.system["output_path"]

    @property
    def plots_path(self):
        return self.system["plots_path"]

    @property
    def ckpt_path(self):
        return self.system["ckpt_path"]

    @property
    def save_int(self):
        return int(self.training.get("save_interval", 100))

    @property
    def early_stopping_patience(self):
        """Patience (in epochs) for validation-based early stopping.

        A value <= 0 disables early stopping.
        """

        value = self.training.get("early_stopping_patience", 0)
        if value is None:
            return 0
        return int(value)

    @property
    def early_stopping_restore_best(self):
        """Whether to restore the best model weights when stopping early."""

        return bool(self.training.get("early_stopping_restore_best", True))

    @property
    def train_type(self):
        return str(self.training["train_type"])

    @property
    def unf_type(self):
        return str(self.unfolding.get("unf_type", "lepqua_CT14lo"))

    @property
    def unfold_size(self):
        return int(self.unfolding.get("unfold_size", 200000))

    @property
    def sample_size(self):
        return int(self.unfolding.get("sample_size", 10000))

    @property
    def moment_conditioning_enabled(self):
        """Return whether moment conditioning is enabled."""
        return self._moment_conditioning_enabled

    @property
    def n_dims(self):
        """Number of output dimensions (neutrino four-vectors)."""
        return int(self.data_processing.get("n_dims", 8))

    @property
    def predict_conditioning_features(self):
        """Whether the model should also predict conditioning features."""
        return bool(self.data_processing.get("predict_conditioning_features", False))

    @property
    def shape_in(self):
        """Input shape tuple for backward compatibility."""
        return (self.input_dim,)

    @property
    def shape_out(self):
        """Output shape tuple for backward compatibility."""
        return (self.output_dim,)

    @property
    def save_ckpts(self):
        """Whether to save checkpoints during training."""
        return bool(self.training.get("save_checkpoints", True))

    @property
    def load_preprocessed_training_data(self):
        """Whether to load preprocessed training data."""
        return bool(self.compatibility.get("load_preprocessed_training_data", True))

    # =============================================================================
    # UTILITY METHODS
    # =============================================================================

    def set_seed(self, seed: Optional[int] = None):
        """Set random seed for reproducibility."""
        if seed is None:
            seed = self.seed
        if self.random_seed:
            seed = np.random.randint(1000)

        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
        np.random.seed(seed)

    def create_directories(self):
        """Create necessary output directories."""
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.plots_path, exist_ok=True)
        os.makedirs(self.ckpt_path, exist_ok=True)

    def get_model_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for model initialization."""
        return {
            "device": self.device,
            "beta_1": self.beta_1,
            "beta_T": self.beta_T,
            "T": self.T,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "time_dim": self.time_dim,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
        }

    def get_data_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for data preprocessing."""
        return {
            "pt_conditioning_moments": self.pt_conditioning_moments,
            "eta_conditioning_moments": self.eta_conditioning_moments,
            "phi_conditioning_moments": self.phi_conditioning_moments,
            "mt_conditioning_moments": self.mt_conditioning_moments,
            "moment_conditioning_enabled": self.moment_conditioning_enabled,
            "pT_range": self.pT_range,
            "E_range": self.E_range,
            "eta_range": self.eta_range,
            "phi_range": self.phi_range,
            "max_zero_columns": self.data_processing.get("max_zero_columns", 2),
            "random_state": self.data_processing.get("random_state", 42),
            "norm_vec": self.norm_vec,
        }

    def get_file_paths(self) -> Dict[str, Any]:
        """Get all file paths from configuration."""
        return {
            "detector_files": self.data.get("preprocessing", {}).get(
                "detector_files", []
            ),
            "truth_files": self.data.get("preprocessing", {}).get("truth_files", []),
            "validation_detector_files": self.data.get("preprocessing", {}).get(
                "validation_detector_files", []
            ),
            "validation_truth_files": self.data.get("preprocessing", {}).get(
                "validation_truth_files", []
            ),
            "detector_dataset_path": self.data.get("detector_dataset_path"),
            "truth_dataset_path": self.data.get("truth_dataset_path"),
            "detector_val_dataset_path": self.data.get("detector_val_dataset_path"),
            "truth_val_dataset_path": self.data.get("truth_val_dataset_path"),
            "detector_sim_path": self.data.get("detector_sim_path"),
            "truth_path": self.data.get("truth_path"),
            "data_path": self.data.get("data_path"),
            "raw_data_path": self.data.get("raw_data_path"),
        }

    def print_summary(self):
        """Print a summary of the configuration."""
        logger.info("=" * 60)
        logger.info("UNIFIED DIFFUSION CONFIGURATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Configuration file: {self.config_path}")
        logger.info(f"Device: {self.device}")
        logger.info(f"State name: {self.state_name}")
        logger.info("Model Architecture:")
        logger.info(f"  Input dim: {self.input_dim}")
        logger.info(f"  Output dim: {self.output_dim}")
        logger.info(f"  Hidden dim: {self.hidden_dim}")
        logger.info(f"  Layers: {self.num_layers}")
        logger.info("Training Parameters:")
        logger.info(f"  Epochs: {self.epochs}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Learning rate: {self.lr}")
        logger.info("Data Processing:")
        logger.info(f"  Moment conditioning enabled: {self.moment_conditioning_enabled}")
        logger.info(f"  pT moments: {self.pt_conditioning_moments}")
        logger.info(f"  Eta moments: {self.eta_conditioning_moments}")
        logger.info(f"  Phi moments: {self.phi_conditioning_moments}")
        logger.info(f"  mT moments: {self.mt_conditioning_moments}")
        logger.info("=" * 60)


def load_unified_config(
    config_path: str = "diffusion_config.yaml",
) -> UnifiedDiffusionConfig:
    """
    Convenience function to load unified configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        UnifiedDiffusionConfig instance
    """
    return UnifiedDiffusionConfig(config_path)
