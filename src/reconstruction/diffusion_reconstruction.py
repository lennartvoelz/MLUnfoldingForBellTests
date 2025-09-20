import os
import numpy as np
import pandas as pd

# Set environment variables to avoid CUDA issues
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from src.utils.lorentz_vector import LorentzVector
from src.reconstruction.diffusion import (
    DiffusionProcess,
    Model,
    DiffusionConfig,
    DiffusionDataPreprocessor,
)
from src.utils.mode_calculation import KinematicModeCalculator


class DiffusionReconstruction:
    """
    Diffusion model-based reconstruction for particle physics events.
    Integrates with the existing reconstruction framework.
    """

    def __init__(self, config_dict=None, model_path=None):
        """
        Initialize diffusion reconstruction.

        Parameters:
            config_dict: Configuration dictionary (optional)
            model_path: Path to trained model checkpoint
        """
        # Initialize configuration
        self.config = DiffusionConfig()
        if config_dict:
            self._update_config(config_dict)

        # Initialize data preprocessor
        self.data_preprocessor = DiffusionDataPreprocessor(self.config)

        # Initialize mode calculator
        self.mode_calculator = KinematicModeCalculator()

        # Model components
        self.model = None
        self.diffusion_process = None

        # Load model if path provided
        if model_path:
            self.load_model(model_path)

    def _update_config(self, config_dict):
        """Update configuration with provided dictionary."""
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def initialize_model(self):
        """Initialize the diffusion model and process."""
        model_config = self.config.get_model_config()

        # Create model
        self.model = Model(
            device=model_config["device"],
            beta_1=model_config["beta_1"],
            beta_T=model_config["beta_T"],
            T=model_config["T"],
            input_dim=model_config["input_dim"],
            output_dim=model_config["output_dim"],
        )

        # Create diffusion process
        self.diffusion_process = DiffusionProcess(
            beta_1=model_config["beta_1"],
            beta_T=model_config["beta_T"],
            T=model_config["T"],
            diffusion_fn=self.model,
            device=model_config["device"],
            shape=self.config.shape_out,
        )

    def load_model(self, model_path):
        """
        Load trained model from checkpoint.

        Parameters:
            model_path: Path to model checkpoint
        """
        if self.model is None:
            self.initialize_model()

        # Load state dict with cleanup for compiled models
        state_dict = torch.load(
            model_path, weights_only=True, map_location=self.config.device
        )

        # Remove unwanted prefix from compiled models
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def prepare_detector_data(self, detector_data):
        """
        Prepare detector-level data for reconstruction.

        Parameters:
            detector_data: DataFrame or array with detector information

        Returns:
            Preprocessed and normalized conditioning data
        """
        if isinstance(detector_data, np.ndarray):
            # Convert to DataFrame with expected column names
            columns = [
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
            detector_data = pd.DataFrame(detector_data, columns=columns)

        # Convert to four-vectors and calculate conditioning features
        lep1_4vec, lep2_4vec, missing_4vec = (
            self.data_preprocessor.convert_to_four_vectors(detector_data)
        )
        conditioning_features = self.data_preprocessor.calculate_conditioning_features(
            lep1_4vec, lep2_4vec, missing_4vec
        )

        # Combine detector features with conditioning
        detector_features = np.concatenate([lep1_4vec, lep2_4vec, missing_4vec], axis=1)
        X = np.concatenate([detector_features, conditioning_features], axis=1)

        # Normalize
        X_normalized, _ = self.data_preprocessor.normalize_data(
            X, np.zeros((X.shape[0], 8))
        )

        return torch.from_numpy(X_normalized).float().to(self.config.device)

    def reconstruct_events(self, detector_data, n_samples=1):
        """
        Reconstruct neutrino four-vectors from detector-level data.

        Parameters:
            detector_data: Detector-level information
            n_samples: Number of samples to generate per event

        Returns:
            Reconstructed neutrino four-vectors
        """
        if self.model is None or self.diffusion_process is None:
            raise ValueError(
                "Model not initialized. Call initialize_model() or load_model() first."
            )

        # Prepare conditioning data
        conditioning_data = self.prepare_detector_data(detector_data)

        # Generate samples
        with torch.no_grad():
            if n_samples == 1:
                # Single sample per event
                reconstructed = self.diffusion_process.sampling(
                    conditioning_data.shape[0], conditioning_data
                )
            else:
                # Multiple samples per event
                all_samples = []
                for _ in range(n_samples):
                    samples = self.diffusion_process.sampling(
                        conditioning_data.shape[0], conditioning_data
                    )
                    all_samples.append(samples)
                reconstructed = torch.stack(
                    all_samples, dim=1
                )  # Shape: (n_events, n_samples, 8)

        # Convert back to numpy and denormalize
        reconstructed_np = reconstructed.cpu().numpy()

        if n_samples == 1:
            reconstructed_denorm = self.data_preprocessor.denormalize_output(
                reconstructed_np
            )
        else:
            # Denormalize each sample
            reconstructed_denorm = np.zeros_like(reconstructed_np)
            for i in range(n_samples):
                reconstructed_denorm[:, i, :] = (
                    self.data_preprocessor.denormalize_output(reconstructed_np[:, i, :])
                )

        return reconstructed_denorm

    def calculate_reconstruction_modes(self, detector_data, n_samples=100):
        """
        Calculate modes of reconstructed distributions.

        Parameters:
            detector_data: Detector-level information
            n_samples: Number of samples for mode calculation

        Returns:
            Dictionary with mode values for different variables
        """
        # Generate multiple samples
        reconstructed = self.reconstruct_events(detector_data, n_samples)

        # Calculate modes for each particle
        neutrino1_samples = reconstructed[:, :, :4].reshape(
            -1, 4
        )  # Flatten across events and samples
        neutrino2_samples = reconstructed[:, :, 4:].reshape(-1, 4)

        modes = {
            "neutrino1": self.mode_calculator.calculate_all_modes(neutrino1_samples),
            "neutrino2": self.mode_calculator.calculate_all_modes(neutrino2_samples),
        }

        return modes

    def reconstruct_to_lorentz_vectors(self, detector_data):
        """
        Reconstruct events and return as LorentzVector objects.

        Parameters:
            detector_data: Detector-level information

        Returns:
            List of tuples (neutrino1, neutrino2) as LorentzVector objects
        """
        reconstructed = self.reconstruct_events(detector_data)

        results = []
        for i in range(reconstructed.shape[0]):
            neutrino1_4vec = reconstructed[i, :4]
            neutrino2_4vec = reconstructed[i, 4:]

            neutrino1 = LorentzVector(neutrino1_4vec, type="four-vector")
            neutrino2 = LorentzVector(neutrino2_4vec, type="four-vector")

            results.append((neutrino1, neutrino2))

        return results

    def evaluate_reconstruction_quality(self, detector_data, truth_data):
        """
        Evaluate reconstruction quality against truth data.

        Parameters:
            detector_data: Detector-level information
            truth_data: Truth-level neutrino four-vectors

        Returns:
            Dictionary with evaluation metrics
        """
        reconstructed = self.reconstruct_events(detector_data)

        # Calculate differences
        diff = reconstructed - truth_data

        # Calculate metrics
        mae = np.mean(np.abs(diff), axis=0)
        mse = np.mean(diff**2, axis=0)
        rmse = np.sqrt(mse)

        metrics = {
            "mae_per_component": mae,
            "mse_per_component": mse,
            "rmse_per_component": rmse,
            "mae_total": np.mean(mae),
            "mse_total": np.mean(mse),
            "rmse_total": np.mean(rmse),
        }

        return metrics

    def get_reconstruction_array(self, detector_data):
        """
        Get reconstruction in the format expected by the evaluation system.

        Parameters:
            detector_data: Detector-level information

        Returns:
            Array with format [lep1_4vec, lep2_4vec, neutrino1_4vec, neutrino2_4vec]
        """
        # Get detector-level leptons
        if isinstance(detector_data, pd.DataFrame):
            lep1_4vec = detector_data[
                ["p_l_1_E", "p_l_1_x", "p_l_1_y", "p_l_1_z"]
            ].values
            lep2_4vec = detector_data[
                ["p_l_2_E", "p_l_2_x", "p_l_2_y", "p_l_2_z"]
            ].values
        else:
            lep1_4vec = detector_data[:, :4]
            lep2_4vec = detector_data[:, 4:8]

        # Reconstruct neutrinos
        reconstructed = self.reconstruct_events(detector_data)
        neutrino1_4vec = reconstructed[:, :4]
        neutrino2_4vec = reconstructed[:, 4:]

        # Combine in expected format
        full_reconstruction = np.concatenate(
            [lep1_4vec, lep2_4vec, neutrino1_4vec, neutrino2_4vec], axis=1
        )

        return full_reconstruction
