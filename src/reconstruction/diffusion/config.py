import os
import numpy as np

import torch


class DiffusionConfig:
    """Configuration class for diffusion model parameters."""

    def __init__(self):
        # Random seed
        self.seed = 42
        self.random_seed = False

        # Data paths
        self.load_preprocessed_training_data = True
        self.input_path = "../ml-unfold-datasets/"

        # Training datasets
        self.train_type = "combined18"
        self.unf_type = "lepqua_CT14lo"

        # Training parameters
        self.epochs = 5000
        self.batch_size = 2048
        self.unfold_size = 1000000
        self.sample_size = self.unfold_size // 100

        # Model checkpointing
        self.save_int = 100
        self.save_ckpts = True
        self.state_name = f"{self.train_type}_b{self.batch_size}_it{self.epochs}"

        # Output paths
        self.output_path = f"./outputs/{self.train_type}/"
        self.plots_path = f"./plots/{self.train_type}/"
        self.ckpt_path = f"model-state/{self.train_type}_b{self.batch_size}_it5000/"

        # Diffusion hyperparameters
        self.lr = 3e-4
        self.beta_1 = 1e-4
        self.beta_T = 0.02
        self.T = 500

        # Data dimensions
        self.n_dims = 8  # 4 components for each of 2 neutrinos (E, px, py, pz)
        self.pt_conditioning_moments = 3
        self.eta_conditioning_moments = 0
        self.phi_conditioning_moments = 2

        # Input: detector-level (leptons + missing momentum) + conditioning
        self.shape_in = (
            self.n_dims
            + self.pt_conditioning_moments
            + self.eta_conditioning_moments
            + self.phi_conditioning_moments,
        )
        # Output: truth-level neutrino four-vectors
        self.shape_out = (self.n_dims,)

        # Device - handle CUDA compatibility issues
        try:
            if torch.cuda.is_available():
                # Test if CUDA actually works
                test_tensor = torch.tensor([1.0]).cuda()
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device("cpu")
        except Exception as e:
            print(f"CUDA not working properly ({e}), falling back to CPU")
            self.device = torch.device("cpu")

        # Normalization ranges
        self.eta_range = 2.5
        self.phi_range = 3.14
        self.pT_range = 200
        self.E_range = 300
        self.norm_vec = np.array(
            [
                self.E_range,
                self.pT_range,
                self.pT_range,
                self.pT_range,  # neutrino 1
                self.E_range,
                self.pT_range,
                self.pT_range,
                self.pT_range,  # neutrino 2
            ]
        )

    def set_seed(self, seed=None):
        """Set random seed for reproducibility."""
        if seed is None:
            seed = self.seed
        if self.random_seed:
            seed = np.random.randint(1000)

        torch.manual_seed(seed)
        np.random.seed(seed)

    def create_directories(self):
        """Create necessary output directories."""
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.plots_path, exist_ok=True)
        os.makedirs(self.ckpt_path, exist_ok=True)

    def get_model_config(self):
        """Get configuration dictionary for model initialization."""
        return {
            "device": self.device,
            "beta_1": self.beta_1,
            "beta_T": self.beta_T,
            "T": self.T,
            "input_dim": self.shape_in[0],
            "output_dim": self.shape_out[0],
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
        }

    def get_data_config(self):
        """Get configuration dictionary for data preprocessing."""
        return {
            "n_conditioning_moments": self.n_conditioning_moments,
            "n_angle_features": self.n_angle_features,
            "norm_vec": self.norm_vec,
            "eta_range": self.eta_range,
            "phi_range": self.phi_range,
            "pT_range": self.pT_range,
            "E_range": self.E_range,
        }
