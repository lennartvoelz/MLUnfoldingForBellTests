import numpy as np

import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiffusionModel(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim=256, num_layers=4, time_dim=32, dropout=0.1
    ):
        """
        Neural network for diffusion model in particle physics reconstruction.

        Parameters:
            input_dim: Dimension of input data (detector-level + conditioning)
            output_dim: Dimension of output data (truth-level four-vectors)
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            time_dim: Time embedding dimension
            dropout: Dropout rate (default: 0.1)
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_proj = nn.Linear(output_dim, hidden_dim)

        self.has_conditioning = input_dim > 0
        if self.has_conditioning:
            self.cond_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.cond_proj = None

        layers = []
        layers += [
            nn.Linear(hidden_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]

        self.main_net = nn.Sequential(*layers)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, cond, t, train=True):
        """
        Forward pass of the diffusion model.

        Parameters:
            x: Noisy data (truth-level four-vectors with noise)
            cond: Conditioning data (detector-level information)
            t: Timestep
            train: Training mode flag

        Returns:
            Predicted noise
        """
        if isinstance(t, int):
            t = torch.tensor([t] * x.shape[0], device=x.device)
        time_emb = self.time_mlp(t.float())

        x_proj = self.input_proj(x)

        if self.has_conditioning and cond.shape[-1] > 0:
            cond_proj = self.cond_proj(cond)
            h = x_proj + cond_proj
        else:
            h = x_proj

        h = torch.cat([h, time_emb], dim=-1)
        h = self.main_net(h)

        noise_pred = self.output_proj(h)
        return noise_pred


class Model(nn.Module):
    def __init__(
        self, device, beta_1, beta_T, T, input_dim, output_dim,
        hidden_dim=256, num_layers=4, time_dim=32, dropout=0.1,
        loss_function="mse", delta_0=0.1, alpha_decay=3.0
    ):
        """
        Complete diffusion model wrapper for particle physics reconstruction.

        Parameters:
            device: PyTorch device
            beta_1: Initial beta value
            beta_T: Final beta value
            T: Number of timesteps
            input_dim: Input dimension (detector-level)
            output_dim: Output dimension (truth-level)
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            time_dim: Time embedding dimension
            dropout: Dropout rate
            loss_function: Loss function type ("mse" or "pseudo_huber")
            delta_0: Initial delta parameter for Pseudo Huber Loss
            alpha_decay: Decay rate for time-dependent delta scheduling
        """
        super().__init__()

        self.device = device
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T
        self.loss_function = loss_function
        self.delta_0 = delta_0
        self.alpha_decay = alpha_decay

        self.betas = torch.linspace(beta_1, beta_T, T).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.model = DiffusionModel(
            input_dim, output_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, time_dim=time_dim, dropout=dropout
        ).to(device)

    def forward(self, x, cond, t, train=True):
        """Forward pass through the model."""
        return self.model(x, cond, t, train)

    def _pseudo_huber_loss(self, pred, target, delta):
        """
        Compute Pseudo Huber Loss: H_δ(x) = δ²(√(1 + x²/δ²) − 1)

        Parameters:
            pred: Predicted values
            target: Target values
            delta: Delta parameter controlling switching point

        Returns:
            Pseudo Huber loss
        """
        diff = pred - target
        delta_sq = delta ** 2
        loss = delta_sq * (torch.sqrt(1 + (diff ** 2) / delta_sq) - 1)
        return loss.mean()

    def loss_fn(self, x_0, cond):
        """
        Calculate diffusion loss for training.

        Supports both MSE and Pseudo Huber Loss with time-dependent scheduling.

        Parameters:
            x_0: Clean truth-level data
            cond: Conditioning data (detector-level)

        Returns:
            Loss value
        """
        batch_size = x_0.shape[0]

        t = torch.randint(0, self.T, (batch_size,), device=self.device)

        noise = torch.randn_like(x_0)
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t]).unsqueeze(-1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t]).unsqueeze(-1)

        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

        noise_pred = self.model(x_t, cond, t, train=True)

        if self.loss_function == "pseudo_huber":
            # Time-dependent delta scheduling: δ(t) = δ₀ · exp(-α · t/T)
            # Convert t to float for division
            t_normalized = t.float() / self.T
            delta_t = self.delta_0 * torch.exp(-self.alpha_decay * t_normalized)
            # Expand delta_t to match noise_pred shape for element-wise operations
            delta_t = delta_t.unsqueeze(-1)
            loss = self._pseudo_huber_loss(noise_pred, noise, delta_t)
        else:
            # Default to MSE loss
            loss = nn.MSELoss()(noise_pred, noise)

        return loss
