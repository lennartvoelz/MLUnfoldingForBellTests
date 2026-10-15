import torch


class DiffusionProcess:
    def __init__(self, beta_1, beta_T, T, diffusion_fn, device, shape):
        """Diffusion process for particle physics event reconstruction."""
        self.betas = torch.linspace(start=beta_1, end=beta_T, steps=T)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(
            1 - torch.linspace(start=beta_1, end=beta_T, steps=T), dim=0
        ).to(device=device)
        self.alpha_prev_bars = torch.cat(
            [torch.Tensor([1]).to(device=device), self.alpha_bars[:-1]]
        )
        self.shape = shape
        self.diffusion_fn = diffusion_fn
        self.device = device

    def reverse_step(self, x, y):
        """Performs the reverse diffusion process step by step."""
        for t in reversed(range(len(self.alpha_bars))):
            predict_epsilon = self.diffusion_fn(x, y, t, train=False)

            mu_theta_xt = torch.sqrt(1 / self.alphas[t]) * (
                x - self.betas[t] / torch.sqrt(1 - self.alpha_bars[t]) * predict_epsilon
            )

            noise = torch.zeros_like(x) if t == 0 else torch.randn_like(x)
            sqrt_beta = torch.sqrt(self.betas[t])
            sigma_t = sqrt_beta

            x = mu_theta_xt + sigma_t * noise
            yield x

    @torch.no_grad()
    def sampling(self, sampling_number, y):
        """Generate samples using the reverse diffusion process."""
        sample_noise = torch.randn((sampling_number, *self.shape)).to(
            device=self.device
        )

        final = None
        for t, sample in enumerate(self.reverse_step(sample_noise, y)):
            final = sample

        return final

    def forward_step(self, x_0, t):
        """Forward diffusion process - adds noise to clean data."""
        noise = torch.randn_like(x_0)
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t])
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t])

        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise
