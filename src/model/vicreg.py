import torch
import torch.nn as nn


class VICReg(nn.Module):
    """
    Variance-Invariance-Covariance Regularization (variance + covariance only).

    The invariance term (pushing augmented views together) is omitted — GoL
    training has no augmented pairs. Only variance and covariance are needed.

    Variance term: each of the latent_dim dimensions should have std >= 1
    across the batch. Penalises dimensional collapse where one or more
    dimensions output near-constant values regardless of input.

    Covariance term: off-diagonal elements of the (latent_dim x latent_dim)
    covariance matrix should be near 0. Penalises feature redundancy where
    two dimensions encode the same information.

    Applied to z_0 vectors only (initial grid encodings). The training loop
    accumulates z_0 across gradient_accumulation_steps batches and calls
    this loss once per accumulation cycle on the full buffer (256 vectors).
    This directly regularises the distribution that z_cloud represents.
    Mid-trajectory z_t vectors are shaped by the contrastive loss instead.
    """

    def __init__(self, latent_dim: int = 128, lambda_var: float = 25.0, lambda_cov: float = 25.0):
        super().__init__()
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        # Pre-compute off-diagonal mask; register as buffer so it moves with .to(device)
        mask = ~torch.eye(latent_dim, dtype=torch.bool)
        self.register_buffer('off_diag', mask)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (N, latent_dim) float32 — batch of z_0 vectors
               N should be >= latent_dim for a reliable covariance estimate
               (training loop targets N=256 via gradient accumulation)
        Returns:
            loss: scalar
        """
        N, D = z.shape

        # --- Variance term ---
        # Each dimension should have std near 1 across the batch
        std = torch.sqrt(z.var(dim=0) + 1e-4)                  # (D,)
        loss_var = torch.mean(torch.relu(1.0 - std))

        # --- Covariance term ---
        # Off-diagonal elements of the normalised covariance matrix should be 0
        z_c = z - z.mean(dim=0)                                 # (N, D) centred
        cov = (z_c.T @ z_c) / (N - 1)                          # (D, D)
        loss_cov = (cov[self.off_diag] ** 2).mean()

        return self.lambda_var * loss_var + self.lambda_cov * loss_cov
