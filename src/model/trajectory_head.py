import torch
import torch.nn as nn


class TrajectoryHead(nn.Module):
    """
    Per-timestep behavioral signal predictor: z_t -> R^10.

    Predicts the normalized 10-signal value [P, dcx, dcy, V, E, N_cc,
    S_lag_2, S_lag_4, S_lag_8, S_lag_16] at whatever timestep t produced z_t.
    Applied independently at each rollout step — no recurrence, no sequence
    state. Each z_t is treated as a standalone input.

    Hidden dim 128 (matching latent dim) is deliberately modest. A larger head
    could fit the 10 signals through complex nonlinear mappings even when z_t
    does not cleanly encode behavioral information, masking encoder deficiencies.
    The pressure to predict signals accurately must fall on the encoder and
    transition function, not be absorbed by an expressive head.

    Training instrument only. Not used at inference time — Stage 4 decodes
    z to a grid, re-simulates under exact GoL physics, and computes signals
    from the real trajectory.
    """

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 128, n_signals: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_signals),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim) or (B, T, latent_dim) float32
        Returns:
            signals: same leading shape + (n_signals,)
                     (B, 10) or (B, T, 10)
        """
        return self.net(z)
