import torch
import torch.nn as nn


class Transition(nn.Module):
    """
    Latent transition function: z_t -> z_{t+1}.

    3-layer MLP, hidden dim 256, SiLU activations.
    Implemented as a residual network: output = z + MLP(z).

    Residual formulation matters here because:
    - z_t and z_{t+1} are close for most GoL patterns (one step is a small
      change). Predicting the delta is numerically easier than predicting the
      full next state from scratch.
    - At random initialisation, MLP(z) ≈ 0, so the network starts near the
      identity. This prevents wild rollout divergence in the first training
      steps before any useful gradient has flowed.
    - Gradients flow cleanly through the skip connection during the long
      rollouts (up to 256 chained applications) that Phase 3 requires.

    Deliberately small (hidden=256) so the network cannot memorise specific
    z->z mappings from the training set. Forces it to learn the abstract
    GoL transition rule that generalises to novel z vectors at inference.
    """

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim) float32
        Returns:
            z_next: (B, latent_dim) float32
        """
        return z + self.net(z)

    def rollout(self, z0: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Unroll the transition function for `steps` steps.

        Args:
            z0:    (B, latent_dim) initial latent states
            steps: number of steps to unroll
        Returns:
            trajectory: (B, steps+1, latent_dim) including z0 at index 0
        """
        traj = [z0]
        z = z0
        for _ in range(steps):
            z = self.forward(z)
            traj.append(z)
        return torch.stack(traj, dim=1)  # (B, steps+1, latent_dim)
