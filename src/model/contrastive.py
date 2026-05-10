import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalContrastiveLoss(nn.Module):
    """
    Multiscale temporal contrastive loss on latent trajectories.

    For each trajectory in the batch, samples a triplet:
        anchor:   z at timestep t
        near:     z at timestep t + k  (short-term, same trajectory)
        far:      z at timestep t + K  (long-term, same trajectory)
        negative: z from a different trajectory

    Three simultaneous constraints:
        1. near closer to anchor than negative  (short-term coherence)
        2. far  closer to anchor than negative  (long-term identity)
        3. near closer to anchor than far       (temporal ordering)

    Phase 2: random negatives — any z from a different trajectory in the batch.
    Phase 3: hard negatives  — the cross-trajectory z geometrically closest to
             the anchor in current latent space. Hard negatives sharpen cluster
             boundaries because they force the encoder to distinguish
             near-identical vectors from different behavioral classes.

    Near lag k=5, far lag K=50, both fixed throughout training.
    A triplet can only be sampled when rollout depth >= K+k = 55.
    The training loop skips the contrastive loss when this is not satisfied.
    """

    def __init__(
        self,
        near_k: int   = 5,
        far_K: int    = 50,
        margin_near:  float = 0.5,
        margin_far:   float = 1.0,
        margin_order: float = 0.3,
    ):
        super().__init__()
        self.near_k       = near_k
        self.far_K        = far_K
        self.margin_near  = margin_near
        self.margin_far   = margin_far
        self.margin_order = margin_order

    def forward(
        self,
        z_traj: torch.Tensor,
        hard_negatives: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            z_traj:         (B, T, latent_dim) latent trajectory for each item
                            in the batch. T = rollout_depth + 1 (includes t=0).
            hard_negatives: if True, select hardest cross-trajectory negative
                            for each anchor; if False, use random negatives.
        Returns:
            loss: scalar, or 0.0 tensor if T < near_k + far_K + 1
        """
        B, T, D = z_traj.shape
        min_T = self.near_k + self.far_K + 1

        if T < min_T:
            return z_traj.new_zeros(1).squeeze()

        # Sample one anchor timestep per trajectory, ensuring t+K < T
        max_t = T - self.far_K - 1
        t = torch.randint(0, max_t + 1, (B,), device=z_traj.device)  # (B,)

        # Gather anchor, near, far for each item in batch
        idx  = torch.arange(B, device=z_traj.device)
        z_anchor = z_traj[idx, t]                    # (B, D)
        z_near   = z_traj[idx, t + self.near_k]      # (B, D)
        z_far    = z_traj[idx, t + self.far_K]       # (B, D)

        # Negative: cross-trajectory z at the same anchor timestep
        if hard_negatives:
            z_neg = self._hard_negatives(z_anchor, z_traj, t)
        else:
            z_neg = self._random_negatives(z_anchor, z_traj, t)

        return self._triplet_loss(z_anchor, z_near, z_far, z_neg)

    # ------------------------------------------------------------------
    def _random_negatives(
        self, z_anchor: torch.Tensor, z_traj: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Random cross-trajectory negative at anchor timestep."""
        B = z_anchor.shape[0]
        # Circular shift of batch indices ensures every anchor gets a
        # negative from a different trajectory without any index overlap.
        shift = torch.randint(1, B, (1,)).item()
        neg_idx = (torch.arange(B, device=z_traj.device) + shift) % B
        return z_traj[neg_idx, t]  # (B, D)

    def _hard_negatives(
        self, z_anchor: torch.Tensor, z_traj: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Hard negative mining: for each anchor i at timestep t[i], find the z
        from a different trajectory j at the same timestep t[i] that is closest
        in Euclidean distance to z_anchor[i].

        O(B^2) per anchor. B=32 makes this negligible.
        """
        B, D = z_anchor.shape
        z_neg = torch.empty_like(z_anchor)
        for i in range(B):
            # All trajectories at anchor i's timestep: (B, D)
            z_at_ti = z_traj[:, t[i]]
            dists = ((z_at_ti - z_anchor[i].unsqueeze(0)) ** 2).sum(dim=-1)  # (B,)
            dists[i] = float('inf')   # exclude self
            z_neg[i] = z_at_ti[dists.argmin()]
        return z_neg

    def _triplet_loss(
        self,
        z_anchor: torch.Tensor,
        z_near:   torch.Tensor,
        z_far:    torch.Tensor,
        z_neg:    torch.Tensor,
    ) -> torch.Tensor:
        d = F.pairwise_distance

        # Constraint 1: near closer to anchor than negative
        loss_near = torch.relu(
            d(z_anchor, z_near) - d(z_anchor, z_neg) + self.margin_near
        ).mean()

        # Constraint 2: far closer to anchor than negative
        loss_far = torch.relu(
            d(z_anchor, z_far) - d(z_anchor, z_neg) + self.margin_far
        ).mean()

        # Constraint 3: near closer to anchor than far (temporal ordering)
        loss_order = torch.relu(
            d(z_anchor, z_near) - d(z_anchor, z_far) + self.margin_order
        ).mean()

        return loss_near + loss_far + loss_order
