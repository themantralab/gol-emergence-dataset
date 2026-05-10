import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    z in R^128 -> (1, 128, 128) logits.

    Exact mirror of Encoder: linear expands z back to 256*8*8, then four
    transposed conv layers (stride-2) restore spatial resolution:
        8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
    Channel progression 256->128->64->32->1.

    Output is raw logits (no sigmoid). Apply sigmoid for BCE loss;
    threshold at 0.5 to reconstruct a binary grid.

    BatchNorm+ReLU after each transposed conv except the last, which outputs
    logits directly.
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)

        self.deconv = nn.Sequential(
            # 256 x 8 x 8 -> 128 x 16 x 16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 128 x 16 x 16 -> 64 x 32 x 32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 64 x 32 x 32 -> 32 x 64 x 64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 32 x 64 x 64 -> 1 x 128 x 128  (no BN, no activation — raw logits)
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1, bias=True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim) float32
        Returns:
            logits: (B, 1, 128, 128) float32, raw (no sigmoid)
        """
        h = self.fc(z)                      # (B, 256*8*8)
        h = h.view(-1, 256, 8, 8)           # (B, 256, 8, 8)
        return self.deconv(h)               # (B, 1, 128, 128)

    def decode_binary(self, z: torch.Tensor) -> torch.Tensor:
        """Convenience: decode z to a binary uint8 grid (no gradient)."""
        with torch.no_grad():
            logits = self.forward(z)
            return (logits.squeeze(1) > 0).to(torch.uint8)  # (B, 128, 128)
