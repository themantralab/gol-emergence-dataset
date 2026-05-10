import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    128x128 binary grid -> z in R^128.

    4 conv layers with stride-2 downsampling collapse spatial resolution:
        128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8
    Channel progression 1->32->64->128->256 builds representational depth
    as spatial resolution decreases.

    Flatten 256*8*8=16384, then a single linear projection to 128.
    No output activation: z is raw. VICReg enforces variance and decorrelation
    during training without imposing a distributional shape.

    BatchNorm+ReLU after each conv. BN removes the dying-neuron risk that
    makes ReLU problematic in deeper networks and stabilises training without
    adding a Gaussian prior to the latent space.
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv = nn.Sequential(
            # 1 x 128 x 128 -> 32 x 64 x 64
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 32 x 64 x 64 -> 64 x 32 x 32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 64 x 32 x 32 -> 128 x 16 x 16
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 128 x 16 x 16 -> 256 x 8 x 8
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # 256 * 8 * 8 = 16384 -> latent_dim
        self.fc = nn.Linear(256 * 8 * 8, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 128, 128) float32 in [0, 1]
        Returns:
            z: (B, latent_dim) float32, raw (no activation)
        """
        h = self.conv(x)           # (B, 256, 8, 8)
        h = h.flatten(start_dim=1) # (B, 16384)
        return self.fc(h)          # (B, latent_dim)
