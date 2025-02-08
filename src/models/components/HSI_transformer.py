from torch import nn, Tensor


# Hyperspectral Encoder (3D CNN)
class HSIEncoder(nn.Module):

    def __init__(self,
                 input_channels: int = 220,
                 embed_dim=512) -> None:

        super().__init__()

        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.fc = nn.Linear(64, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv3d(x)  # (batch, 64, 1, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x
