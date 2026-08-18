import torch
import torch.nn as nn


class SteeringNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Follows PilotNet (Nvidia) CNN
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), # 3 RGB layers of the image, outputs 24 stacked grids with smaller dimensions
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), 
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 1 * 18, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x.squeeze(1)


if __name__ == "__main__":
    model = SteeringNet()
    dummy = torch.randn(4, 3, 66, 200)
    out = model(dummy)
    print(out.shape)