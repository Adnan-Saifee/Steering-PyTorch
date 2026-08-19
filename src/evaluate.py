import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import DrivingDataset
from model import SteeringNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Lambda(lambda img: img.crop((0, int(img.size[1] * 0.4), img.size[0], img.size[1]))),
    transforms.Resize((66, 200)),
    transforms.ToTensor(),
])

# This time comparing against test_set
test_set = DrivingDataset("data/test.csv", "data/raw/images/data", transform=transform)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

model = SteeringNet().to(device)
model.load_state_dict(torch.load("models/checkpoints/best.pt", map_location=device))
model.eval()

criterion = nn.MSELoss()
total_loss = 0.0
all_preds = []
all_angles = []

with torch.no_grad():
    for images, angles in test_loader:
        images, angles = images.to(device), angles.to(device).float()
        preds = model(images)
        loss = criterion(preds, angles)
        total_loss += loss.item() * images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_angles.extend(angles.cpu().numpy())

test_loss = total_loss / len(test_set)
all_preds = np.array(all_preds)
all_angles = np.array(all_angles)
# Added mae and rmse as well
mae = np.mean(np.abs(all_preds - all_angles))
rmse = np.sqrt(np.mean((all_preds - all_angles) ** 2))

print(f"test MSE loss: {test_loss:.4f}")
print(f"test MAE: {mae:.4f} degrees")
print(f"test RMSE: {rmse:.4f} degrees")

plt.figure(figsize=(10, 6))
plt.scatter(all_angles, all_preds, alpha=0.5, s=10)
plt.xlabel("actual angle")
plt.ylabel("predicted angle")
plt.title("Predicted vs Actual Steering Angle")
plt.plot([-10, 10], [-10, 10], 'r--', label="perfect prediction")
plt.legend()
plt.grid()
plt.savefig("results/sample_predictions.png", dpi=100, bbox_inches="tight")
print("Saved plot to results/sample_predictions.png")