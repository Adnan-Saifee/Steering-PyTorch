import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import DrivingDataset, time_split
from model import SteeringNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda" if torch.cuda.is_available() else "cpu")

def crop_sky(img):
    width, height = img.size
    return img.crop((0, int(height * 0.4), width, height))
 
transform = transforms.Compose([
    transforms.Lambda(crop_sky),
    transforms.Resize((66, 200)),
    transforms.ToTensor(),
])

train_df, val_df, test_df = time_split("data/labels.csv")
train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

train_set = DrivingDataset("data/train.csv", "data/raw/images/data", transform=transform)
val_set = DrivingDataset("data/val.csv", "data/raw/images/data", transform=transform)

# Automatically creates batches so we don't have to compute gradients of all the data one at a time
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

model = SteeringNet().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 20
train_losses = []
val_losses = []
best_val_loss = float("inf")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, angles in train_loader:
        images, angles = images.to(device), angles.to(device).float()

        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, angles)
        loss.backward()
        optimizer.step()

        # loss * num images in set
        running_loss += loss.item() * images.size(0)

    # Gives you average training loss per epoch
    train_loss = running_loss / len(train_set)
    train_losses.append(train_loss)

    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for images, angles in val_loader:
            images, angles = images.to(device), angles.to(device).float()
            preds = model(images)
            loss = criterion(preds, angles)
            running_val_loss += loss.item() * images.size(0)

    val_loss = running_val_loss / len(val_set)
    val_losses.append(val_loss)

    print(f"epoch {epoch+1}/{epochs} - train loss: {train_loss:.4f} - val loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "models/checkpoints/best.pt")

plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.xlabel("epoch")
plt.ylabel("MSE loss")
plt.legend()
plt.savefig("results/loss_curve.png")