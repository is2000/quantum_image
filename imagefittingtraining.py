from imagefitting1 import HybridImageFittingModel
from imagefitting1 import quantum_circuit
import torch
from torch import nn
import torch.optim as optim
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

'''class ToyImageRegressionDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.images = torch.rand(num_samples, 1, 16, 16)  # Random grayscale images
        self.targets = self.images.view(num_samples, -1).mean(dim=1)  # Average pixel intensity as target

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]

dataset = ToyImageRegressionDataset(num_samples=1000)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
'''

def generate_image_from_model(model, width, height):
    model.eval()
    coords = [(x / width, y / height) for y in range(height) for x in range(width)]
    coords_tensor = torch.tensor(coords, dtype=torch.float32)
    with torch.no_grad():
        preds = model(coords_tensor).view(height, width).cpu().numpy()
    return preds

class ImageRegressionDataset(Dataset):
    def __init__(self, image_path):
        # Load image as grayscale
        img = Image.open(image_path).convert('L')  # 'L' mode = grayscale
        img = img.resize((64, 64)) 
       # img.show()
        img.save("resized_image.png")
        img = np.asarray(img, dtype=np.float32) / 255.0

        self.height, self.width = img.shape
        self.data = []

        # Generate (x, y) coordinates and corresponding pixel values
        for y in range(self.height):
            for x in range(self.width):
                norm_x = x / (self.width - 1)
                norm_y = y / (self.height - 1)
                pixel_val = img[y, x]
                self.data.append(((norm_x, norm_y), pixel_val))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        coords, value = self.data[idx]
        return torch.tensor(coords, dtype=torch.float32), torch.tensor([value], dtype=torch.float32)

from torch.utils.data import DataLoader

dataset = ImageRegressionDataset("/home/ishita/Desktop/quantum/trial code/image.png")
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = HybridImageFittingModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

epochs = 100

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        targets = targets.view(-1, 1).float()
        optimizer.zero_grad()
        inputs = inputs.detach()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        scheduler.step(loss.item())

    epoch_loss = running_loss / len(dataset)
    if (epoch + 1)%10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")

    if (epoch + 1)%100 == 0:
        pred_img = generate_image_from_model(model, dataset.width, dataset.height)
        plt.imshow(pred_img, cmap = 'gray')
        plt.title(f"Epoch {epoch + 1}")
        plt.axis('off')
        plt.show()
        plt.savefig("predicted_image.png")