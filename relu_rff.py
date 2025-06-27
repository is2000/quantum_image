import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Load image (grayscale)
img_path = "/home/ishita/Desktop/quantum/trial code/tray_image.png"
img = Image.open(img_path).convert('L') 
img = img.resize((16, 16))
img_array = np.array(img) / 255.0

# (x, y) coordinate grid
H, W = img_array.shape
xs = np.linspace(0, 1, W)
ys = np.linspace(0, 1, H)
xx, yy = np.meshgrid(xs, ys)
coords = np.stack([xx.ravel(), yy.ravel()], axis=-1) 
targets = img_array.ravel().reshape(-1, 1)  
# Convert to tensors
coords_tensor = torch.tensor(coords, dtype=torch.float32)
targets_tensor = torch.tensor(targets, dtype=torch.float32)

# DataLoader (optional)
dataset = TensorDataset(coords_tensor, targets_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

class RFFReLURegression(nn.Module):
    def __init__(self, input_dim=2, rff_dim=256, output_dim=1, sigma=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.rff_dim = rff_dim
        self.output_dim = output_dim
        self.sigma = sigma

        # Sample random weights and bias for RFF mapping
        self.register_buffer("W", torch.randn(input_dim, rff_dim) / sigma)
        self.register_buffer("b", 2 * np.pi * torch.rand(rff_dim))

        # Layers
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(rff_dim, output_dim)

    def rff_mapping(self, x):
        projection = x @ self.W + self.b  # shape: [batch_size, rff_dim]
        return torch.cos(projection) * np.sqrt(2.0 / self.rff_dim)

    def forward(self, x):
        # x: [batch_size, 2] (e.g., x, y coordinates)
        rff_feats = self.rff_mapping(x)  # [batch_size, rff_dim]
        activated = self.relu(rff_feats)
        return self.output_layer(activated)  # [batch_size, 1]

model = RFFReLURegression(rff_dim=512, sigma=0.2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(300):
    total_loss = 0.0
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

with torch.no_grad():
    predicted = model(coords_tensor).numpy().reshape(H, W)

def compute_psnr(original, reconstructed, max_pixel_value=1.0):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')  # Perfect reconstruction
    psnr = 10 * np.log10(max_pixel_value ** 2 / mse)
    return psnr

# Calculate PSNR
psnr_value = compute_psnr(img_array, predicted)
print(f"PSNR: {psnr_value:.2f} dB")


plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(img_array, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Reconstructed (RFF+ReLU)")
plt.imshow(predicted, cmap='gray')
plt.show()

