import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import matplotlib.pyplot as plt


# -------------------------------
# RFF + ReLU Model
# -------------------------------
class RFFReLURegression(nn.Module):
    def __init__(self, input_dim=2, rff_dim=512, output_dim=1, sigma=0.1):
        super().__init__()
        self.register_buffer("W", torch.randn(input_dim, rff_dim) / sigma)
        self.register_buffer("b", 2 * np.pi * torch.rand(rff_dim))
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(rff_dim, output_dim)

    def rff_mapping(self, x):
        projection = x @ self.W + self.b
        return torch.cos(projection) * np.sqrt(2.0 / self.W.shape[1])

    def forward(self, x):
        features = self.rff_mapping(x)
        return self.output_layer(self.relu(features))


# -------------------------------
# PSNR Function
# -------------------------------
def compute_psnr(original, reconstructed, max_val=1.0):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(max_val ** 2 / mse)


# -------------------------------
# Load and Prepare Data
# -------------------------------
def prepare_data(img_path, low_res_size=(8, 8), high_res_size=(16, 16)):
    # Load high-res ground truth
    hr_img = Image.open(img_path).convert('L').resize(high_res_size)
    hr_array = np.array(hr_img, dtype=np.float32) / 255.0

    # Downsample to low-res for training
    lr_img = hr_img.resize(low_res_size, resample=Image.BICUBIC)
    lr_array = np.array(lr_img, dtype=np.float32) / 255.0

    # Prepare coordinates and targets
    H_lr, W_lr = lr_array.shape
    xs_lr = np.linspace(0, 1, W_lr)
    ys_lr = np.linspace(0, 1, H_lr)
    xx_lr, yy_lr = np.meshgrid(xs_lr, ys_lr)
    coords_lr = np.stack([xx_lr.ravel(), yy_lr.ravel()], axis=-1)
    targets_lr = lr_array.ravel().reshape(-1, 1)

    return coords_lr, targets_lr, hr_array


# -------------------------------
# Main Super-Resolution Pipeline
# -------------------------------
def run_super_resolution(img_path):
    coords_lr, targets_lr, hr_array = prepare_data(img_path)

    coords_tensor = torch.tensor(coords_lr, dtype=torch.float32)
    targets_tensor = torch.tensor(targets_lr, dtype=torch.float32)
    dataloader = DataLoader(TensorDataset(coords_tensor, targets_tensor), batch_size=32, shuffle=True)

    model = RFFReLURegression()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(300):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    # Predict on high-res grid
    H_hr, W_hr = hr_array.shape
    xs_hr = np.linspace(0, 1, W_hr)
    ys_hr = np.linspace(0, 1, H_hr)
    xx_hr, yy_hr = np.meshgrid(xs_hr, ys_hr)
    coords_hr = np.stack([xx_hr.ravel(), yy_hr.ravel()], axis=-1)
    coords_hr_tensor = torch.tensor(coords_hr, dtype=torch.float32)

    with torch.no_grad():
        preds_hr = model(coords_hr_tensor).numpy().reshape(H_hr, W_hr)

    # PSNR
    psnr = compute_psnr(hr_array, preds_hr)
    print(f"\nPSNR of Super-Resolved Image: {psnr:.2f} dB")

    # Visualization
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Original HR")
    plt.imshow(hr_array, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Low-Res Input")
    plt.imshow(Image.open(img_path).convert('L').resize((8, 8)), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Super-Resolved (RFF+ReLU)")
    plt.imshow(preds_hr, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    run_super_resolution("tray_image.png")  # ← Replace with your path
