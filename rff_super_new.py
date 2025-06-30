import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from math import log10

# coordinate grid
def get_rect_mgrid(width, height):

    xs = torch.linspace(-1, 1, steps=width)
    ys = torch.linspace(-1, 1, steps=height)
    X, Y = torch.meshgrid(xs, ys, indexing='xy') #[height, width]
    mgrid = torch.vstack((X.flatten(), Y.flatten())).T #[height*width, 2]
    return mgrid

# Random Fourier Features (RFF) layer
class RFFLayer(nn.Module):
    def __init__(self, in_features, mapping_size, scale=1.0):
        super().__init__()
        self.mapping_size = mapping_size
        self.B = nn.Parameter(scale * torch.randn((in_features, mapping_size)), requires_grad=False)

    def forward(self, x):
        # x: [batch, N, in_features]
        x_proj = x @ self.B  # [batch, N, mapping_size]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1) # [batch, N, 2 * mapping_size]

# ReLU MLP Layer
class MLPImplicitLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))

# Full Implicit Neural Representation (INR) model with RFF and ReLU
class INRModel(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        rff_mapping_size=256,
        rff_scale=1.0,
        outermost_linear=True
    ):
        super().__init__()
        layers = []

        # input layer with RFF
        layers.append(RFFLayer(in_features, rff_mapping_size, rff_scale))
        # output of RFFLayer
        layers.append(MLPImplicitLayer(2 * rff_mapping_size, hidden_features))

        # middle layers
        for _ in range(hidden_layers):
            layers.append(MLPImplicitLayer(hidden_features, hidden_features))

        # final layer
        if outermost_linear:
            layers.append(nn.Linear(hidden_features, out_features))
        else:
            layers.append(MLPImplicitLayer(hidden_features, out_features))

        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)  # [batch, N, out_features]
        return output, coords

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1.0 
    psnr = 20 * log10(max_pixel / np.sqrt(mse))
    return psnr

def main():
    img_path = "tray_image.png"
    try:
        img = Image.open(img_path).convert("L")
    except FileNotFoundError:
        print(f"Error: Image '{img_path}' not found. Please ensure the image is in the same directory as the script.")
        return

    # Resize the image to a low resolution
    original_size = img.size # (width, height)
    lr_width, lr_height = 64, 64
    img_lr = img.resize((lr_width, lr_height), Image.BILINEAR)
    arr_lr = np.array(img_lr).astype("float32") / 255.0 # shape [lr_height, lr_width], values in [0,1]

    H_lr, W_lr = arr_lr.shape

    # Flatten into (x,y) coords and pixel values for the low-resolution image
    coords_lr = get_rect_mgrid(W_lr, H_lr) # [lr_height*lr_width, 2]
    pixels_lr = torch.from_numpy(arr_lr).view(-1, 1) # [lr_height*lr_width, 1]

    device = torch.device("cpu") # Using CPU for simplicity

    coords_batch = coords_lr.unsqueeze(0).to(device) # [1, N, 2]
    pixels_batch = pixels_lr.unsqueeze(0).to(device) # [1, N, 1]

    # Instantiate model
    model = INRModel(
        in_features=2,     # (x,y)
        hidden_features=256,
        hidden_layers=3,   # middle layers
        out_features=1,    
        rff_mapping_size=256, # Size of random fourier features mapping B matrix
        rff_scale=5.0, # Scale for RFF, often tuned. Higher for more detail.
        outermost_linear=True # Final layer is linear activation
    ).to(device)

    print(model)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Reduced learning rate for stability

    # Early Stopping Parameters
    patience = 25
    min_delta = 1e-5
    best_loss = float('inf')
    patience_counter = 0

    EPOCHS = 250 # Increased epochs for better convergence with RFF

    print("\nStarting training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        out, _ = model(coords_batch) # out: [1, N, 1]
        loss = criterion(out, pixels_batch) # pixels_batch: [1, N, 1]

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == 1: # Print every 50 epochs or on first epoch
            print(f"Epoch {epoch}/{EPOCHS}   loss = {loss.item():.6f}")

        # Early Stopping
        if loss.item() + min_delta < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print("\nTraining complete.")

    # Super-resolution inference
    scale = 2 # factor for super-resolution
    #Hr, Wr = original_size[1] * scale, original_size[0] * scale # Note: PIL (width, height), numpy (height, width)
    Hr, Wr = lr_height * scale, lr_width * scale # PIL (width, height), numpy (height, width)


    # Build new coordinate grid at higher resolution
    coords_hr = get_rect_mgrid(Wr, Hr).to(device) # [Wr*Hr, 2]
    coords_hr_batch = coords_hr.unsqueeze(0) # [1, Wr*Hr, 2]

    # Run model for high-resolution reconstruction
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        recon_hr_tensor, _ = model(coords_hr_batch) # [1, Wr*Hr, 1]
        recon_hr = recon_hr_tensor.squeeze(0).cpu().view(Hr, Wr).numpy() # [Hr, Wr] float32 in [0,1]

    # bicubic upsampled version of the original high-res image as ground truth
    img_orig_hr = img.resize((Wr, Hr), Image.BILINEAR)
    arr_orig_hr = np.array(img_orig_hr).astype("float32") / 255.0

    # Bicubic interpolation baseline from your original PIL image
    bicubic_hr = np.array(img_lr.resize((Wr, Hr), Image.BILINEAR)).astype("float32") / 255.0

    # Calculate PSNR
    psnr_super_res = calculate_psnr(arr_orig_hr, recon_hr)
    psnr_bicubic = calculate_psnr(arr_orig_hr, bicubic_hr)

    print(f"\nPSNR of Super-Resolution Reconstruction: {psnr_super_res:.2f} dB")
    print(f"PSNR of Bicubic Interpolation Baseline: {psnr_bicubic:.2f} dB")

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(arr_lr, cmap="gray")
    axes[0].set_title(f"Low-Res Input {H_lr}x{W_lr}")
    axes[1].imshow(np.clip(recon_hr, 0, 1), cmap="gray")
    axes[1].set_title(f"INR Super-Res {Hr}x{Wr}\nPSNR: {psnr_super_res:.2f} dB")
    axes[2].imshow(np.clip(bicubic_hr, 0, 1), cmap="gray")
    axes[2].set_title(f"Bicubic Interp {Hr}x{Wr}\nPSNR: {psnr_bicubic:.2f} dB")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
