# -*- coding: utf-8 -*-

"""
Training script for patch-based learning of log-normal permeability parameters 
from mechanical aperture patches using a Residual U-Net.

The training inputs are mechanical aperture patches a_m(x,y).
The supervised targets are the log-normal parameters mu(x,y) and sigma(x,y),
computed from posterior predictive permeability moments obtained from the
physics-based Bayesian correction.

Related paper:
    Probabilistic Upscaling of Hydrodynamics in Geological Fractures Under Uncertainty

See Perez et al. (2025), Geophysical Research Letters 
https://doi.org/10.1029/2025GL117776
for the Bayesian correction used to generate the posterior permeability moments
stored in Dataset_apertures_to_perms.npy

"""

import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


# =============================================================================
# Reproducibility
# =============================================================================
def set_all_seeds(seed=42):
    """ Set random seeds for reproducibility """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# =============================================================================
# Paired dataset: mechanical aperture patches -> log-normal permeability parameters
# =============================================================================
class PatchDataset(Dataset):
    """
    Convert posterior predictive moments of K(x,y) into log-normal parameters
    so that K(x,y) ~ LogNormal(mu(x,y), sigma(x,y)^2)
    """
    def __init__(self, a_m, K_mean, K_std, eps=1e-8):
        self.a_m = torch.tensor(a_m).unsqueeze(1).float()  # input aperture patches, shape [N, 1, H, W]

        K_mean = torch.tensor(K_mean).float()
        K_std = torch.tensor(K_std).float()

        # Avoid numerical issues before log-normal conversion
        K_mean = torch.clamp(K_mean, min=eps)
        K_std = torch.clamp(K_std, min=eps)

        # Convert posterior moments into log-normal parameters 
        variance_ratio = (K_std / K_mean) ** 2
        sigma2 = torch.log(1 + variance_ratio)
        sigma = torch.sqrt(sigma2)
        mu = torch.log(K_mean) - 0.5 * sigma2

        self.target = torch.stack([mu, sigma], dim=1)   # target fields [mu(x,y), sigma(x,y)], shape [N, 2, H, W]

    def __len__(self):
        return len(self.a_m)

    def __getitem__(self, idx):
        return self.a_m[idx], self.target[idx]

# =============================================================================
# Residual U-Net surrogate for patch-wise prediction of mu(x,y) and sigma(x,y)
# =============================================================================
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, dropout_rate=0.1, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU()

        if dropout:
            self.dropout1 = nn.Dropout2d(dropout_rate)
            self.dropout2 = nn.Dropout2d(dropout_rate)
        else:
            self.dropout1 = nn.Identity()
            self.dropout2 = nn.Identity()

        # Use GroupNorm with a valid divisor of out_channels
        self.groups = min(groups, out_channels)
        while self.groups > 1 and out_channels % self.groups != 0:
            self.groups //= 2

        self.norm1 = nn.GroupNorm(self.groups, out_channels)
        self.norm2 = nn.GroupNorm(self.groups, out_channels)

        # Match shortcut dimensions when channel count changes
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        return x + identity


class DownBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, dropout=False, dropout_rate=0.1):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResNetBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                dropout=dropout,
                dropout_rate=dropout_rate
            ) for i in range(num_layers)
        ])
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout else nn.Identity()

    def forward(self, x):
        hidden_states = x
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        downsampled = self.downsample(hidden_states)
        return self.dropout(downsampled), hidden_states  # downsampled output + skip features


class UpBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, dropout=False, dropout_rate=0.1):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResNetBlock(
                2 * in_channels if i == 0 else out_channels,
                out_channels,
                dropout=dropout,
                dropout_rate=dropout_rate
            ) for i in range(num_layers)
        ])
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout else nn.Identity()

    def forward(self, hidden_states, res_hidden_states):
        hidden_states = self.upsample(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        return hidden_states


class UNetOutputMuSigma(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Separate output heads for the two log-normal permeability parameters
        self.mu_head = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.sigma_head = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),
            nn.Softplus()  # enforce sigma(x,y) > 0
        )

    def forward(self, x):
        mu = self.mu_head(x)
        sigma = self.sigma_head(x)
        return torch.cat([mu, sigma], dim=1)


class UNetResidual(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=32, dropout=False, dropout_rate=0.1):
        super().__init__()

        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder
        self.down1 = DownBlock2D(base_channels, base_channels * 2, dropout=dropout, dropout_rate=dropout_rate)
        self.down2 = DownBlock2D(base_channels * 2, base_channels * 4, dropout=dropout, dropout_rate=dropout_rate)
        self.down3 = DownBlock2D(base_channels * 4, base_channels * 8, dropout=dropout, dropout_rate=dropout_rate)

        # Bottleneck
        self.mid_resnet1 = ResNetBlock(base_channels * 8, base_channels * 8, dropout=dropout, dropout_rate=dropout_rate)
        self.mid_resnet2 = ResNetBlock(base_channels * 8, base_channels * 8, dropout=dropout, dropout_rate=dropout_rate)

        # Decoder
        self.up3 = UpBlock2D(base_channels * 8, base_channels * 4, dropout=dropout, dropout_rate=dropout_rate)
        self.up2 = UpBlock2D(base_channels * 4, base_channels * 2, dropout=dropout, dropout_rate=dropout_rate)
        self.up1 = UpBlock2D(base_channels * 2, base_channels, dropout=dropout, dropout_rate=dropout_rate)

        self.conv_out = UNetOutputMuSigma(base_channels)

    def forward(self, x):
        x = self.conv_in(x)

        down1, skip1 = self.down1(x)
        down2, skip2 = self.down2(down1)
        down3, skip3 = self.down3(down2)

        mid = self.mid_resnet1(down3)
        mid = self.mid_resnet2(mid)

        up3 = self.up3(mid, skip3)
        up2 = self.up2(up3, skip2)
        up1 = self.up1(up2, skip1)

        return self.conv_out(up1)


# =============================================================================
# Multi-output loss: data misfit + Sobel edge regularisation + GradNorm weighting
# =============================================================================
def sobel_gradient(image):
    """Compute Sobel gradient magnitude for a batch of 2D images."""
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)

    image = image.unsqueeze(1)
    grad_x = F.conv2d(image, sobel_x, padding=1)
    grad_y = F.conv2d(image, sobel_y, padding=1)
    return torch.sqrt(grad_x**2 + grad_y**2 + 1e-8).squeeze(1)


class MultiOutputLoss_GradNorm_EdgeReg(nn.Module):
    def __init__(self, model, mode="last_layer", alpha=0.0, lambda_edge=0.1, epsilon=1e-8, gradnorm_every=1):
        super().__init__()
        self.model = model
        self.mode = mode
        self.alpha = alpha
        self.lambda_edge = lambda_edge
        self.epsilon = epsilon
        self.gradnorm_every = gradnorm_every
        self.step_count = 0

        # Learnable weights for the two regression tasks
        self.loss_weights = nn.Parameter(torch.zeros(2))

        self.initial_losses = None
        self.gradnorm_params = self._select_params()

        # Optional logging containers
        self.loss_mu_history = []
        self.loss_sigma_history = []
        self.loss_grad_mu_history = []
        self.loss_grad_sigma_history = []

    def _select_params(self):
        if self.mode == "last_layer":
            return list(self.model.conv_out.parameters())
        elif self.mode == "decoder":
            return (
                list(self.model.up1.parameters()) +
                list(self.model.up2.parameters()) +
                list(self.model.up3.parameters()) +
                list(self.model.conv_out.parameters())
            )
        elif self.mode == "full":
            return list(self.model.parameters())
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def forward(self, predictions, targets, model_params, return_components=False):
        pred_mu = predictions[:, 0, :, :]
        pred_sigma = predictions[:, 1, :, :]
        target_mu = targets[:, 0, :, :]
        target_sigma = targets[:, 1, :, :]

        # Data-fitting losses for the predicted log-normal parameters
        loss_mu = F.mse_loss(pred_mu, target_mu)
        loss_sigma = F.mse_loss(pred_sigma, target_sigma)

        # Sobel-based edge regularisation to preserve spatial permeability structures
        grad_pred_mu = sobel_gradient(pred_mu)
        grad_target_mu = sobel_gradient(target_mu.detach())
        loss_grad_mu = F.mse_loss(grad_pred_mu, grad_target_mu)

        grad_pred_sigma = sobel_gradient(pred_sigma)
        grad_target_sigma = sobel_gradient(target_sigma.detach())
        loss_grad_sigma = F.mse_loss(grad_pred_sigma, grad_target_sigma)

        # Learnable task weights constrained through softmax normalisation
        losses_value = torch.stack([loss_mu, loss_sigma])
        weights = torch.softmax(self.loss_weights, dim=0) * 2
        weights = weights.to(losses_value.device)

        if self.initial_losses is None:
            self.initial_losses = losses_value.detach()
            print("Initial value losses:", self.initial_losses)

        weighted_value_loss = (weights * losses_value).sum()
        edge_loss = self.lambda_edge * (weights[0] * loss_grad_mu + weights[1] * loss_grad_sigma)
        total_loss = weighted_value_loss + edge_loss

        # Apply GradNorm only to the selected regression losses every N steps
        # The gradient-based regularisation terms are excluded from the GradNorm objective
        if torch.is_grad_enabled() and (self.step_count % self.gradnorm_every == 0):
            G = []
            G_raw = []

            for i, loss_i in enumerate(losses_value):
                grads = torch.autograd.grad(
                    loss_i,
                    self.gradnorm_params,
                    retain_graph=True,
                    create_graph=True,
                    allow_unused=True
                )
                raw_norm = torch.sqrt(sum((g ** 2).sum() for g in grads if g is not None))
                G_raw.append(raw_norm)

                weighted_norm = torch.sqrt(sum(((weights[i] * g) ** 2).sum() for g in grads if g is not None))
                G.append(weighted_norm)

            G = torch.stack(G)
            G_raw = torch.stack(G_raw)
            G_avg = G.mean()

            with torch.no_grad():
                relative_loss = losses_value.detach() / (self.initial_losses + self.epsilon)
                r = relative_loss / relative_loss.mean()
                target_G = G_avg * (r ** self.alpha)

            gradnorm_loss = nn.L1Loss()(G, target_G.detach())
            gradnorm_loss.backward(retain_graph=True)

        self.step_count += 1

        if return_components:
            comps = {
                "loss_mu": loss_mu.detach().item(),
                "loss_sigma": loss_sigma.detach().item(),
                "loss_grad_mu": loss_grad_mu.detach().item(),
                "loss_grad_sigma": loss_grad_sigma.detach().item(),
                "w_mu": float(weights.detach()[0].cpu()),
                "w_sigma": float(weights.detach()[1].cpu()),
            }
            return total_loss, comps

        return total_loss

# =============================================================================
# Helpers
# =============================================================================
def load_patch_data(file_path):
    patch = np.load(file_path, allow_pickle=True)
    a_m = patch.item().get('a_m')          # mechanical aperture patches a_m(x,y)
    K_mean = patch.item().get('K_mean')    # posterior predictive mean E[K(x,y)]
    K_std = patch.item().get('K_std')      # posterior predictive standard deviation std[K(x,y)]
    return a_m, K_mean, K_std


def make_train_val_loaders(a_m, K_mean, K_std, batch_size=32, seed=42):
    """
    Split the paired patch dataset into training and validation subsets.
    
    The split is performed at the fracture-realisation level before augmentation,
    so that all augmented patches derived from the same realisation remain in the
    same subset and no spatial information leaks across splits.
    """
    full_dataset = PatchDataset(a_m, K_mean, K_std)

    N = len(full_dataset)     # 560 patches
    n_aug = 4                 # number of augmented versions per simulation
    n_sim = N // n_aug        # 140 base simulations
    assert N == n_sim * n_aug

    # Keep augmented patches from the same simulation in the same split
    sim_id = np.arange(N) // n_aug

    rng = np.random.default_rng(seed)
    sim_ids = np.arange(n_sim)
    rng.shuffle(sim_ids)

    n_train_sim = int(0.8 * n_sim)
    train_size = n_train_sim * n_aug
    val_size = N - train_size

    train_sim = set(sim_ids[:n_train_sim])
    val_sim = set(sim_ids[n_train_sim:])

    train_idx = np.where(np.isin(sim_id, list(train_sim)))[0].tolist()
    val_idx = np.where(np.isin(sim_id, list(val_sim)))[0].tolist()

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return full_dataset, train_loader, val_loader, train_idx, val_idx, train_size, val_size


def plot_patch_example(a_m, K_mean, K_std, save_dir, ind=250):

    fig, ax = plt.subplots(1, 4, figsize=(15, 6), constrained_layout=True)

    im0 = ax[0].imshow(a_m[ind, :, :])
    cbar0 = fig.colorbar(im0, ax=ax[0], orientation="horizontal", shrink=0.8, pad=0.04)
    cbar0.set_label(r"[$\mu m$]")
    ax[0].set_title("Mechanical Aperture:\n" + r"$a_m(x,y)$")
    ax[0].axis("off")

    im1 = ax[1].imshow(K_mean[ind, :, :])
    cbar1 = fig.colorbar(im1, ax=ax[1], orientation="horizontal", shrink=0.8, pad=0.04)
    cbar1.set_label(r"[$\mu m^2$]")
    ax[1].set_title("Posterior Predictive Mean:\n" + r"$\mathbb{E}[K(x,y)]$")
    ax[1].axis("off")

    im2 = ax[2].imshow(K_std[ind, :, :])
    cbar2 = fig.colorbar(im2, ax=ax[2], orientation="horizontal", shrink=0.8, pad=0.04)
    cbar2.set_label(r"[$\mu m^2$]")
    ax[2].set_title("Posterior Predictive Std: \n" + r"$\sqrt{\mathrm{Var}[K(x,y)]}$")
    ax[2].axis("off")

    im3 = ax[3].imshow(a_m[ind, :, :] ** 2 / 12)
    cbar3 = fig.colorbar(im3, ax=ax[3], orientation="horizontal", shrink=0.8, pad=0.04)
    cbar3.set_label(r"[$\mu m^2$]")
    ax[3].set_title("Local Cubic Law:\n" + r"$K_{CL}^{a_m}(x,y)=a_m(x,y)^2/12$")
    ax[3].axis("off")

    file_path = os.path.join(save_dir, "Patch_visualisation.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    

# =============================================================================
# Main Training loop 
# =============================================================================
def main():
    set_all_seeds()

    fig_dir = "./figs"                                              # Directory for generated figures
    outputs_dir = "./outputs"                                       # Directory for intermediate results
    model_dir = "./models"                                          # Direction with pre-trained Residual U-Net model
    patch_path = "./dataset/Dataset_apertures_to_perms.npy"         # Apertures data on the patches (Training only) 

    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"Is CUDA available?: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name())

    # -------------------------
    # Load data
    # -------------------------
    a_m, K_mean, K_std = load_patch_data(patch_path)
    print(f"Dataset shape (N, Nx, Ny): {a_m.shape}")

    # Optional quick visualisation
    plot_patch_example(a_m, K_mean, K_std, save_dir=fig_dir)

    # -------------------------
    # Data split / loaders
    # -------------------------
    full_dataset, train_loader, val_loader, train_idx, val_idx, train_size, val_size = make_train_val_loaders(a_m, K_mean, K_std, batch_size=32)

    print("Train size", train_size, "Validation size", val_size)

    # -------------------------
    # Model Setup
    # -------------------------
    model = UNetResidual(in_channels=1, out_channels=2, base_channels=32, dropout=True, dropout_rate=0.15).to(device)

    input_shape = (1, 128, 128)
    summary(model, input_shape)

    # -------------------------
    # Loss optimisation
    # -------------------------
    n_epochs = 1000
    return_components = False

    criterion = MultiOutputLoss_GradNorm_EdgeReg(
        model=model,
        mode="last_layer",
        alpha=0.1,
        lambda_edge=1.0,
        gradnorm_every=20
    )
    
    # Optimise both network parameters and learnable loss weights
    all_params = list({p for p in model.parameters()}.union(criterion.parameters()))
    
    optimizer = AdamW(all_params, lr=1e-3, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    loss_batch_list = []
    train_loss_list = []
    val_loss_list = []

    if return_components:
        mu_list_train = []
        mu_list_val = []
        grad_mu_list_train = []
        grad_mu_list_val = []
        sigma_list_train = []
        sigma_list_val = []
        grad_sigma_list_train = []
        grad_sigma_list_val = []
        w_mu_list = []
        w_sig_list = []

   
    start_time = time.time()

    for epoch in range(n_epochs):
        # -------------------------
        # Training 
        # -------------------------
        model.train()
        train_loss = 0.0
        train_batches = 0

        sum_mu = sum_sig = 0.0
        sum_gmu = sum_gsig = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}")
        for batch in progress_bar:
            a_m_batch = batch[0].to(device)
            K_targets = batch[1].to(device)

            optimizer.zero_grad()
            pred = model(a_m_batch)

            if return_components:
                loss, comps = criterion(pred, K_targets, list(model.parameters()), return_components=True)
                sum_mu += comps["loss_mu"]
                sum_sig += comps["loss_sigma"]
                sum_gmu += comps["loss_grad_mu"]
                sum_gsig += comps["loss_grad_sigma"]
            else:
                loss = criterion(pred, K_targets, list(model.parameters()))

            loss_batch_list.append(loss.item())

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1
            progress_bar.set_postfix({"train_loss": f"{train_loss / train_batches:.6f}"})

        train_loss /= train_batches
        train_loss_list.append(train_loss)

        if return_components:
            train_mu = sum_mu / train_batches
            train_sig = sum_sig / train_batches
            train_gmu = sum_gmu / train_batches
            train_gsig = sum_gsig / train_batches

            mu_list_train.append(train_mu)
            grad_mu_list_train.append(train_gmu)
            sigma_list_train.append(train_sig)
            grad_sigma_list_train.append(train_gsig)

            w_mu = comps["w_mu"]
            w_sig = comps["w_sigma"]

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        val_loss = 0.0
        val_batches = 0
        sum_mu = sum_sig = 0.0
        sum_gmu = sum_gsig = 0.0

        with torch.no_grad():
            for batch in val_loader:
                a_m_batch = batch[0].to(device)
                K_targets = batch[1].to(device)
                pred = model(a_m_batch)

                if return_components:
                    loss, comps = criterion(pred, K_targets, list(model.parameters()), return_components=True)
                    sum_mu += comps["loss_mu"]
                    sum_sig += comps["loss_sigma"]
                    sum_gmu += comps["loss_grad_mu"]
                    sum_gsig += comps["loss_grad_sigma"]
                else:
                    loss = criterion(pred, K_targets, list(model.parameters()))

                val_loss += loss.item()
                val_batches += 1

        val_loss /= val_batches
        val_loss_list.append(val_loss)

        if return_components:
            val_mu = sum_mu / val_batches
            val_sig = sum_sig / val_batches
            val_gmu = sum_gmu / val_batches
            val_gsig = sum_gsig / val_batches

            mu_list_val.append(val_mu)
            grad_mu_list_val.append(val_gmu)
            sigma_list_val.append(val_sig)
            grad_sigma_list_val.append(val_gsig)
            w_mu_list.append(w_mu)
            w_sig_list.append(w_sig)

        scheduler.step()

        print(f"Epoch {epoch + 1}/{n_epochs}:")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        if return_components:
            print(
                f"Train Loss={train_loss:.3e} (mu={train_mu:.3e}, sig={train_sig:.3e}, "
                f"gmu={train_gmu:.3e}, gsig={train_gsig:.3e}) | "
                f"Val L={val_loss:.3e} (mu={val_mu:.3e}, sig={val_sig:.3e}, "
                f"gmu={val_gmu:.3e}, gsig={val_gsig:.3e}) | "
                f"Task Weights w=[{w_mu:.2f},{w_sig:.2f}]"
            )

    elapsed_time = time.time() - start_time
    print("Computational time = ", elapsed_time)

    # -------------------------
    # Save outputs
    # -------------------------
    model_path = os.path.join(model_dir, "residual_unet_pretrained.pth")
    
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss_list,
        "val_loss": val_loss_list,
    }
    torch.save(state, model_path)
    
    if return_components:
        history_path = os.path.join(outputs_dir, "history_loss.mat")
        mdict = {
            "mu_list_train": mu_list_train,
            "mu_list_val": mu_list_val,
            "grad_mu_list_train": grad_mu_list_train,
            "grad_mu_list_val": grad_mu_list_val,
            "sigma_list_train": sigma_list_train,
            "sigma_list_val": sigma_list_val,
            "grad_sigma_list_train": grad_sigma_list_train,
            "grad_sigma_list_val": grad_sigma_list_val,
            "w_mu_list": w_mu_list,
            "w_sig_list": w_sig_list,
        }
        savemat(history_path, mdict)


if __name__ == "__main__":
    main()