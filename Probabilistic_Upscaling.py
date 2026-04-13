# -*- coding: utf-8 -*-
"""
Inference and post-processing script for probabilistic upscaling on full-fracture
mechanical aperture fields using a trained Residual U-Net.

The script loads a trained Residual U-Net model previously fitted on paired patches of mechanical 
aperture and log-normal permeability parameters mu(x,y) and sigma(x,y) (see Res_UNet_training.py).

Given a full-fracture mechanical aperture field a_m(x,y), it
applies the trained network to predict the spatially distributed log-normal
parameters mu(x,y) and sigma(x,y), from which probabilistic permeability
descriptors are reconstructed over the full domain.

The predicted outputs include:
    - the log-normal permeability parameters mu(x,y) and sigma(x,y),
    - the expected permeability E[K(x,y)],
    - the mode of K(x,y),
    - lower and upper quantile bounds for a prescribed confidence interval.

The script also provides utilities to:
    - visualise representative training/validation patches,
    - compare target and predicted patch-wise outputs,
    - evaluate distributional accuracy,
    - plot full-fracture probabilistic permeability maps and uncertainty maps,
    - save the predicted probabilistic descriptors for further upscaling.

Related paper:
    Probabilistic Upscaling of Hydrodynamics in Geological Fractures Under Uncertainty
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import norm
from scipy.io import savemat

from Res_UNet_training import (
    set_all_seeds,
    UNetResidual,
    load_patch_data,
    make_train_val_loaders,
)

# =============================================================================
# Helpers
# =============================================================================
def load_fracture_data(file_path):
    """
    Load the aperture field of the full fracture.
    """
    frac = np.load(file_path, allow_pickle=True)
    print(f"Fracture domain shape: {frac.shape}")
    return frac

def compute_extent(frac):
    """
    Compute plotting extent in millimeters for the full fracture given the
    physical voxel size.
    """
    pixel_size_m = 2.75e-6  # voxel size 2.75 micrometers

    Npy, Npx = frac.shape
    x = np.arange(Npx) * pixel_size_m 
    y = np.arange(Npy) * pixel_size_m 

    extent = [x[0] / 1e-3, x[-1] / 1e-3, y[-1] / 1e-3, y[0] / 1e-3]
    return extent

def load_trained_model(file_name, device="cuda"):
    """
    Load trained Residual U-Net checkpoint using the same architecture
    settings as during training.
    """
    checkpoint = torch.load(file_name, map_location=device, weights_only=False)

    model = UNetResidual(in_channels=1, out_channels=2, base_channels=32, dropout=True, dropout_rate=0.15).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def plot_fracture_aperture(frac, save_dir):
    """
    Plot full-fracture aperture field
    """
    extent = compute_extent(frac)

    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)

    im = ax.imshow(frac, cmap="turbo", extent=extent)
    ax.set_title("Thickness-weighted mechanical aperture", fontsize=14)
    ax.set_ylabel(r"y ($mm$)", fontsize=12)
    ax.set_xlabel(r"x ($mm$)", fontsize=12)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"Mechanical aperture ($\mu m$)", fontsize=11)
    
    file_path = os.path.join(save_dir, "Mechanical_aperture.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    

def plot_datasets(model, train_loader, val_loader, save_dir, device="cuda", index_dict=None):
    """
    Plot representative examples of paired input-target training patches
    
    index_dict: Dictionary of representative patch indices to plot, e.g.
                {"Train": [150, 300], "Val": [35, 75]}.
    If None, default representative sample is used.
    """
    dataset_map = {
        "Train": train_loader.dataset,
        "Val": val_loader.dataset
    }

    if index_dict is None:
        index_dict = {"Train": [300]}

    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for split_name, indices in index_dict.items():
            dataset = dataset_map.get(split_name)
            if dataset is None:
                print(f"Unknown split: {split_name}")
                continue

            for idx in indices:
                input_tensor, target = dataset[idx]

                a_m = input_tensor[0:1].squeeze().detach().cpu().numpy()
                mu = target[0].detach().cpu().numpy()
                sigma = target[1].detach().cpu().numpy()

                fig = plt.figure(figsize=(12, 6))
                plt.suptitle(f"{split_name} Sample Index: {idx}")

                plt.subplot(1, 3, 1)
                plt.imshow(a_m, cmap="Spectral_r")
                plt.title(r"$a_m(x,y)$", fontsize=14)
                plt.colorbar(orientation="horizontal")

                plt.subplot(1, 3, 2)
                plt.imshow(mu, cmap="Spectral_r")
                plt.title(r"Target $\mu(x,y)$", fontsize=14)
                plt.colorbar(orientation="horizontal")

                plt.subplot(1, 3, 3)
                plt.imshow(sigma, cmap="Spectral_r")
                plt.title(r"Target $\sigma(x,y)$", fontsize=14)
                plt.colorbar(orientation="horizontal")

                plt.tight_layout(rect=[0, 0, 1, 0.95])
                file_path = os.path.join(save_dir, f"Data_{split_name}_Index_{idx}.png")
                plt.savefig(file_path, dpi=300, bbox_inches="tight")
                plt.close(fig)

# =============================================================================
# Loss convergence 
# =============================================================================
def plot_loss_convergence(checkpoint, save_dir):
    """
    Plot loss convergence for training and validation 
    """
    train_loss = np.asarray(checkpoint["train_loss"], dtype=float)
    val_loss = np.asarray(checkpoint["val_loss"], dtype=float)

    epochs = np.arange(1, len(train_loss) + 1)

    best_ep = int(np.nanargmin(val_loss)) + 1
    best_val = float(np.nanmin(val_loss))

    # Normalized (relative) loss
    train_n = train_loss / train_loss[0]
    val_n = val_loss / val_loss[0]

    # Generalization gap
    gap = np.abs(val_loss - train_loss)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    # Absolute losses
    ax = axes[0]
    ax.semilogy(epochs, train_loss, color="darkorange", label=r"$\mathcal{L}_{\mathrm{train}}$")
    ax.semilogy(epochs, val_loss, label=r"$\mathcal{L}_{\mathrm{val}}$")
    ax.axvline(best_ep, linestyle="--", linewidth=1, color="black", alpha=0.8)
    ax.scatter([best_ep], [best_val], zorder=3, label=r"Best $\mathcal{L}_{\mathrm{val}} = $" + f"{best_val:.2e}")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(r"Total Loss: $\mathcal{L}$", fontsize=11)
    ax.set_title("Training and validation loss", fontsize=11)
    ax.legend(loc="upper center", fontsize=11)

    # Normalised losses (relative improvement)
    ax = axes[1]
    ax.semilogy(epochs, train_n, linewidth=2, color="darkorange", label=r"$\mathcal{L}_{\mathrm{train}}$")
    ax.semilogy(epochs, val_n, linewidth=2, label=r"$\mathcal{L}_{\mathrm{val}}$")
    ax.axvline(best_ep, linestyle="--", linewidth=1, color="black", alpha=0.8)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(r"Normalised Loss: $\mathcal{L}/\mathcal{L}_0$", fontsize=11)
    ax.set_title("Relative improvement")
    ax.legend(loc="upper center", fontsize=11)

    # Generalization gap 
    ax = axes[2]
    ax.semilogy(epochs, gap, linewidth=2)
    ax.axhline(0, linestyle="--", linewidth=1, color="black", alpha=0.8)
    ax.axvline(best_ep, linestyle="--", linewidth=1, color="black", alpha=0.8)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(r"Loss Difference: $|\mathcal{L}_{\mathrm{val}} - \mathcal{L}_{\mathrm{train}}|$", fontsize=11)
    ax.set_title("Generalisation gap")

    file_path = os.path.join(save_dir, "Loss_convergence.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    

# =============================================================================
# Distributional Accuracy 
# =============================================================================
def kl_lognormal_map(mu_p, sigma_p, mu_q, sigma_q, eps=1e-3):
    """
    Pixelwise KL divergence:
    D_KL( LogNormal(mu_p, sigma_p^2) || LogNormal(mu_q, sigma_q^2) )

    All inputs can be scalars or NumPy arrays of same shape.
    mu, sigma are log-space parameters (sigma is std, not variance).
    """
    sigma_p = np.maximum(sigma_p, eps)
    sigma_q = np.maximum(sigma_q, eps)

    kl = (np.log(sigma_q / sigma_p)
          + (sigma_p**2 + (mu_p - mu_q)**2) / (2.0 * sigma_q**2)
          - 0.5)
    return kl


def distributional_accuracy(model, train_loader, val_loader, save_dir, device="cuda", index_dict=None):
    """
    Evaluate patch-scale distributional accuracy using the symmetric pixelwise
    KL divergence between target and predicted log-normal distributions.
    Each patch is summarised by the median and 95th percentile of the
    divergence over pixels.  
    """
    dataset_map = {
        "Train": train_loader.dataset,
        "Val": val_loader.dataset
    }

    if index_dict is None:
        index_dict = {
            "Train": np.arange(len(train_loader.dataset)),
            "Val": np.arange(len(val_loader.dataset))
        }

    model.eval()

    x_vals = []
    split_labels = []
    median_vals = []
    p95_vals = []

    current_x = 0

    with torch.no_grad():
        # Evaluate train and validation patches separately
        for split_name in ["Train", "Val"]:
            indices = index_dict.get(split_name, [])
            dataset = dataset_map.get(split_name)

            if dataset is None:
                print(f"Unknown split: {split_name}")
                continue

            for idx in indices:
                input_tensor, target = dataset[idx]

                a_m = input_tensor[0:1].unsqueeze(0).to(device)

                mu = target[0].detach().cpu().numpy()
                sigma = target[1].detach().cpu().numpy()

                output = model(a_m)
                mu_pred = output[0, 0].detach().cpu().numpy()
                sigma_pred = output[0, 1].detach().cpu().numpy()
                
                # Symmetric KL divergence between target and predicted log-normal posteriors
                kl_map_sym = 0.5 * (
                    kl_lognormal_map(mu, sigma, mu_pred, sigma_pred) +
                    kl_lognormal_map(mu_pred, sigma_pred, mu, sigma)
                )
                
                # Exclude pixels where the target posterior collapses
                mask = sigma > 1e-3
                kl_masked = kl_map_sym[mask]

                # Summarise each patch-level divergence map by its median and 95th percentile
                if kl_masked.size == 0:
                    median_kl = np.nan
                    p95_kl = np.nan
                else:
                    median_kl = np.median(kl_masked)
                    p95_kl = np.quantile(kl_masked, 0.95)

                x_vals.append(current_x)
                split_labels.append(split_name)
                median_vals.append(median_kl)
                p95_vals.append(p95_kl)

                current_x += 1

    x_vals = np.array(x_vals)
    split_labels = np.array(split_labels)
    median_vals = np.array(median_vals)
    p95_vals = np.array(p95_vals)

    train_mask = split_labels == "Train"
    val_mask = split_labels == "Val"
    split_idx = np.sum(train_mask)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)

    axes[0].scatter(x_vals[train_mask], median_vals[train_mask], s=26, alpha=0.75, label="Training")
    axes[0].scatter(x_vals[val_mask], median_vals[val_mask], s=26, alpha=0.75, label="Validation")
    axes[0].axvline(split_idx - 0.5, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Patch index", fontsize=11)
    axes[0].set_ylabel("Median of pixelwise KL divergence", fontsize=11)

    ymin0, ymax0 = axes[0].get_ylim()
    axes[0].set_ylim(ymin0 * 0.8, ymax0 * 1.1)
    axes[0].legend(fontsize=10, loc="lower right")

    axes[1].scatter(x_vals[train_mask], p95_vals[train_mask], s=26, alpha=0.75, label="Training")
    axes[1].scatter(x_vals[val_mask], p95_vals[val_mask], s=26, alpha=0.75, label="Validation")
    axes[1].axvline(split_idx - 0.5, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Patch index", fontsize=11)
    axes[1].set_ylabel("95th percentile of pixelwise KL divergence", fontsize=11)

    ymin1, ymax1 = axes[1].get_ylim()
    axes[1].set_ylim(ymin1 * 0.8, ymax1 * 1.1)
    axes[1].legend(fontsize=10, loc="lower right")

    fig.tight_layout()
    
    file_path = os.path.join(save_dir, "Distribution_comparisons.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    
# =============================================================================
# Pixelwise comparison between target and predicted outputs 
# =============================================================================
def pixelwise_comparison(model, train_loader, val_loader, save_dir, device='cuda', index_dict=None):
    """
    Compare target and predicted log-normal parameter fields mu(x,y) and
    sigma(x,y) on representative training and validation patches, together
    with relative error maps.
   
    index_dict: Dictionary of representative patch indices to plot, e.g.
                {"Train": [150, 300], "Val": [35, 75]}.
    If None, default representative samples are used.
    """
        
    dataset_map = {
        "Train": train_loader.dataset,
        "Val": val_loader.dataset
    }

    if index_dict is None:
        index_dict = {"Train": [150, 300], "Val": [35, 75]}

    os.makedirs(save_dir, exist_ok=True)

    model.eval()

    with torch.no_grad():
        for split_name, indices in index_dict.items():
            dataset = dataset_map.get(split_name)
            if dataset is None:
                print(f"Unknown split: {split_name}")
                continue

            for idx in indices:
                input_tensor, target = dataset[idx]  # aperture input and target log-normal parameter fields

                a_m = input_tensor[0:1].unsqueeze(0).to(device)  # Shape: [1, 1, H, W]
                
                mu = target[0].detach().cpu().numpy()
                sigma = target[1].detach().cpu().numpy()
    
                output = model(a_m)
                mu_pred = output[0, 0].detach().cpu().numpy()
                sigma_pred = output[0, 1].detach().cpu().numpy()

                mu_vmin, mu_vmax = np.min(mu), np.max(mu)
                sigma_vmin, sigma_vmax = np.min(sigma), np.max(sigma)
                
                mu_err = np.abs(mu_pred - mu) / (np.max(np.abs(mu)))
                sigma_err = np.abs(sigma_pred - sigma) / (np.max(np.abs(sigma)))
                
                fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(6,10), constrained_layout=True)
                
                def _imshow_with_cbar(ax, img, title, cmap, vmin=None, vmax=None):
                    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
                    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.7, pad=0.08)
                    ax.set_title(title)
                    ax.axis('off')
                    return im, cbar
                
                # Left column: target, prediction, and relative error for mu(x,y)
                _imshow_with_cbar(axes[0, 0], mu,       r"Target $\mu(x,y)$",    cmap="Spectral_r", vmin=mu_vmin, vmax=mu_vmax)
                _imshow_with_cbar(axes[1, 0], mu_pred,  r"Predicted $\mu(x,y)$", cmap="Spectral_r", vmin=mu_vmin, vmax=mu_vmax)
                _imshow_with_cbar(axes[2, 0], mu_err,   r"Relative Error $\mu(x,y)$", cmap="coolwarm", vmax = 0.9*np.max(mu_err))
                
                # Right column: target, prediction, and relative error for sigma(x,y)
                _imshow_with_cbar(axes[0, 1], sigma,      r"Target $\sigma(x,y)$",    cmap="Spectral_r", vmin=sigma_vmin, vmax=sigma_vmax)
                _imshow_with_cbar(axes[1, 1], sigma_pred, r"Predicted $\sigma(x,y)$", cmap="Spectral_r", vmin=sigma_vmin, vmax=sigma_vmax)
                _imshow_with_cbar(axes[2, 1], sigma_err,  r"Relative Error $\sigma(x,y)$", cmap="coolwarm", vmax = 0.9*np.max(sigma_err))
                
            
                file_path = os.path.join(save_dir, f"Outputs_{split_name}_{idx}.png")
                plt.savefig(file_path, dpi=300, bbox_inches="tight")
                plt.close(fig)

# =============================================================================
# Full-fracture inference with the trained Residual U-Net
# =============================================================================

def compute_log_normal_bounds(output, CI=0.95): 
    """
    Reconstruct probabilistic permeability descriptors from the predicted
    log-normal parameters.

    Given the U-Net outputs mu(x,y) and sigma(x,y), assume that the local
    permeability follows K(x,y) ~ LogNormal(mu(x,y), sigma(x,y)^2),
    and compute the corresponding lower quantile, upper quantile, mean, and mode fields.
    """
    mu, sigma = output.squeeze(0).cpu().numpy()
    mean = np.exp(mu+0.5*sigma**2)  

    # Compute quantile
    alpha = (1 - CI) / 2
    z_lower = norm.ppf(alpha)
    z_upper = norm.ppf(1 - alpha)

    lower = np.exp(mu + z_lower * sigma)
    upper = np.exp(mu + z_upper * sigma)
    mode = np.exp(mu-sigma**2)

    return lower, upper, mean, mode 


def predict_full_fracture(model, frac, device="cuda", CI=0.95):
    """
    Apply the trained Residual U-Net to a full mechanical-aperture field and
    reconstruct probabilistic permeability descriptors over the full fracture domain.

    The input frac is a full-fracture mechanical aperture map a_m(x,y).
    The network predicts the log-normal permeability parameters mu(x,y) and
    sigma(x,y), from which the lower quantile, upper quantile, mean, and mode
    of K(x,y) are derived for the prescribed confidence interval.

    The input is padded so that the spatial dimensions are divisible by 8,
    then cropped back to the original fracture size after inference.
    """
    frac_tensor = torch.tensor(frac.copy()).unsqueeze(0).unsqueeze(0).float().to(device)  # [1, 1, H, W]

    # Pad to divisible by 8
    pad_h = (8 - frac.shape[0] % 8) % 8
    pad_w = (8 - frac.shape[1] % 8) % 8

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded = torch.nn.functional.pad(frac_tensor, (pad_left, pad_right, pad_top, pad_bottom))

    model.eval()
    with torch.no_grad():
        output = model(padded)

    # Crop back to original size
    output = output[:, :, pad_top:pad_top + frac.shape[0], pad_left:pad_left + frac.shape[1]]

    lower, upper, mean, mode = compute_log_normal_bounds(output, CI=CI)
    mu, sigma = output.squeeze(0).cpu().numpy()

    return {
        "mu": mu,
        "sigma": sigma,
        "lower": lower,
        "upper": upper,
        "mean": mean,
        "mode": mode,
    }


def save_probabilistic_outputs(results, save_dir, prefix="95CI", frac_name ="NaturalFrac_1"):
    """
    Save predicted probabilistic outputs to .mat files.
    """
    os.makedirs(save_dir, exist_ok=True)

    mdict_stats = {
        "lower": results["lower"],
        "upper": results["upper"],
        "mean": results["mean"],
        "mode": results["mode"],
    }
    savemat(os.path.join(save_dir, f"{prefix}_descriptors_{frac_name}.mat"), mdict_stats)

    mdict_params = {
        "mu": results["mu"],
        "sigma": results["sigma"],
    }
    savemat(os.path.join(save_dir, f"{prefix}_mu_sigma_params_{frac_name}.mat"), mdict_params)


def plot_local_cubic_law(frac, save_dir, vmax=None):
    """
    Plot the deterministic local Cubic-Law permeability field
    K_CL^{a_m}(x,y) = a_m(x,y)^2 / 12 derived from the mechanical aperture.
    """
    extent = compute_extent(frac)
    
    fig, ax = plt.subplots(figsize=(5, 6), constrained_layout=True)
    im = ax.imshow(frac**2 / 12, cmap="turbo", vmax=vmax, extent=extent)
    ax.set_title(r"Local Cubic Law: $K_{CL}^{a_m}(x,y)$", fontsize=12)
    ax.set_xlabel(r"x ($mm$)", fontsize=11)
    ax.set_ylabel(r"y ($mm$)", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(r"Permeability ($\mu m^2$)", fontsize=11)

    file_path = os.path.join(save_dir, "Local_cubic_law.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    


def plot_probabilistic_maps(frac, results, save_dir,
                            vmax_lower=None, vmax_mode=None, vmax_mean=None, vmax_upper=None):
    """
    Plot probabilistic permeability descriptors reconstructed from the predicted
    log-normal parameters: lower quantile, mode, mean, and upper quantile.
    """
    extent = compute_extent(frac)
    
    lower = results["lower"]
    mode = results["mode"]
    mean = results["mean"]
    upper = results["upper"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 11), constrained_layout=False)

    # Lower bound
    im0 = axes[0, 0].imshow(lower, vmax=vmax_lower, cmap="turbo", extent=extent)
    axes[0, 0].set_title(r"$Q_{0.025}[K(x,y)]$", fontsize=14)
    axes[0, 0].set_xlabel(r"x ($mm$)", fontsize=12)
    axes[0, 0].set_ylabel(r"y ($mm$)", fontsize=12)
    cbar0 = fig.colorbar(im0, ax=axes[0, 0], shrink=0.9)
    cbar0.set_label(r"Permeability ($\mu m^2$)", fontsize=11)

    # Mode 
    im1 = axes[0, 1].imshow(mode, vmax=vmax_mode, cmap="turbo", extent=extent)
    axes[0, 1].set_title(r"Mode$[K(x,y)]$", fontsize=14)
    axes[0, 1].set_xlabel(r"x ($mm$)", fontsize=12)
    cbar1 = fig.colorbar(im1, ax=axes[0, 1], shrink=0.9)
    cbar1.set_label(r"Permeability ($\mu m^2$)", fontsize=11)

    # Mean 
    im2 = axes[1, 0].imshow(mean, vmax=vmax_mean, cmap="turbo", extent=extent)
    axes[1, 0].set_title(r"$\mathbb{E}\,[K(x,y)]$", fontsize=14)
    axes[1, 0].set_xlabel(r"x ($mm$)", fontsize=12)
    axes[1, 0].set_ylabel(r"y ($mm$)", fontsize=12)
    cbar2 = fig.colorbar(im2, ax=axes[1, 0], shrink=0.9)
    cbar2.set_label(r"Permeability ($\mu m^2$)", fontsize=11)

    # Upper bound 
    im3 = axes[1, 1].imshow(upper, vmax=vmax_upper, cmap="turbo", extent=extent)
    axes[1, 1].set_title(r"$Q_{0.975}[K(x,y)]$", fontsize=14)
    axes[1, 1].set_xlabel(r"x ($mm$)", fontsize=12)
    cbar3 = fig.colorbar(im3, ax=axes[1, 1], shrink=0.9)
    cbar3.set_label(r"Permeability ($\mu m^2$)", fontsize=12)

    fig.tight_layout()
    
    file_path = os.path.join(save_dir, "Probabilistic_permeability_maps.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    
def plot_uncertainty_maps(frac, results, save_dir, vmax_width=None, vmax_relative=None):
    """
    Plot two local uncertainty descriptors:
    - the permeability uncertainty width Q_0.975 - Q_0.025,
    - the relative uncertainty ln(Q_0.975 / Q_0.025).
    
    The relative uncertainty is scale-independent and directly proportional to
    the predicted log-normal standard deviation sigma(x,y).
    """
    extent = compute_extent(frac)

    lower = results["lower"]
    upper = results["upper"]

    uncertainty_width = upper - lower
    relative_uncertainty = np.log(upper / lower)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)

    # Absolute uncertainty width 
    im0 = axes[0].imshow(uncertainty_width, cmap="turbo", vmax=vmax_width, extent=extent)
    axes[0].set_title(r"Permeability Uncertainty width", fontsize=13)
    axes[0].set_xlabel(r"x ($mm$)", fontsize=12)
    axes[0].set_ylabel(r"y ($mm$)", fontsize=12)
    cbar0 = fig.colorbar(im0, ax=axes[0], shrink=0.9)
    cbar0.set_label(r"$Q_{0.975} - Q_{0.025}$", fontsize=11)

    # Relative uncertainty
    im1 = axes[1].imshow(relative_uncertainty, cmap="turbo", vmax=vmax_relative, extent=extent)
    axes[1].set_title(r"Relative uncertainty", fontsize=13)
    axes[1].set_xlabel(r"x ($mm$)", fontsize=12)
    axes[1].set_ylabel(r"y ($mm$)", fontsize=12)
    cbar1 = fig.colorbar(im1, ax=axes[1], shrink=0.9)
    cbar1.set_label(r"$\ln(Q_{0.975}/Q_{0.025})$", fontsize=11)

    file_path = os.path.join(save_dir, "Uncertainty_maps.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



# =============================================================================
# Main
# =============================================================================
def main():
    set_all_seeds()

    fig_dir = "./figs"                                                       # Directory for generated figures
    outputs_dir = "./outputs"                                                # Directory for intermediate results
    model_dir = "./models"                                                   # Direction with pre-trained Residual U-Net model
    frac_name = "NaturalFrac_1"                                              # Full-fracture name
    frac_path = f"./dataset/Apertures_{frac_name}.npy"                       # Apertures data for the full fracture
    model_path = os.path.join(model_dir, "residual_unet_pretrained.pth")     # Residual-UNet model pre-trained on the patches
    
    training_validation = False
    if training_validation:
        patch_path = "./dataset/Dataset_apertures_to_perms.npy"                  # Apertures data on the patches (Validation only) 

    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"Is CUDA available?: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name())

    
    # =============================================================================
    # Load full-fracture mechanical aperture field and save the aperture to a .mat file
    # =============================================================================
    frac = load_fracture_data(frac_path)
    plot_fracture_aperture(frac, save_dir=fig_dir)
    savemat(os.path.join(outputs_dir, f"Apertures_{frac_name}.mat"), {"a_m":frac})
   
    # =============================================================================
    # Load trained Residual U-Net model
    # =============================================================================
    model, checkpoint = load_trained_model(model_path, device=device)

    
    # =============================================================================
    # Probabilistic Upscaling & Results Plotting
    # =============================================================================
    print("Probabilistic upscaling on full-fracture aperture field")
    results = predict_full_fracture(model, frac, device=device, CI=0.95)
    
    print("Plotting results")
    plot_local_cubic_law(
        frac,
        save_dir=fig_dir
    )

    plot_probabilistic_maps(
        frac,
        results,
        save_dir=fig_dir,
        vmax_lower=13,
        vmax_mode=27,
        vmax_mean=1550,
        vmax_upper=7500
    )
    
    plot_uncertainty_maps(
        frac,
        results,
        save_dir=fig_dir,
        vmax_width=7500,
        vmax_relative=None
    )
        
    print("Save outputs")
    save_probabilistic_outputs(
        results,
        save_dir=outputs_dir,
        frac_name=frac_name
    )

    if training_validation:
        # =============================================================================
        # Load aperture patches and posterior predictive permeability moments (128 x 128)
        # =============================================================================
        a_m, K_mean, K_std = load_patch_data(patch_path)
        print(f"Patches Dataset Shape (N, Nx, Ny): {a_m.shape}")
        
        # =============================================================================
        # Training / Validation split
        # =============================================================================
        full_dataset, train_loader, val_loader, train_idx, val_idx, train_size, val_size = make_train_val_loaders(
            a_m, K_mean, K_std, batch_size=32)
        print("Residual U-Net training split: Train size", train_size, "Validation size", val_size)
        
        # =============================================================================
        # Plot representative input-target training patches
        # =============================================================================
        plot_datasets(model, train_loader, val_loader, save_dir=fig_dir, device=device)

        # =============================================================================
        # Plot Loss function convergence 
        # =============================================================================
        plot_loss_convergence(checkpoint, save_dir=fig_dir)
        
        # =============================================================================
        # Pixelwise comparison between target and predicted outputs 
        # =============================================================================
        pixelwise_comparison(model, train_loader, val_loader, save_dir=fig_dir, device=device)

        # =============================================================================
        # Distributional Accuracy 
        # =============================================================================
        distributional_accuracy(model, train_loader, val_loader, save_dir=fig_dir, device=device)


if __name__ == "__main__":
    main()