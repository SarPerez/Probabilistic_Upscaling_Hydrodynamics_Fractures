# -*- coding: utf-8 -*-
"""
Post-processing and visualisation of MRST upscaling results for a single fracture.

The script:
- loads Darcy-upscaled permeability values computed with MRST,
- reconstructs a log-normal distribution of the upscaled permeability from
  the reduced descriptor fields (mode and mean),
- optionally compares this analytical distribution with the Monte Carlo
  propagation results,
- plots the corresponding probability distributions together with reference
  permeability values.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.io import loadmat

# =============================================================================
#  Load MRST upscaling results
# =============================================================================

fig_dir = "./figs"    
outputs_dir = "./outputs"                                       # Directory for intermediate results
                                                                # Directory for generated figures
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)

frac_name = "NaturalFrac_1"
data = loadmat(f"{outputs_dir}/mrst_upscaling_results_{frac_name}.mat")         # Load MRST upscaling results for analysis


Keff_MonteCarlo = data["Keff_MonteCarlo_m2"].squeeze()                      # Monte Carlo samples of Keff [m^2] - 1D array or empty if Monte Carlo is off 
Keff_cubic_law = float(data["Keff_cubic_law_m2"].squeeze())                 # Upscaled classical Cubic-Law permeability [m^2]
Keff_local_cubic_law = float(data["Keff_local_cubic_law_m2"].squeeze())     # Upscaled Darcy permeability from local Cubic Law [m^2]
Keff_lower = float(data["Keff_lower_m2"].squeeze())                         # Upscaled lower quantile [m^2]
Keff_mode  = float(data["Keff_mode_m2"].squeeze())                          # Upscaled mode [m^2]
Keff_mean  = float(data["Keff_mean_m2"].squeeze())                          # Upscaled mean [m^2]
Keff_upper = float(data["Keff_upper_m2"].squeeze())                         # Upscaled upper quantile [m^2]

# Reference Stokes-consistent permeabilities [m^2]
data_Stokes = {
    "NaturalFrac_1": 4.48112e-12,
    "SyntFrac_1": 5.18198e-12,
    "NaturalFrac_2": 6.20803e-11,
    "SyntFrac_2": 7.51971e-11
}

Keff_Stokes = data_Stokes[frac_name]

# =============================================================================
# KL divergence between two log-normal distributions
# =============================================================================

def kl_lognormal(mu_p, sigma_p, mu_q, sigma_q):
    """
    Compute the KL divergence
    D_KL( LogNormal(mu_p, sigma_p^2) || LogNormal(mu_q, sigma_q^2) ).
    """
    return (np.log(sigma_q / sigma_p)
            + (sigma_p**2 + (mu_p - mu_q)**2) / (2 * sigma_q**2)
            - 0.5)

# =============================================================================
# Analytical probability distribution from reduced descriptors
# =============================================================================

z = 1.96  # Gaussian quantile for an approximate 95% confidence interval

# Reconstruct a log-normal distribution from the upscaled mean and mode
sigma2 = (2 / 3) * (np.log(Keff_mean) - np.log(Keff_mode))
sigma = np.sqrt(sigma2)
mu = np.log(Keff_mode) + sigma**2

median = np.exp(mu)
L_CI = np.exp(mu - z * sigma)
U_CI = np.exp(mu + z * sigma)

print("Reduced descriptors statistics")
print(f"Fitted mu = {mu:.3f}, sigma = {sigma:.3f}")
print(f"Mode     = {Keff_mode:.3e}")
print(f"Median   = {median:.3e}")
print(f"Mean     = {Keff_mean:.3e}")
print(f"95% CI   = [{L_CI:.3e}, {U_CI:.3e}]\n")

# =============================================================================
# Monte Carlo validation
# =============================================================================

# Keep only valid positive samples for log-normal fitting
K_list = np.asarray(Keff_MonteCarlo, dtype=float).squeeze()
K_list = K_list[np.isfinite(K_list)]
K_list = K_list[K_list > 0]

has_mc = K_list.size > 0

if has_mc:
    # Fit a log-normal model to the Monte Carlo distribution of Keff
    logK = np.log(K_list)
    mu_MC = np.mean(logK)
    sigma_MC = np.std(logK, ddof=1) if K_list.size > 1 else 0.0

    mean_MC = np.exp(mu_MC + 0.5 * sigma_MC**2)
    mode_MC = np.exp(mu_MC - sigma_MC**2)
    median_MC = np.exp(mu_MC)
    L_CI_MC = np.exp(mu_MC - z * sigma_MC)
    U_CI_MC = np.exp(mu_MC + z * sigma_MC)

    print("Monte Carlo simulation statistics")
    print(f"Fitted mu = {mu_MC:.3f}, sigma = {sigma_MC:.3f}")
    print(f"Mode   = {mode_MC:.3e}")
    print(f"Median = {median_MC:.3e}")
    print(f"Mean  = {mean_MC:.3e}")
    print(f"95% CI = [{L_CI_MC:.3e}, {U_CI_MC:.3e}]\n")
else:
    print("No Monte Carlo samples available in Keff_MonteCarlo_m2.\n")


# =============================================================================
# Plot probability distributions at the upscaled level
# =============================================================================

COL_PDF     = "#0072B2"   # analytical log-normal PDF
COL_CI      = "#56B4E9"   # confidence interval shading
COL_NS      = "#009E73"   # Stokes reference / mode marker
COL_CL      = "#D55E00"   # Cubic-Law references
COL_MC_HIST = "#999999"   # Monte Carlo histogram
COL_MC_PDF  = "#000000"   # fitted Monte Carlo PDF

# Build the plotting range in permeability space
x_min = max(min(Keff_lower, L_CI) / 5, 1e-20)
x_max = max(Keff_upper, U_CI, Keff_cubic_law, Keff_local_cubic_law, Keff_Stokes) * 8

if has_mc:
    x_min = min(x_min, max(K_list.min() / 5, 1e-20))
    x_max = max(x_max, K_list.max() * 5)

x = np.logspace(np.log10(x_min), np.log10(x_max), 800)
pdf = (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))

fig, ax = plt.subplots(figsize=(8, 5))

# Analytical PDF reconstructed from reduced descriptors
ax.plot(x, pdf, color=COL_PDF, alpha=0.6, lw=2, label="Log-normal PDF")

mask = (x >= L_CI) & (x <= U_CI)
ax.fill_between(x[mask], pdf[mask], color=COL_CI, alpha=0.3, label="95% CI")

# Vertical markers for the main reduced descriptors
ax.axvline(Keff_mean, color="darkblue", linestyle="-", lw=2)
ax.axvline(Keff_mode, color=COL_NS, linestyle="-", lw=2)

# Reference permeability values
ax.axvline(Keff_Stokes, color=COL_NS, linestyle="--",
           label=rf"$K_{{NS}}$ = {Keff_Stokes*1e12:.2f} $\mu m^2$")
ax.axvline(Keff_local_cubic_law, color=COL_CL, linestyle="--",
           label=rf"$K_D^{{am}}$ = {Keff_local_cubic_law*1e12:.2f} $\mu m^2$")
ax.axvline(Keff_cubic_law, color=COL_CL, linestyle="-",
           label=rf"$K_{{CL}}$ = {Keff_cubic_law*1e12:.2f} $\mu m^2$")

# Annotate mean, mode, and upper-tail probability
y_ref = pdf[np.argmin(np.abs(x - Keff_mode))]

ax.annotate("Mode",
            xy=(Keff_mode, y_ref * 0.6),
            xytext=(Keff_mode * 1.5, y_ref * 0.4),
            arrowprops=dict(arrowstyle="->", linestyle="-", lw=1.5, color=COL_NS),
            fontsize=10, color=COL_NS, fontweight="bold")

ax.annotate("Mean",
            xy=(Keff_mean, y_ref * 0.6),
            xytext=(Keff_mean * 1.5, y_ref * 0.4),
            arrowprops=dict(arrowstyle="->", linestyle="-", lw=1.5, color="darkblue"),
            fontsize=10, color="darkblue", fontweight="bold")

ax.annotate(r"$<2.5\%$",
            xy=(U_CI, y_ref * 0.6),
            xytext=(U_CI * 2.5, y_ref * 0.4),
            fontsize=12, color=COL_CL, fontweight="bold")

# Highlight the upper tail beyond the 97.5th percentile
ymax = pdf.max()
ax.fill_between(x, 0, ymax * 1.1, where=(x >= U_CI),
                facecolor="none", hatch="//", edgecolor=COL_CL, alpha=0.25)

# Overlay Monte Carlo histogram and fitted PDF when available
if has_mc and K_list.size > 1 and sigma_MC > 0:
    bins = np.logspace(np.log10(max(K_list.min(), 1e-300)),
                       np.log10(K_list.max()),
                       50)

    ax.hist(K_list, bins=bins, density=True,
            alpha=0.35, edgecolor="none",
            color=COL_MC_HIST, label=r"Histogram $K_{eff}$ (MC)")

    pdf_MC = (1 / (x * sigma_MC * np.sqrt(2 * np.pi))) * np.exp(-(np.log(x) - mu_MC)**2 / (2 * sigma_MC**2))
    ax.plot(x, pdf_MC, color=COL_MC_PDF, lw=2, linestyle="--", alpha=0.35,
            label="Log-normal PDF (MC)")

    kl_PQ = kl_lognormal(mu, sigma, mu_MC, sigma_MC)
    kl_QP = kl_lognormal(mu_MC, sigma_MC, mu, sigma)
    print(f"KL divergence between the analytical and Monte Carlo log-normal distributions: "
          f"D_KL(P||P_MC) = {kl_PQ:.3e}, D_KL(P_MC||P) = {kl_QP:.3e}")

# Axis formatting
ax.set_xscale("log")
ax.set_xlabel(r"Upscaled permeability ($\mu m^2$) - Log-scale")
ax.set_ylabel("Density")
ax.set_yticks([])

ax.set_xlim(1e-12, 5e-9)
ax.set_ylim(0, ymax * 1.1)

# Display x-axis values in micrometer^2 while keeping SI units internally
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*1e12:.1f}"))

ax.legend(loc="upper right", framealpha=1.0)
plt.title("Uncertainty on Upscaled Permeability")

plt.tight_layout()

file_path = os.path.join(fig_dir, "Upscaled_permeability.png")
plt.savefig(file_path, dpi=300, bbox_inches="tight")
plt.close(fig)

