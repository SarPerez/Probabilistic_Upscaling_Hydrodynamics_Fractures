# Probabilistic Upscaling of Hydrodynamics in Geological Fractures Under Uncertainty

This repository implements a **probabilistic workflow for upscaling hydrodynamics in geological fractures**, propagating uncertainty from mechanical aperture measurements to local permeability statistics and finally to upscaled hydraulic responses.

It contains the code and example input files associated with the article:

*Probabilistic Upscaling of Hydrodynamics in Geological Fractures Under Uncertainty*, **S. Perez, F. Doster, H. Menke, A. ElSheikh, and A. Busch**

The workflow combines:
- a physics-based Bayesian correction of aperture–permeability model misspecification,
- a Residual U-Net surrogate for predicting probabilistic permeability descriptors from mechanical aperture fields,
- and Darcy-scale upscaling to propagate local uncertainty to effective fracture-scale hydraulic responses.

## Overview

Fluid flow in fractured geological media is strongly controlled by aperture heterogeneity, roughness, connectivity, and uncertainty in subsurface characterisation. Classical deterministic aperture–permeability mappings, such as Cubic-Law-based approaches, often fail to capture the hydraulic response of rough and topologically complex natural fractures.

This repository implements a probabilistic workflow in which:
- mechanical aperture fields are used as uncertain geometric inputs rather than hydraulic proxies,
- local permeability uncertainty is represented through spatially varying log-normal parameters,
- and fracture-scale effective permeability is estimated by propagating uncertainty through Darcy-flow upscaling.

The repository is designed to support both reproducibility of the published workflow and reuse of the pretrained model for direct application to new aperture maps. It includes lightweight example aperture inputs for direct testing. Large files are archived separately on Zenodo.

## Archived data and pretrained model

The large data and model assets associated with this repository are archived on Zenodo (see `dataset/README.md` and `models/README.md`):

**Perez, S. (2026) ‘Probabilistic Upscaling of Hydrodynamics in Geological Fractures Under Uncertainty — Data and Pretrained Model’. Zenodo. doi:10.5281/zenodo.19554304.**

Download:
- https://doi.org/10.5281/zenodo.19554304

The Zenodo archive contains:
- `dataset/Dataset_apertures_to_perms.npy`  
  Training/validation dataset used for retraining and validation.
- `models/residual_unet_pretrained.pth`  
  Pretrained Residual U-Net checkpoint used for direct inference.

After downloading and extracting the archive:
- place `Dataset_apertures_to_perms.npy` in `./dataset/`
- place `residual_unet_pretrained.pth` in `./models/`

## Main code modules / workflow overview

### 1. Train the surrogate model — `Res_UNet_training.py`

This script:
- loads mechanical aperture $a_m(x,y)$ patches and posterior predictive permeability moments from `Dataset_apertures_to_perms.npy`,
- converts the posterior moments into log-normal parameters $\mu(x,y)$ and $\sigma(x,y)$,
- trains a Residual U-Net to learn the mapping $a_m(x,y) \rightarrow \big(\mu(x,y), \ \sigma(x,y)\big)$,
- saves the trained model and training history.

### 2. Direct inference of probabilistic permeability maps — `Probabilistic_Upscaling.py`

This script:
- loads the pretrained Residual U-Net from `residual_unet_pretrained.pth`,
- loads a full-fracture mechanical aperture field,
- predicts the spatially distributed log-normal parameters $\mu(x,y)$ and $\sigma(x,y)$,
- reconstructs probabilistic permeability descriptors: lower quantile, mode, mean, upper quantile,
- saves figures and `.mat` outputs for Darcy upscaling.

### 3. Compute Darcy-upscaled permeability with MRST — `Darcy_Upscaling_MRST.m`

This MATLAB / MRST script:
- computes the Darcy-upscaled permeability of the local Cubic-Law field,
- computes the upscaled permeability of the predicted descriptor fields: lower quantile, mode, mean, upper quantile,
- optionally performs Monte Carlo propagation from the predicted $\mu(x,y)$ and $\sigma(x,y)$ fields.

### 4. Post-processing and probabilistic upscaling analysis — `Upscaled_Prob_Dist.py`

This script:
- loads the MRST upscaling results,
- reconstructs an analytical log-normal distribution of upscaled permeability from the reduced descriptors,
- optionally compares it with the Monte Carlo distribution,
- plots the final uncertainty distribution at the upscaled level together with reference permeability values.

## MATLAB / MRST requirements

The Darcy-scale upscaling stage is implemented in MATLAB using MRST.

To run `Darcy_Upscaling_MRST.m`, you need:
- MATLAB
- a working installation of MRST
- the relevant MRST modules required, namely the `incomp` module

Please make sure MRST is installed and added to the MATLAB path before running the upscaling script.

## Citation
