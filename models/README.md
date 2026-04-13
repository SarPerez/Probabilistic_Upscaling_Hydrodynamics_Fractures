# Pretrained model

The pretrained Residual U-Net checkpoint is archived separately on Zenodo:

**Perez, S. (2026) ‘Probabilistic Upscaling of Hydrodynamics in Geological Fractures Under Uncertainty — Data and Pretrained Model’. Zenodo. doi:10.5281/zenodo.19554304.**

Archived file:

- `residual_unet_pretrained.pth`

Download link:

- https://doi.org/10.5281/zenodo.19554304

This pretrained checkpoint is provided for direct inference with `Probabilistic_Upscaling.py`. It allows users to apply the probabilistic upscaling workflow directly to the example fracture aperture fields included in the repository, or to new compatible aperture inputs, without retraining the Residual U-Net from scratch.

After downloading the archive, place the file in this directory so that the expected path is:

```text
models/residual_unet_pretrained.pth
