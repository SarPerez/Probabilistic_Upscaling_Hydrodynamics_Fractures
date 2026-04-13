# Dataset notes

This directory includes some examples of aperture input files for lightweight inference and testing.

The full training/validation dataset is archived separately on Zenodo:

**Perez, S. (2026) ‘Probabilistic Upscaling of Hydrodynamics in Geological Fractures Under Uncertainty — Data and Pretrained Model’. Zenodo. doi:10.5281/zenodo.19554304.**

## Included in this GitHub repository

This repository keeps small example full-fracture aperture maps in `dataset/` so that users can:

- run the probabilistic upscaling workflow directly on example data,
- understand the expected input format,
- and adapt the code to new aperture fields.

Example files may include:

- `Apertures_NaturalFrac_1.npy`
- `Apertures_NaturalFrac_2.npy`

These example aperture maps are used as direct inputs for full-fracture inference with `Probabilistic_Upscaling.py`.

## Archived separately

The full processed patch dataset used for training and validation is **not stored directly in the GitHub repository** because of file size.

Archived file:

- `Dataset_apertures_to_perms.npy`

Download link:

- https://doi.org/10.5281/zenodo.19554304

After downloading the archive, place the file in this directory so that the expected path is:

```text
dataset/Dataset_apertures_to_perms.npy
