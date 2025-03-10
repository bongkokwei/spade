# SPADE: Spatial-Mode Demultiplexing for Two-Dimensional Resolution of Optical Point Sources

This repository contains implementations of the Spatial-Mode Demultiplexing (SPADE) technique for quantum-limited resolution of optical point sources, as described in the paper "Quantum limit for two-dimensional resolution of two incoherent optical point sources" by Ang, Nair, and Tsang (2017).

## Overview

The SPADE technique enables surpassing the conventional Rayleigh resolution limit for resolving closely spaced point sources. This project provides simulations of both 1D and 2D SPADE implementations, allowing estimation of the separation between two incoherent point sources with precision approaching the quantum Cramér-Rao bound.

## Files

- `spade_1d.py`: Implementation of the 1D SPADE technique for estimating separation between two point sources along a single dimension.
- `spade_2d.py`: Extension of SPADE to two dimensions, enabling complete transverse-plane localisation of two incoherent point sources.
- `requirements.txt`: Python dependencies needed to run the simulations.

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

The primary dependencies are:
- numpy
- scipy
- matplotlib
- lpmodes
- tqdm

## Features

### 1D SPADE (`spade_1d.py`)

- Implements Hermite-Gaussian spatial mode functions
- Provides numerical and analytical methods for calculating overlap factors
- Implements Maximum Likelihood Estimator (MLE) for estimating source separation
- Simulates photon detection in various spatial modes
- Compares SPADE performance to the direct imaging method and the quantum Cramér-Rao bound

### 2D SPADE (`spade_2d.py`)

- Extends SPADE to two dimensions for complete transverse-plane localisation
- Implements 2D Hermite-Gaussian spatial mode functions
- Provides visualisation of 2D point-spread functions
- Simulates 2D separation estimation for arbitrary separation vectors
- Evaluates performance across the entire 2D parameter space

## Usage

### Running 1D SPADE simulation

```python
python spade_1d.py
```

This will run the 1D SPADE simulation with default parameters and generate plots comparing the estimation performance to the quantum Cramér-Rao bound.

### Running 2D SPADE simulation

```python
python spade_2d.py
```

This will run the 2D SPADE simulation and generate heatmaps showing the estimation performance across various source separations.

## Key Functions

### 1D SPADE

- `hermite_gaussian_mode`: Generates Hermite-Gaussian spatial mode functions
- `numerical_overlap_factor`: Calculates the overlap between point-spread functions and Hermite-Gaussian modes
- `analytical_overlap_factor`: Provides analytical formula for the overlap factor
- `estimate_separation_mle`: Implements the Maximum Likelihood Estimator
- `simulate_photon_counts`: Simulates photon detection in different spatial modes
- `run_simulation`: Performs the complete SPADE simulation

### 2D SPADE

- `hermite_gaussian_mode_2d`: Generates 2D Hermite-Gaussian spatial mode functions
- `calculate_double_gaussian_psf`: Creates a double Gaussian PSF for two point sources
- `plot_double_gaussian`: Visualises the 2D PSF
- `numerical_overlap_factor_2d`: Calculates overlap factors for 2D modes
- `estimate_separation_2d_mle`: Implements 2D Maximum Likelihood Estimator
- `run_simulation`: Performs the complete 2D SPADE simulation
- `plot_simulation_results_2d`: Visualises 2D simulation results

## Theoretical Background

The simulations are based on the quantum Fisher information theory described in the paper by Ang, Nair, and Tsang (2017). The SPADE technique achieves resolution beyond the conventional Rayleigh limit by projecting the optical field onto Hermite-Gaussian modes rather than measuring position directly.

The quantum Cramér-Rao bound for estimating the separation between two incoherent point sources is independent of the value of separation, enabling super-resolution even for sub-Rayleigh distances.

## References

Ang, S. Z., Nair, R., & Tsang, M. (2017). Quantum limit for two-dimensional resolution of two incoherent optical point sources. Physical Review A, 95(6), 063847.