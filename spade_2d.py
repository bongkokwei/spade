import numpy as np
from scipy.special import hermite, factorial
from scipy.integrate import quad, dblquad
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from functools import partial
import os


def gaussian_psf_2d(x, y, x0, y0, sigma):
    """
    2D Gaussian point-spread function centered at (x0, y0).
        Parameters:
            x, y (float): Spatial coordinates
            x0, y0 (float): Center positions
            sigma (float): Width of the point-spread function
        Returns:
            float: Value of the 2D Gaussian PSF at position (x, y)
    """
    normalization = 1 / np.sqrt(2 * np.pi * sigma**2)
    return normalization * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (4 * sigma**2))


def hermite_gaussian_mode_2d(x, y, qx, qy, sigma):
    """
    2D Hermite-Gaussian spatial mode function.
        Parameters:
            x, y (float): Spatial coordinates
            qx, qy (int): Mode indices along x and y
            sigma (float): Width of the point-spread function
        Returns:
            float: Value of the 2D Hermite-Gaussian mode at position (x, y)
    """
    normalization_x = 1 / np.sqrt(2**qx * factorial(qx))
    normalization_y = 1 / np.sqrt(2**qy * factorial(qy))

    # Hermite polynomials
    H_qx = hermite(qx)
    H_qy = hermite(qy)

    # Normalized coordinates
    x_normalized = x / (np.sqrt(2) * sigma)
    y_normalized = y / (np.sqrt(2) * sigma)

    # Prefactor
    prefactor = 1 / np.sqrt(2 * np.pi * sigma**2)

    # Evaluate 2D Hermite-Gaussian mode
    return (
        prefactor
        * normalization_x
        * normalization_y
        * H_qx(x_normalized)
        * H_qy(y_normalized)
        * np.exp((-(x**2) - y**2) / (4 * sigma**2))
    )


def calculate_double_gaussian_psf(xx, yy, params):
    """
    Calculate a double Gaussian PSF with specified parameters.

    Parameters:
    -----------
    xx, yy : ndarray
        Meshgrid arrays of x and y coordinates.
    params : dict
        Dictionary containing the following parameters:
        - sigma : float
            Standard deviation of the Gaussian.
        - centroid_x, centroid_y : float
            Coordinates of the centroid.
        - sep_x, sep_y : float
            Separation in x and y directions.
        - amp_1, amp_2 : float
            Amplitudes of the two Gaussian components.

    Returns:
    --------
    ndarray : The combined 2D double Gaussian PSF
    """
    # Extract parameters with defaults
    sigma = params.get("sigma", 1.0)
    centroid_x = params.get("centroid_x", 0.0)
    centroid_y = params.get("centroid_y", 0.0)
    sep_x = params.get("sep_x", 3.0)
    sep_y = params.get("sep_y", 3.0)
    amp_1 = params.get("amp_1", 0.5)
    amp_2 = params.get("amp_2", 0.5)

    # Calculate center coordinates
    theta_1_x = centroid_x * sigma
    theta_1_y = centroid_y * sigma
    theta_2_x = sep_x * sigma
    theta_2_y = sep_y * sigma

    # Calculate positions of the two Gaussians
    x1_0 = theta_1_x + theta_2_x / 2
    y1_0 = theta_1_y + theta_2_y / 2
    x2_0 = theta_1_x - theta_2_x / 2
    y2_0 = theta_1_y - theta_2_y / 2

    # Calculate combined PSF
    psf_2d = amp_1 * gaussian_psf_2d(
        xx, yy, x1_0, y1_0, sigma
    ) + amp_2 * gaussian_psf_2d(xx, yy, x2_0, y2_0, sigma)

    return psf_2d


def plot_double_gaussian(
    sigma=1.0,
    separation=15,
    num_points=100,
    centroid_x=0,
    centroid_y=0,
    sep_x=4,
    sep_y=2,
    figsize=(6, 6),
    amp_1=0.5,
    amp_2=0.5,
    ax=None,
):
    """
    Plots a double Gaussian point spread function.

    Parameters:
    -----------
    sigma : float
        Standard deviation of the Gaussian.
    separation : float
        Separation factor for the x and y ranges.
    num_points : int
        Number of points along each dimension.
    centroid_x, centroid_y : float
        Coordinates of the centroid.
    sep_x, sep_y : float
        Separation in x and y directions.
    figsize : tuple
        Figure size (width, height) in inches.
    ax : matplotlib.axes or array of axes, optional
        Pre-existing axes for the plot. If None, creates new figure and axes.

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """

    # Set up grid
    x = np.linspace(-separation * sigma, separation * sigma, num_points)
    y = np.linspace(-separation * sigma, separation * sigma, num_points)

    # Create meshgrid
    xx, yy = np.meshgrid(x, y)

    params = {
        "sigma": sigma,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "sep_x": sep_x,
        "sep_y": sep_y,
        "amp_1": amp_1,
        "amp_2": amp_2,
    }
    psf_2d = calculate_double_gaussian_psf(xx, yy, params)

    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # First subplot - double Gaussian
    ax.contourf(
        xx,
        yy,
        psf_2d,
        levels=100,
    )
    ax.set_xlabel(r"x-coord ($\sigma$)")
    ax.set_ylabel(r"y-coord ($\sigma$)")

    return fig, ax


def exact_overlap_factor_2d(qx, qy, params):
    # Extract parameters
    sigma = params.get("sigma", 1.0)
    sep_x = params.get("sep_x", 3.0)
    sep_y = params.get("sep_y", 3.0)

    Qx = sep_x**2 / (16 * sigma**2)
    Qy = sep_y**2 / (16 * sigma**2)

    P1_q = np.exp(-Qx - Qy) * ((Qx**qx) * (Qy**qy)) / (factorial(qx) * factorial(qy))
    return P1_q


def numerical_overlap_factor_2d(qx, qy, params, integration_limit=10):
    """
    Numerically calculate the overlap factor (probability of detecting a photon in the (qx,qy) mode)
    for 2D Hermite-Gaussian modes.

    Parameters:
    -----------
    qx, qy (int): Mode indices for x and y directions
    params (dict): Dictionary containing PSF parameters:
        - sigma: Width of the point-spread function
        - centroid_x, centroid_y: Coordinates of the centroid
        - sep_x, sep_y: Separation in x and y directions
        - amp_1, amp_2: Amplitudes of the two Gaussian components
    integration_limit (float): Integration limit (should be large enough)
    num_points (int): Number of points in the numerical grid

    Returns:
    --------
    float: Numerically computed probability of detecting a photon in the (qx,qy) mode
    """

    # Extract parameters
    sigma = params.get("sigma", 1.0)
    centroid_x = params.get("centroid_x", 0.0)
    centroid_y = params.get("centroid_y", 0.0)
    sep_x = params.get("sep_x", 3.0)
    sep_y = params.get("sep_y", 3.0)
    amp_1 = params.get("amp_1", 0.5)
    amp_2 = params.get("amp_2", 0.5)

    # Calculate center coordinates
    theta_1_x = centroid_x * sigma
    theta_1_y = centroid_y * sigma
    theta_2_x = sep_x * sigma
    theta_2_y = sep_y * sigma

    # Calculate positions of the two Gaussians
    x1_0 = theta_1_x + theta_2_x / 2
    y1_0 = theta_1_y + theta_2_y / 2
    x2_0 = theta_1_x - theta_2_x / 2
    y2_0 = theta_1_y - theta_2_y / 2

    # Define the integrand for the first source
    def integrand1(y, x):
        psi1 = gaussian_psf_2d(x, y, x1_0, y1_0, sigma)
        phi_q = hermite_gaussian_mode_2d(x, y, qx, qy, sigma)
        return psi1 * phi_q

    # Define the integrand for the second source
    def integrand2(y, x):
        psi2 = gaussian_psf_2d(x, y, x2_0, y2_0, sigma)
        phi_q = hermite_gaussian_mode_2d(x, y, qx, qy, sigma)
        return psi2 * phi_q

    # Perform numerical integration
    x_limit = integration_limit * sigma
    y_limit = integration_limit * sigma

    overlap1, _ = dblquad(
        integrand1,
        -x_limit,
        x_limit,  # x limits
        lambda x: -y_limit,  # y lower limit
        lambda x: y_limit,  # y upper limit
    )

    overlap2, _ = dblquad(
        integrand2,
        -x_limit,
        x_limit,  # x limits
        lambda x: -y_limit,  # y lower limit
        lambda x: y_limit,  # y upper limit
    )

    # Calculate the probability (weighted by the amplitudes)
    P1_q = amp_1 * abs(overlap1) ** 2 + amp_2 * abs(overlap2) ** 2

    return P1_q


def calculate_overlap_factors_grid(
    qx_max,
    qy_max,
    params,
    integration_limit=10,
    exact=True,
):
    """
    Calculate overlap factors for a grid of mode indices (qx, qy).

    Note: using exact_overlap_factor_2d function right now to speed up calc,
    the output of numerical_overlap_factor_2d and exact_overlap_factor_2d
    have been tested to produce same results

    Parameters:
    -----------
    qx_max, qy_max (int): Maximum mode indices to calculate
    params (dict): Dictionary of PSF parameters
    integration_limit (float): Integration limit

    Returns:
    --------
    ndarray: 2D array of overlap factors, where element [qx, qy] is the overlap factor
             for mode (qx, qy)
    """

    overlap_factors = np.zeros((qx_max + 1, qy_max + 1))

    if exact:
        overlap_factors_2d = partial(exact_overlap_factor_2d, params=params)
    else:
        overlap_factors_2d = partial(
            numerical_overlap_factor_2d,
            params=params,
            integration_limit=integration_limit,
        )

    # This can be slow for large qx_max, qy_max
    for qx in range(qx_max + 1):
        for qy in range(qy_max + 1):
            overlap_factors[qx, qy] = overlap_factors_2d(qx, qy)

    return overlap_factors


def simulate_photon_counts_2d(
    params,
    N,
    qx_max=3,
    qy_max=3,
    integration_limit=10,
):
    """
    Simulate photon counts for 2D SPADE measurement.

    Parameters:
    -----------
    params (dict): Dictionary containing PSF parameters:
        - sigma: Width of the point-spread function
        - centroid_x, centroid_y: Coordinates of the centroid
        - sep_x, sep_y: Separation in x and y directions
        - amp_1, amp_2: Amplitudes of the two Gaussian components
    N (int): Average number of photons to detect
    qx_max, qy_max (int): Maximum mode indices to consider

    Returns:
    --------
    tuple: (mode_counts, total_photons)
        - mode_counts: 2D array with photon counts for each mode (qx, qy)
        - total_photons: Total number of photons detected
    """

    # Calculate theoretical probabilities for each mode
    probabilities = calculate_overlap_factors_grid(qx_max, qy_max, params)

    # Flatten the 2D probability array for multinomial sampling
    flat_probabilities = probabilities.flatten()

    # Normalize probabilities (to ensure they sum to 1)
    flat_probabilities = flat_probabilities / np.sum(flat_probabilities)

    # Sample from Poisson distribution to get the total number of photons
    # L = np.random.poisson(N)  # Actual number of photons detected
    L = N

    if L == 0:
        return np.zeros((qx_max + 1, qy_max + 1), dtype=int), 0

    # Sample from multinomial distribution to get photon counts in each mode
    flat_mode_counts = np.random.multinomial(L, flat_probabilities)

    # Reshape back to 2D array
    mode_counts = flat_mode_counts.reshape((qx_max + 1, qy_max + 1))

    return mode_counts, L


def estimate_separation_2d_mle(mode_counts, sigma):
    """
    Improved maximum likelihood estimator for the 2D separation that accounts
    for the joint distribution of photons in the 2D mode space.

    Parameters:
    -----------
    mode_counts (ndarray): 2D array with number of photons detected in each Hermite-Gaussian mode
    sigma (float): Width of the point-spread function

    Returns:
    --------
    tuple: (theta_x, theta_y, Q_x, Q_y)
        - theta_x: Estimated separation along x-axis
        - theta_y: Estimated separation along y-axis
        - Q_x: Quality factor for x direction
        - Q_y: Quality factor for y direction
    """

    # Total number of detected photons
    L = np.sum(mode_counts)
    if L == 0:
        return 0, 0, 0, 0  # No photons detected

    qx_max, qy_max = mode_counts.shape

    H_X = 0
    H_Y = 0
    for qx in range(qx_max):
        for qy in range(qy_max):
            H_X += qx * mode_counts[qx, qy]  # Sum of all qx indices weighted by counts
            H_Y += qy * mode_counts[qx, qy]  # Sum of all qy indices weighted by counts

    d_X_estimate = 4 * sigma * np.sqrt(H_X / L)
    d_Y_estimate = 4 * sigma * np.sqrt(H_Y / L)

    return d_X_estimate, d_Y_estimate, H_X, H_Y


def run_simulation(
    mean_photon_num=1000,
    num_trials=10,
    num_separations=10,
    qx_max=3,
    qy_max=3,
    psf_params=None,
    integraton_limit=15,
    sep_limit=10,
):
    """
    Run a simulation to estimate separation over a 2D grid of x and y separation values.

    Parameters:
    -----------
    sigma (float): Width of the point-spread function
    N (int): Number of photons per trial
    num_trials (int): Number of trials to run for each separation combination
    num_separations (int): Number of different separation values to test in each dimension
    qx_max, qy_max (int): Maximum mode indices to consider
    params (dict): Base parameters for the simulation

    Returns:
    --------
    tuple: (theta2, theta2_est)
        - theta2: True separation magnitudes (sqrt(x²+y²))
        - theta2_est: Estimated separation magnitudes for each trial
    """

    # Initialize params dictionary if not provided
    if psf_params is None:
        psf_params = {
            "sigma": 1,
            "centroid_x": 0.0,
            "centroid_y": 0.0,
            "sep_x": 0.0,
            "sep_y": 0.0,
            "amp_1": 0.5,
            "amp_2": 0.5,
        }
    sigma = psf_params.get("sigma", 1)

    # Generate range of true separations (from 0 to 10*sigma)
    theta2_x_arr = np.linspace(0, sep_limit * sigma, num_separations)
    theta2_y_arr = np.linspace(0, sep_limit * sigma, num_separations)

    ## Arrays to store results ##
    # True separation magnitude
    theta2 = np.zeros((num_separations, num_separations))

    # Estimated separation magnitude
    theta2x = theta2y = theta2_est = np.zeros(
        (num_separations, num_separations, num_trials)
    )

    # Create progress bar
    pbar = tqdm(total=num_separations * num_separations, desc="Simulation Progress")

    # Nested for loop to cycle through all possible combinations of theta in 2D
    for i, theta2_x in enumerate(theta2_x_arr):
        for j, theta2_y in enumerate(theta2_y_arr):
            # Set the separations in the params
            psf_params["sep_x"] = theta2_x
            psf_params["sep_y"] = theta2_y

            # Calculate true separation magnitude
            theta2[i, j] = np.sqrt(theta2_x**2 + theta2_y**2)

            for k in range(num_trials):
                # Simulate photon counts
                mode_counts, _ = simulate_photon_counts_2d(
                    psf_params,
                    mean_photon_num,
                    qx_max=qx_max,
                    qy_max=qy_max,
                    integration_limit=integraton_limit,
                )

                # Estimate separation
                theta_x_est, theta_y_est, _, _ = estimate_separation_2d_mle(
                    mode_counts, sigma
                )

                # Store true and estimated magnitudes
                theta2x[i, j, k] = theta_x_est
                theta2y[i, j, k] = theta_y_est
                theta2_est[i, j, k] = np.sqrt(theta_x_est**2 + theta_y_est**2)

            # Update progress bar
            pbar.update(1)

    # Close progress bar
    pbar.close()

    return theta2x, theta2y, theta2_est, theta2, theta2_x_arr, theta2_y_arr


def plot_simulation_results_2d(
    results,
    params,
    simlation_param,
    figsize=(12, 5),
):
    """
    Plot the results of the 2D separation estimation simulation.

    Parameters:
    -----------
    theta2 (ndarray): True separation magnitudes
    theta2_est (ndarray): Estimated separation magnitudes
    figsize (tuple): Figure size

    Returns:
    --------
    fig, axes: Figure and axes objects
    """

    theta2x, theta2y, theta2_est, theta2, theta2_x_arr, theta2_y_arr = results

    sigma = params.get("sigma", 1)
    L = simlation_param.get("mean_photon_num", 100)
    qx_max = simlation_param.get("qx_max", 1)
    qy_max = simlation_param.get("qy_max", 1)

    # Calculate mean and std of estimates across trials
    theta2x_mean = np.mean(theta2x, axis=-1)
    theta2x_std = np.std(theta2x, axis=-1)
    theta2y_mean = np.mean(theta2y, axis=-1)
    theta2y_std = np.std(theta2y, axis=-1)
    theta2_mean = np.sqrt(theta2x_mean**2 + theta2y_mean**2)
    theta2_est_mean = np.mean(theta2_est, axis=-1)

    # Create figure with two subplots
    fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(
        rf"Num photons = {L}, $\sigma_x$ = {sigma}, $\sigma_y$ = {sigma}, $qMAX$ = {qx_max, qy_max}"
    )

    # Plot mean estimated separation
    contour = ax.contourf(
        theta2_x_arr,
        theta2_y_arr,
        # np.abs(theta2_mean - theta2) / sigma,
        np.abs(theta2_est_mean - theta2) / sigma,
        origin="lower",
        cmap="viridis",
    )

    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(r"Absolute Difference / $\sigma$")
    ax.set_title("Mean Estimated Diff")
    ax.set_xlabel(r"X separation / $\sigma$")
    ax.set_ylabel(r"Y separation  / $\sigma$")

    ax1.pcolormesh(
        theta2_x_arr,
        theta2_y_arr,
        theta2x_std**2 * (L / 4 * sigma**2),
    )
    ax1.set_title("Simulated MSE of SPADE in X-dir")
    ax1.set_xlabel(r"X separation / $\sigma$")
    ax1.set_ylabel(r"Y separation  / $\sigma$")
    ax1.grid()

    im = ax2.pcolormesh(
        theta2_x_arr,
        theta2_y_arr,
        theta2y_std**2 * (L / 4 * sigma**2),
    )
    ax2.set_title("Simulated MSE of SPADE in Y-dir")
    ax2.set_xlabel(r"X separation / $\sigma$")
    ax2.set_ylabel(r"Y separation  / $\sigma$")
    ax2.grid()
    fig = ax2.get_figure()
    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label(r"MSE / $(L/4\sigma^2)$")

    plt.tight_layout()
    return fig, ax


# Example usage:
if __name__ == "__main__":
    # Base parameters
    psf_params = {
        "sigma": 1.0,
        "centroid_x": 0.0,
        "centroid_y": 0.0,
        "sep_x": 0.0,
        "sep_y": 0.0,
        "amp_1": 0.5,
        "amp_2": 0.5,
    }

    for qx_max, qy_max in [(2, 2)]:
        simulation_param = {
            "qx_max": qx_max,
            "qy_max": qy_max,
            "mean_photon_num": 1000,
            "num_trials": 1000,
            "num_separations": 50,
            "integraton_limit": 15,
            "sep_limit": 10,
            "psf_params": psf_params,
        }

        results = run_simulation(**simulation_param)

        # Plot results
        fig, axes = plot_simulation_results_2d(
            results,
            psf_params,
            simulation_param,
            figsize=(16, 5),
        )

        # Create the data directory if it doesn't exist
        if not os.path.exists("./data"):
            os.makedirs("./data")

        # Save the figure
        file_dir = (
            f"./data/simulation_results_L_"
            f"{simulation_param['mean_photon_num']}_q_"
            f"{simulation_param['qx_max']}{simulation_param['qy_max']}.png"
        )
        fig.savefig(
            file_dir,
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Figure saved to {file_dir}")
    plt.show()
