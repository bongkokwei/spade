import numpy as np
from scipy.special import hermite, factorial
from scipy.integrate import quad
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def gaussian_psf(x, x0, sigma):
    """
    Gaussian point-spread function centered at x0.

    Parameters:
    x (float): Spatial coordinate
    x0 (float): Center position
    sigma (float): Width of the point-spread function

    Returns:
    float: Value of the Gaussian PSF at position x
    """
    return (1 / (2 * np.pi * sigma**2)) ** (1 / 4) * np.exp(
        -((x - x0) ** 2) / (4 * sigma**2)
    )


def hermite_gaussian_mode(x, q, sigma):
    """
    Hermite-Gaussian spatial mode function.

    Parameters:
    x (float): Spatial coordinate
    q (int): Mode index
    sigma (float): Width of the point-spread function

    Returns:
    float: Value of the Hermite-Gaussian mode at position x
    """
    prefactor = (1 / (2 * np.pi * sigma**2)) ** (1 / 4)
    normalization = 1 / np.sqrt(2**q * factorial(q))

    # Generate Hermite polynomial of order q
    H_q = hermite(q)

    # Evaluate the Hermite-Gaussian mode
    x_normalized = x / (np.sqrt(2) * sigma)
    return (
        prefactor * normalization * H_q(x_normalized) * np.exp(-(x**2) / (4 * sigma**2))
    )


def numerical_overlap_factor(q, theta2, sigma, integration_limit=10):
    """
    Numerically calculate the overlap factor (probability of detecting a photon in the qth mode).

    Parameters:
    q (int): Mode index
    theta2 (float): Separation between the two sources
    sigma (float): Width of the point-spread function
    integration_limit (float): Integration limit (should be large enough)

    Returns:
    float: Numerically computed probability of detecting a photon in the qth mode
    """
    # Source positions (assuming centroid at zero)
    x1 = -theta2 / 2
    x2 = theta2 / 2

    # Define the integrand for the first source
    def integrand1(x):
        psi1 = gaussian_psf(x, x1, sigma)
        phi_q = hermite_gaussian_mode(x, q, sigma)
        return psi1 * phi_q

    # Define the integrand for the second source
    def integrand2(x):
        psi2 = gaussian_psf(x, x2, sigma)
        phi_q = hermite_gaussian_mode(x, q, sigma)
        return psi2 * phi_q

    # Perform numerical integration
    overlap1, _ = quad(
        integrand1, -integration_limit * sigma, integration_limit * sigma
    )
    overlap2, _ = quad(
        integrand2, -integration_limit * sigma, integration_limit * sigma
    )

    # Calculate the probability (1/2 weight for each source)
    P1_q = (1 / 2) * (abs(overlap1) ** 2 + abs(overlap2) ** 2)

    return P1_q


def analytical_overlap_factor(q, theta2, sigma):
    """
    Calculate the overlap factor using the analytical formula (equation 4.5).

    Parameters:
    q (int): Mode index
    theta2 (float): Separation between the two sources
    sigma (float): Width of the point-spread function

    Returns:
    float: Analytically computed probability of detecting a photon in the qth mode
    """
    Q = (theta2**2) / (16 * sigma**2)

    if q == 0:
        return np.exp(-Q)
    else:
        return np.exp(-Q) * (Q**q) / factorial(q)


# Compare numerical and analytical results for various separations
def compare_results(sigma=1.0, q_max=3):
    separations = np.linspace(0, 10 * sigma, 100)

    print("Comparison of numerical and analytical overlap factors:")
    print("θ₂/σ | q | Numerical | Analytical | Difference")
    print("-------------------------------------------------")

    for theta2 in separations:
        for q in range(q_max + 1):
            num_result = numerical_overlap_factor(q, theta2, sigma)
            ana_result = analytical_overlap_factor(q, theta2, sigma)
            diff = abs(num_result - ana_result)

            print(
                f"{theta2/sigma:.2f} | {q} | {num_result:.6f} | {ana_result:.6f} | {diff:.6f}"
            )
        print("-------------------------------------------------")


def estimate_separation_mle(mode_counts, sigma):
    """
    Maximum likelihood estimator for the separation between two point sources.

    Parameters:
    mode_counts (list or array): Number of photons detected in each Hermite-Gaussian mode
                                [m₀, m₁, m₂, ...]
    sigma (float): Width of the point-spread function

    Returns:
    float: Estimated separation θ₂
    """
    # Total number of detected photons
    L = sum(mode_counts)

    if L == 0:
        return 0  # No photons detected, return zero or some default value

    # Calculate Q using equation 4.7
    Q_ml = 0
    for q, count in enumerate(mode_counts):
        Q_ml += q * count
    Q_ml /= L

    # Calculate separation estimate
    theta2_ml = 4 * sigma * np.sqrt(Q_ml)

    return theta2_ml


def simulate_photon_counts(theta2_true, sigma, N, q_max=5):
    """
    Simulate photon counts for SPADE measurement.

    Parameters:
    theta2_true (float): True separation between sources
    sigma (float): Width of the point-spread function
    N (int): Average number of photons to detect
    q_max (int): Maximum mode index to consider

    Returns:
    tuple: (mode_counts, total_photons)
    """

    # Calculate theoretical probabilities for each mode
    probabilities = np.zeros(q_max + 1)
    for q in range(q_max + 1):
        probabilities[q] = numerical_overlap_factor(
            q,
            theta2_true,
            sigma,
            integration_limit=10,
        )

    # Normalize probabilities (just in case)
    probabilities = probabilities / np.sum(probabilities)

    # Sample from multinomial distribution to get photon counts
    # We're simulating the number of photons detected in each mode
    # The total L doesn't have to be exactly N due to statistical fluctuations
    L = np.random.poisson(N)  # Actual number of photons detected
    if L == 0:
        return np.zeros(q_max + 1, dtype=int), 0

    mode_counts = np.random.multinomial(L, probabilities)
    return mode_counts, L


def estimate_separation_mle(mode_counts, sigma):
    """
    Maximum likelihood estimator for the separation between two point sources.

    Parameters:
    mode_counts (array): Number of photons detected in each Hermite-Gaussian mode
    sigma (float): Width of the point-spread function

    Returns:
    float: Estimated separation θ₂
    """
    # Total number of detected photons
    L = np.sum(mode_counts)

    if L == 0:
        return 0  # No photons detected

    # Calculate Q using equation 4.7
    Q_ml = 0
    for q, count in enumerate(mode_counts):
        Q_ml += q * count
    Q_ml /= L

    # Calculate separation estimate
    theta2_ml = 4 * sigma * np.sqrt(Q_ml)

    return theta2_ml, Q_ml


def estimate_separation_binary_spade(mode_counts, sigma):
    """
    Maximum likelihood estimator for binary SPADE.

    Parameters:
    mode_counts (array): Number of photons detected in each mode [m₀, m₁, m₂, ...]
    sigma (float): Width of the point-spread function

    Returns:
    float: Estimated separation θ₂
    """
    # Extract m0 and total photons
    m0 = mode_counts[0]
    L = np.sum(mode_counts)

    if L == 0 or m0 == 0:
        return 2 * sigma  # Default value for edge cases

    # Calculate Q using equation 5.4
    Q_ml = -np.log(m0 / L)

    # Calculate separation estimate
    theta2_ml = 4 * sigma * np.sqrt(Q_ml)

    return theta2_ml


def calculate_quantum_cramer_rao_bound(theta2, sigma, N):
    """
    Calculate the Cramér-Rao bound for separation estimation.

    Parameters:
    theta2 (float): Separation between sources
    sigma (float): Width of the point-spread function
    N (int): Number of photons

    Returns:
    float: Cramér-Rao bound for the variance of the estimator
    """
    # For full SPADE, the quantum Fisher information is N/(4*sigma²)
    # The bound on the variance is 1/(Fisher information)
    return N / (4 * sigma**2)


def direct_imaging_crb(theta2, sigma, N, num_points=1000):
    """
    Calculate the Cramér-Rao bound for direct imaging using numerical integration.

    Parameters:
    theta2 (float): Separation between sources
    sigma (float): Width of the point-spread function
    N (int): Number of photons
    num_points (int): Number of points for numerical integration

    Returns:
    float: Cramér-Rao bound for direct imaging
    """

    # Define intensity and its derivative
    def intensity(x):
        term1 = np.exp(-((x + theta2 / 2) ** 2) / (2 * sigma**2))
        term2 = np.exp(-((x - theta2 / 2) ** 2) / (2 * sigma**2))
        return (term1 + term2) / (2 * np.sqrt(2 * np.pi * sigma**2))

    def intensity_derivative(x):
        term1 = (
            -(x + theta2 / 2)
            * np.exp(-((x + theta2 / 2) ** 2) / (2 * sigma**2))
            / (2 * sigma**2)
        )
        term2 = (
            (x - theta2 / 2)
            * np.exp(-((x - theta2 / 2) ** 2) / (2 * sigma**2))
            / (2 * sigma**2)
        )
        return (term1 + term2) / (2 * np.sqrt(2 * np.pi * sigma**2))

    # Define integrand for Fisher information
    def integrand(x):
        return (intensity_derivative(x) ** 2) / (intensity(x))

    # Numerical integration
    x_values, dx = np.linspace(
        -50 * sigma,
        50 * sigma,
        num_points,
        retstep=True,
    )
    integrand_values = np.zeros_like(x_values)

    for i, x in enumerate(x_values):
        if intensity(x) > 1e-12:  # Avoid division by near-zero
            integrand_values[i] = integrand(x)

    # Trapezoidal integration
    fisher_info = N * np.trapezoid(integrand_values, dx=dx)

    # Return bound
    return fisher_info


def run_simulation(
    sigma=1.0,
    N=1000,
    num_trials=100,
    num_separations=20,
    qmax=16,
):
    """
    Run a simulation comparing true and estimated separations.

    Parameters:
    sigma (float): Width of the point-spread function
    N (int): Average number of photons per trial
    num_trials (int): Number of trials for each separation value
    num_separations (int): Number of different separation values to test
    """
    # Generate range of true separations (from 0 to 3*sigma)
    true_separations = np.linspace(0, 10 * sigma, num_separations)

    # Arrays to store results
    full_spade_theta2 = np.zeros((num_separations, num_trials))
    full_spade_q = np.zeros((num_separations, num_trials))

    # Create progress bar for outer loop
    theta2_pbar = tqdm(
        total=num_separations,
        desc="pogress",
    )

    # Run trials
    for i, theta2_true in enumerate(true_separations):
        # Create progress bar for inner loop
        trial_pbar = tqdm(
            total=num_trials,
            desc=f"trial {i+1}/{num_separations}",
            leave=False,
        )

        for j in range(num_trials):
            # Simulate photon counts
            mode_counts, L = simulate_photon_counts(
                theta2_true,
                sigma,
                N,
                q_max=qmax,
            )

            # Estimate separation using full SPADE
            theta_est, q_est = estimate_separation_mle(mode_counts, sigma)
            full_spade_theta2[i, j] = theta_est
            full_spade_q[i, j] = q_est
            # Update inner progress bar
            trial_pbar.update(1)

        # Close inner progress bar and update outer one
        trial_pbar.close()
        theta2_pbar.update(1)

    # Close outer progress bar
    theta2_pbar.close()

    # Calculate mean and standard deviation of estimates
    full_spade_mean = np.mean(full_spade_theta2, axis=1)
    full_spade_std = np.std(full_spade_theta2, axis=1)

    # Calculate Cramér-Rao bounds
    qcrb_values = np.array(
        [
            calculate_quantum_cramer_rao_bound(theta2, sigma, N)
            for theta2 in true_separations
        ]
    )

    direct_imaging_crb_values = np.array(
        [direct_imaging_crb(theta2, sigma, N) for theta2 in true_separations]
    )

    # Create figure and axes
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # Plot comparison of true vs estimated separation on first axis
    axs[0].plot(
        true_separations / sigma,
        true_separations / sigma,
        "k--",
        label="True Separation",
    )
    axs[0].errorbar(
        true_separations / sigma,
        full_spade_mean / sigma,
        yerr=full_spade_std / sigma,
        fmt="o-",
        label="Full SPADE",
        capsize=3,
    )
    axs[0].set_xlabel(r"True Separation ($\theta_2/\sigma$)")
    axs[0].set_ylabel(r"Estimated Separation ($\hat{\theta}_2/\sigma$)")
    axs[0].set_title(
        f"Comparison of True vs Estimated Separation with {qmax} detectors"
    )
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(
        true_separations / sigma,
        qcrb_values / qcrb_values,
        "k--",
        label="Quantum CRB",
    )
    axs[1].plot(
        true_separations / sigma,
        direct_imaging_crb_values / qcrb_values,
        "r--",
        label="Direct Imaging CRB",
    )
    axs[1].plot(
        true_separations / sigma,
        full_spade_std**2 / qcrb_values,
        "o-",
        label="Full SPADE Std Dev",
    )
    axs[1].set_xlabel(r"Separation ($\theta_2/\sigma$)")
    axs[1].set_ylabel(r"Fisher information ($N/4\sigma^2$)")
    axs[1].set_title(f"Estimation Precision vs Separation with {qmax} detectors")
    # axs[1].set_yscale("log")
    axs[1].grid(True)
    axs[1].legend()

    # Apply tight layout and display the plot
    fig.tight_layout()
    plt.show()

    # Return the simulation data
    return {
        "true_separations": true_separations,
        "full_spade_estimates": full_spade_theta2,
    }


if __name__ == "__main__":

    # Check if numerical overlap coincides with equation 4.5
    # compare_results(sigma=1.0, q_max=10)

    # Run the simulation
    results = run_simulation(
        sigma=1.0,  # width of PSF
        N=1000,  # mean photon number
        num_trials=50,
        num_separations=30,
        qmax=3,
    )

    print("Simulation complete!")
