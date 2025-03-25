import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from torchvision import datasets
import torch
from scipy.special import hermite


# Function to generate a Gaussian point spread function (PSF)
def gaussian_psf(x, y, sigma):
    """
    Generate a Gaussian point spread function.

    Parameters:
    x, y: Coordinates
    sigma: Width of the PSF

    Returns:
    Normalized Gaussian PSF
    """
    psf = np.exp(-(x**2 + y**2) / (4 * sigma**2))
    return psf / np.sum(psf)  # Normalize


# Function to convolve an image with the PSF
def blur_image(image, sigma):
    """
    Blur an image using a Gaussian PSF.

    Parameters:
    image: Input image (2D array)
    sigma: Width of the PSF

    Returns:
    Blurred image
    """
    h, w = image.shape
    # Create coordinate grids
    y, x = np.mgrid[-h // 2 : h // 2, -w // 2 : w // 2]

    # Generate PSF
    psf = gaussian_psf(x, y, sigma)

    # Perform convolution in Fourier domain
    image_fft = np.fft.fft2(image)
    psf_fft = np.fft.fft2(np.fft.ifftshift(psf), s=image.shape)
    blurred = np.real(np.fft.ifft2(image_fft * psf_fft))

    # Normalize
    if np.sum(blurred) > 0:
        blurred = blurred / np.sum(blurred)

    return blurred


# Function to simulate photon detection in direct imaging (DI)
def direct_imaging(blurred_image, N):
    """
    Simulate photon detection in direct imaging.

    Parameters:
    blurred_image: Blurred image (probability distribution)
    N: Number of photons

    Returns:
    Array of photon counts
    """
    h, w = blurred_image.shape
    flat_probs = blurred_image.flatten()

    # Sample photon detection events
    indices = np.random.choice(h * w, size=N, p=flat_probs)

    # Convert to photon counts
    photon_counts = np.zeros(h * w)
    for idx in indices:
        photon_counts[idx] += 1

    return photon_counts.reshape(h, w)


# Function to compute Hermite-Gaussian modes
def hermite_gaussian_mode(x, y, u, v, sigma):
    """
    Compute Hermite-Gaussian mode.

    Parameters:
    x, y: Coordinates
    u, v: Mode indices
    sigma: Width parameter

    Returns:
    HG mode values
    """

    norm_x = x / (np.sqrt(2) * sigma)
    norm_y = y / (np.sqrt(2) * sigma)

    # Hermite polynomials
    Hu = hermite(u)(norm_x)
    Hv = hermite(v)(norm_y)

    # Gaussian envelope
    gaussian = np.exp(-(norm_x**2 + norm_y**2) / 2)

    norm_factor = 1.0 / np.sqrt(2**u * 2**v * factorial(u) * factorial(v) * np.pi)

    return norm_factor * Hu * Hv * gaussian


# Function to project image onto Hermite-Gaussian modes
def hg_spade(image, sigma, modes):
    """
    Spatial-mode demultiplexing with Hermite-Gaussian modes.

    Parameters:
    image: Input image
    sigma: Width parameter
    modes: List of (u,v) tuples for the modes

    Returns:
    Projection values for each mode
    """
    h, w = image.shape
    y, x = np.mgrid[-h // 2 : h // 2, -w // 2 : w // 2]

    projections = {}

    for u, v in modes:
        # Generate the HG mode
        hg_mode = hermite_gaussian_mode(x, y, u, v, sigma)

        # Calculate projection (overlap integral)
        projection = np.sum(image * np.abs(hg_mode) ** 2)

        mode_name = f"HG{u}{v}"
        projections[mode_name] = projection

    return projections


# Function to simulate SPADE photon detection
def simulate_spade_detection(projections, N):
    """
    Simulate photon detection in SPADE.

    Parameters:
    projections: Mode projections (probabilities)
    N: Number of photons

    Returns:
    Photon counts for each mode
    """
    modes = list(projections.keys())
    probs = np.array([projections[mode] for mode in modes])

    # Normalize if needed
    if np.sum(probs) > 0:
        probs = probs / np.sum(probs)

    # Sample photon detection events
    counts = np.zeros(len(modes))

    # Generate multinomial samples
    if np.sum(probs) > 0:  # Check for valid probability distribution
        counts = np.random.multinomial(N, probs)

    return {modes[i]: counts[i] for i in range(len(modes))}


# Modified SPADE for third order moments
def modified_spade(image, sigma, N, angle_phi=None):
    """
    Implement the modified SPADE approach for third order moments.

    Parameters:
    image: Input image
    sigma: Width parameter
    N: Number of photons
    angle_phi: Angle for optimization (if None, will be calculated)

    Returns:
    Feature vector from the photon counts
    """
    h, w = image.shape
    y, x = np.mgrid[-h // 2 : h // 2, -w // 2 : w // 2]

    # Basic HG modes
    modes = [
        (0, 0),
        (1, 0),
        (0, 1),
        (2, 0),
        (0, 2),
        (1, 1),
        (3, 0),
        (0, 3),
        (2, 1),
        (1, 2),
    ]  # Including higher orders

    # Project image onto the basic HG modes
    projections = hg_spade(image, sigma, modes)

    # Compute optimized angle if not provided
    if angle_phi is None:
        # Implement Eq. 42 and 43 from the paper
        angle_phi = np.arcsin(
            2 * sigma / np.sqrt((2 * sigma) ** 2 + 2 * (2 * sigma) ** 4)
        )

    # Create rotated combinations as described in the paper
    sin_phi = np.sin(angle_phi)
    cos_phi = np.cos(angle_phi)

    # We'll split the photons between the standard and rotated bases
    N_half = N // 2

    # Standard HG modes detection
    std_counts = simulate_spade_detection(projections, N_half)

    # Create the rotated basis projections (HG01±HG02)/√2 as in the paper
    rotated_projections = {}

    # Adding the first combination: sin(φ)|HG01⟩ + cos(φ)|HG02⟩
    rotated_projections["HG01+HG02"] = (
        sin_phi * projections["HG01"] + cos_phi * projections["HG02"]
    )

    # Adding the second combination: cos(φ)|HG01⟩ - sin(φ)|HG02⟩
    rotated_projections["HG01-HG02"] = (
        cos_phi * projections["HG01"] - sin_phi * projections["HG02"]
    )

    # Add other original modes to the rotated projections
    for mode in projections:
        if mode not in ["HG01", "HG02"]:
            rotated_projections[mode] = projections[mode]

    # Simulate detection in the rotated basis
    rot_counts = simulate_spade_detection(rotated_projections, N - N_half)

    # Combine both sets of counts into a feature vector
    feature_vector = {}
    feature_vector.update(std_counts)

    # Rename the rotated basis counts to avoid key collision
    for k, v in rot_counts.items():
        if k in ["HG01+HG02", "HG01-HG02"]:
            feature_vector[k] = v
        else:
            feature_vector[f"{k}_rot"] = v

    return feature_vector


# Load and prepare MNIST dataset
def load_mnist_subset(digits=None, num_samples=2000):
    """
    Load a subset of the MNIST dataset.

    Parameters:
    digits: List of digits to include (default: all)
    num_samples: Number of samples per class

    Returns:
    Images and labels
    """
    try:
        # First try to load using torchvision
        try:
            # Set a more reliable mirror for downloading MNIST
            import os

            os.environ["TORCHVISION_MNIST_MIRROR"] = (
                "https://ossci-datasets.s3.amazonaws.com/mnist"
            )
            mnist_train = datasets.MNIST("./data", train=True, download=True)

            images = mnist_train.data.numpy()
            labels = mnist_train.targets.numpy()
        except:
            # If the above fails, try direct download using scikit-learn
            from sklearn.datasets import fetch_openml

            print("Downloading MNIST from scikit-learn...")
            mnist = fetch_openml("mnist_784", version=1, parser="auto")
            images = mnist.data.to_numpy().reshape(-1, 28, 28)
            labels = mnist.target.astype(int).to_numpy()
            print("Download complete!")
    except Exception as e:
        # If all else fails, generate synthetic data for testing the algorithm
        print(f"Error downloading MNIST: {e}")
        print("Generating synthetic digit-like data for testing...")

        # Create synthetic data with 10 digit classes
        n_digits = 10 if digits is None else len(digits)
        n_samples_per_digit = 200

        # Map digit indices if specific digits are requested
        digit_map = (
            {i: d for i, d in enumerate(digits)}
            if digits is not None
            else {i: i for i in range(10)}
        )

        images = []
        labels = []

        for i in range(n_digits):
            # Create simple patterns resembling digits (very simplified)
            img_template = np.zeros((28, 28))

            # Add different patterns for each digit
            if digit_map[i] == 0:  # Circle for 0
                r, c = np.ogrid[0:28, 0:28]
                center = 14
                radius = 8
                img_template[(r - center) ** 2 + (c - center) ** 2 < radius**2] = 1.0
                img_template[
                    (r - center) ** 2 + (c - center) ** 2 < (radius - 3) ** 2
                ] = 0.0
            elif digit_map[i] == 1:  # Vertical line for 1
                img_template[:, 14:17] = 1.0
            else:  # Random patterns for other digits
                np.random.seed(digit_map[i])  # For reproducibility
                for _ in range(5):  # Add some random lines/shapes
                    r, c = np.random.randint(0, 20, 2)
                    h, w = np.random.randint(5, 15, 2)
                    img_template[r : r + h, c : c + w] = 1.0

            # Add some random noise and variation
            for j in range(n_samples_per_digit):
                noise = np.random.normal(0, 0.1, (28, 28))
                img = np.clip(img_template + noise, 0, 1)
                images.append(img)
                labels.append(digit_map[i])

        images = np.array(images)
        labels = np.array(labels)
        print(f"Generated {len(images)} synthetic samples")

    # Filter by requested digits
    if digits is not None:
        mask = np.zeros_like(labels, dtype=bool)
        for digit in digits:
            mask = mask | (labels == digit)

        images = images[mask]
        labels = labels[mask]

    # Ensure balanced classes and limit samples
    unique_labels = np.unique(labels)
    selected_images = []
    selected_labels = []

    for label in unique_labels:
        indices = np.where(labels == label)[0]
        selected_indices = indices[: min(num_samples, len(indices))]

        selected_images.append(images[selected_indices])
        selected_labels.append(labels[selected_indices])

    # Handle case where there might be no samples
    if len(selected_images) == 0:
        raise ValueError("No samples found for the requested digits")

    selected_images = np.vstack(selected_images)
    selected_labels = np.concatenate(selected_labels)

    # Normalize images if not already normalized
    if selected_images.max() > 1.0:
        selected_images = selected_images / 255.0

    return selected_images, selected_labels


# Main function to run the complete experiment
def run_spade_ml_experiment(
    digits=[0, 1], sigma_rel=9.5, N_photons=1000, n_samples=200
):
    """
    Run the complete SPADE + ML experiment.

    Parameters:
    digits: List of digits to classify
    sigma_rel: Relative width of the PSF
    N_photons: Number of photons
    n_samples: Number of samples per class

    Returns:
    Accuracy and other results
    """
    # Load MNIST subset
    images, labels = load_mnist_subset(digits=digits, num_samples=n_samples)

    # Pad images to avoid cutoff
    padded_images = np.zeros((len(images), 80, 80))
    for i, img in enumerate(images):
        h, w = img.shape
        padded_images[i, 40 - h // 2 : 40 + h // 2, 40 - w // 2 : 40 + w // 2] = img

    # Calculate the feature vectors for all images
    features_spade = []
    features_di = []

    # Define scaling factor
    f = 1.0  # We'll use sigma_rel directly
    sigma = sigma_rel / f

    # Define HG modes for regular SPADE
    basic_modes = [(0, 0), (1, 0), (0, 1)]

    # Add modified SPADE modes if we're classifying 6 vs 9
    if set(digits) == {6, 9}:
        use_modified_spade = True
    else:
        use_modified_spade = False

    print(f"Processing {len(padded_images)} images...")

    for i, img in enumerate(padded_images):
        # Blur the image according to the PSF
        blurred = blur_image(img, sigma)

        # For direct imaging
        di_counts = direct_imaging(blurred, N_photons)

        # Flatten and normalize
        di_features = (
            di_counts.flatten() / N_photons if N_photons > 0 else di_counts.flatten()
        )
        features_di.append(di_features)

        # For SPADE
        if use_modified_spade:
            # Use modified SPADE for third-order moments
            spade_counts = modified_spade(blurred, sigma, N_photons)

            # Convert dictionary to vector
            all_keys = sorted(list(set().union(*[d.keys() for d in [spade_counts]])))
            spade_features = (
                np.array([spade_counts.get(k, 0) for k in all_keys]) / N_photons
            )
        else:
            # Use regular SPADE for second-order moments
            spade_projections = hg_spade(blurred, sigma, basic_modes)
            spade_counts = simulate_spade_detection(spade_projections, N_photons)

            # Convert dictionary to vector
            mode_names = sorted(list(spade_counts.keys()))
            spade_features = (
                np.array([spade_counts[mode] for mode in mode_names]) / N_photons
            )

        features_spade.append(spade_features)

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1} images")

    # Convert to numpy arrays
    features_spade = np.array(features_spade)
    features_di = np.array(features_di)

    # Split into training and test sets
    X_train_spade, X_test_spade, y_train, y_test = train_test_split(
        features_spade, labels, test_size=0.3, random_state=42, stratify=labels
    )

    X_train_di, X_test_di, _, _ = train_test_split(
        features_di, labels, test_size=0.3, random_state=42, stratify=labels
    )

    # Train models
    # Random Forest for SPADE
    rf_spade = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_spade.fit(X_train_spade, y_train)

    # Neural Network for DI
    mlp_di = MLPClassifier(
        hidden_layer_sizes=(256, 128, 32),
        activation="relu",
        random_state=42,
        max_iter=300,
    )
    mlp_di.fit(X_train_di, y_train)

    # Random Forest for DI (for comparison)
    rf_di = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_di.fit(X_train_di, y_train)

    # Evaluate models
    y_pred_spade = rf_spade.predict(X_test_spade)
    y_pred_di_mlp = mlp_di.predict(X_test_di)
    y_pred_di_rf = rf_di.predict(X_test_di)

    acc_spade = accuracy_score(y_test, y_pred_spade)
    acc_di_mlp = accuracy_score(y_test, y_pred_di_mlp)
    acc_di_rf = accuracy_score(y_test, y_pred_di_rf)

    print(f"Classification results for sigma={sigma_rel}, N={N_photons}:")
    print(f"SPADE + RF Accuracy: {acc_spade:.4f}")
    print(f"DI + MLP Accuracy: {acc_di_mlp:.4f}")
    print(f"DI + RF Accuracy: {acc_di_rf:.4f}")

    # Compute confusion matrices
    cm_spade = confusion_matrix(y_test, y_pred_spade)
    cm_di_mlp = confusion_matrix(y_test, y_pred_di_mlp)

    return {
        "accuracy": {"spade_rf": acc_spade, "di_mlp": acc_di_mlp, "di_rf": acc_di_rf},
        "confusion_matrices": {"spade_rf": cm_spade, "di_mlp": cm_di_mlp},
        "features": {
            "spade": (X_train_spade, X_test_spade),
            "di": (X_train_di, X_test_di),
        },
        "labels": (y_train, y_test),
    }


# Plot confusion matrix
def plot_confusion_matrix(cm, classes, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()


# Example usage:
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SPADE ML experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        default="binary_01",
        choices=["binary_01", "binary_69", "multi", "all"],
        help="Which experiment to run",
    )
    parser.add_argument(
        "--sigma", type=float, default=10, help="Relative width of the PSF"
    )
    parser.add_argument("--photons", type=int, default=1000, help="Number of photons")
    parser.add_argument(
        "--samples", type=int, default=100, help="Number of samples per class"
    )

    args = parser.parse_args()

    try:
        if args.experiment == "binary_01" or args.experiment == "all":
            print("\n--- Running binary classification 0 vs 1 experiment ---")
            results_01 = run_spade_ml_experiment(
                digits=[0, 1],
                sigma_rel=args.sigma,
                N_photons=args.photons,
                n_samples=args.samples,
            )

            # Plot confusion matrix
            plot_confusion_matrix(
                results_01["confusion_matrices"]["spade_rf"],
                classes=[0, 1],
                title=f"SPADE + RF Confusion Matrix (σ={args.sigma}, N={args.photons})",
            )

        if args.experiment == "binary_69" or args.experiment == "all":
            print("\n--- Running binary classification 6 vs 9 experiment ---")
            # For 6 vs 9, we use more photons as recommended in the paper
            photons = args.photons if args.photons >= 5000 else 5000
            results_69 = run_spade_ml_experiment(
                digits=[6, 9],
                sigma_rel=args.sigma,
                N_photons=photons,
                n_samples=args.samples,
            )

            # Plot confusion matrix
            plot_confusion_matrix(
                results_69["confusion_matrices"]["spade_rf"],
                classes=[6, 9],
                title=f"Modified SPADE + RF Confusion Matrix (σ={args.sigma}, N={photons})",
            )

        if args.experiment == "multi" or args.experiment == "all":
            print("\n--- Running multi-class classification experiment ---")
            # For multi-class, we use more photons too
            photons = args.photons if args.photons >= 5000 else 5000
            results_multi = run_spade_ml_experiment(
                digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                sigma_rel=args.sigma,
                N_photons=photons,
                n_samples=max(
                    20, args.samples // 5
                ),  # Reduce samples to keep memory usage reasonable
            )

            # Plot multi-class confusion matrix
            plot_confusion_matrix(
                results_multi["confusion_matrices"]["spade_rf"],
                classes=list(range(10)),
                title=f"SPADE + RF Multi-class Confusion Matrix (σ={args.sigma}, N={photons})",
            )

    except Exception as e:
        import traceback

        print(f"Error during experiment: {e}")
        traceback.print_exc()
        print("\nTrying a simpler experiment with fewer samples and photons...")

        # Fallback to a very simple experiment
        simple_results = run_spade_ml_experiment(
            digits=[0, 1], sigma_rel=10, N_photons=100, n_samples=20
        )

        print(
            f"Simple experiment results: SPADE accuracy = {simple_results['accuracy']['spade_rf']:.4f}"
        )
