from sage.all import *
import numpy as np
from IPython.display import display, Math, Markdown


def gaussian_psf_2d(x, y, x0, y0, sigma_x, sigma_y):
    """
    2D Gaussian point-spread function centered at (x0, y0).
        Parameters:
            x, y (float): Spatial coordinates
            x0, y0 (float): Center positions
            sigma (float): Width of the point-spread function
        Returns:
            float: Value of the 2D Gaussian PSF at position (x, y)
    """
    normalization = 1 / sqrt(2 * pi * sigma_x * sigma_y)
    return normalization * exp(
        -((x - x0) ** 2 / (4 * sigma_x**2)) - ((y - y0) ** 2 / (4 * sigma_y**2))
    )


def hermite_gaussian_mode_2d(x, y, qx, qy, sigma_x, sigma_y):
    """
    2D Hermite-Gaussian spatial mode function.
        Parameters:
            x, y (float): Spatial coordinates
            qx, qy (int): Mode indices along x and y
            sigma (float): Width of the point-spread function
        Returns:
            float: Value of the 2D Hermite-Gaussian mode at position (x, y)
    """

    normalization_x = 1 / sqrt(2**qx * factorial(qx))
    normalization_y = 1 / sqrt(2**qy * factorial(qy))

    # Normalized coordinates
    x_normalized = x / (sqrt(2) * sigma_x)
    y_normalized = y / (sqrt(2) * sigma_y)

    # Create symbolic variables for Hermite polynomials
    t_x = var("t_x")
    t_y = var("t_y")

    # Hermite polynomials
    H_qx = hermite(qx, t_x)
    H_qy = hermite(qy, t_y)

    # Substitute the normalized coordinates
    H_qx_eval = H_qx.substitute({t_x: x_normalized})
    H_qy_eval = H_qy.substitute({t_y: y_normalized})

    # Prefactor
    prefactor = 1 / sqrt(2 * pi * sigma_x * sigma_y)

    # Evaluate 2D Hermite-Gaussian mode
    return (
        prefactor
        * normalization_x
        * normalization_y
        * H_qx_eval
        * H_qy_eval
        * exp(-(x**2 / (4 * sigma_x**2)) - (y**2 / (4 * sigma_y**2)))
    )


def lp_mode(p, l, x, y, w0=1.0):
    """
    Generate a Laguerre-Polynomial (LP) mode in 2D.

    Parameters:
    p (int): Radial index (p >= 0)
    l (int): Azimuthal index (can be positive or negative)
    x, y (numeric or symbolic): Cartesian coordinates
    w0 (numeric): Beam waist (default=1.0)

    Returns:
    Expression: LP mode field in 2D

    Note: This implementation uses SageMath's built-in functions and can be
    evaluated using both numeric and symbolic inputs.
    """

    # Convert to polar coordinates
    r = sqrt(x**2 + y**2)
    phi = atan2(y, x)

    # Normalized radius
    rho = (sqrt(2) * r / w0) ** 2

    # Calculate absolute value of azimuthal index
    abs_l = abs_symbolic(l)

    # Calculate normalization constant
    if p == 0 and l == 0:
        norm = sqrt(2 / (pi * w0**2))
    else:
        norm = sqrt(2 * factorial(p) / (pi * w0**2 * factorial(p + abs_l)))

    # Generate the Laguerre polynomial term
    laguerre_term = gen_laguerre(p, abs_l, rho)

    # Generate the radial envelope
    radial_term = (sqrt(2) * r / w0) ** abs_l * exp(-rho / 2)

    # Generate the azimuthal phase term
    if l >= 0:
        azimuthal_term = cos(l * phi) + sin(l * phi) * I
    else:
        azimuthal_term = cos(abs_l * phi) - sin(abs_l * phi) * I

    # Combine all terms to form the LP mode
    mode = norm * radial_term * laguerre_term * azimuthal_term

    return mode


def overlap_analytical_lp(x, y, x0, y0, sigma_x, sigma_y, l, p):
    # Add necessary assumptions to help with integration
    assume(sigma_x > 0)
    assume(sigma_y > 0)

    # Calculate the PSF and HG functions
    psf = gaussian_psf_2d(x, y, x0, y0, sigma_x, sigma_y)
    lp = lp_mode(l, p, x, y, sigma_x)

    # Calculate the product
    product = psf * conjugate(lp)

    # Perform the integration
    # First with respect to x
    integral_x = integrate(product, x, -infinity, infinity)

    # Then with respect to y
    double_integral = integrate(integral_x, y, -infinity, infinity)

    return abs(double_integral) ** 2


def overlap_analytical_hg(x, y, x0, y0, sigma_x, sigma_y, qx, qy):
    # Calculate the PSF and HG functions
    psf = gaussian_psf_2d(x, y, x0, y0, sigma_x, sigma_y)
    hg = hermite_gaussian_mode_2d(x, y, qx, qy, sigma_x, sigma_y)

    # Calculate the product
    product = psf * hg

    # Perform the integration
    # First with respect to x
    integral_x = integrate(product, x, -infinity, infinity)

    # Then with respect to y
    double_integral = integrate(integral_x, y, -infinity, infinity)

    return abs(double_integral) ** 2


def overlap_analytical(x, y, x0, y0, sigma_x, sigma_y, mode="hg", **mode_params):
    """
    Calculate the overlap between a Gaussian PSF and either a Hermite-Gaussian or Laguerre-Polynomial mode.

    Parameters:
    -----------
    x, y : symbolic variables
        Spatial coordinates
    x0, y0 : float
        Center coordinates of the PSF
    sigma_x, sigma_y : float
        Standard deviations of the PSF in x and y directions
    mode : str, optional
        Mode type: 'hg' for Hermite-Gaussian or 'lp' for Laguerre-Polynomial (default: 'hg')
    **mode_params : dict
        Parameters specific to the chosen mode:
        - For 'hg' mode: qx, qy (quantum numbers)
        - For 'lp' mode: l, p (orbital angular momentum and radial quantum number)

    Returns:
    --------
    float
        Squared absolute value of the overlap integral
    """
    # Add necessary assumptions to help with integration
    assume(sigma_x > 0)
    assume(sigma_y > 0)

    # Calculate the PSF
    psf = gaussian_psf_2d(x, y, x0, y0, sigma_x, sigma_y)

    # Calculate the appropriate mode function based on the mode parameter
    if mode.lower() == "lp":
        # Extract LP mode parameters
        l, p = mode_params.get("l", 0), mode_params.get("p", 1)
        mode_function = lp_mode(l, p, x, y, sigma_x)
        product = psf * conjugate(mode_function)
    else:  # default to HG mode
        # Extract HG mode parameters
        qx, qy = mode_params.get("qx", 1), mode_params.get("qy", 1)
        mode_function = hermite_gaussian_mode_2d(x, y, qx, qy, sigma_x, sigma_y)
        product = psf * mode_function

    # Perform the integration
    # First with respect to x
    integral_x = integrate(product, x, -infinity, infinity)
    # Then with respect to y
    double_integral = integrate(integral_x, y, -infinity, infinity)

    return abs(double_integral) ** 2
