"""Differential privacy primitives.
"""
import numpy as np


def laplace_mechanism(val: float, sensitivity: float, epsilon: float):
    """Applies Laplace mechanism to a value, returning a prviatized version suitable for release.

    See section 3.3 of Dwork & Roth, "The Algorithmic Foundations of Differential Privacy" for more information.

    Args:
        val (np.ndarray): Vector to add noise to.
        sensitivity (float): Sensitivity of corresponding function. See section 3.3 of Dwork & Roth for more information.
        epsilon (float): Privacy parameter.
    """

    return val + np.random.laplace(loc=0.0, scale=(sensitivity / epsilon))
