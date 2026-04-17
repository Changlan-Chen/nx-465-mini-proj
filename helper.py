import numpy as np

def mexican_hat_1d(x, A_exc, sigma_exc, A_inh, sigma_inh):
    return A_exc * np.exp(-x**2 / sigma_exc**2) - A_inh * np.exp(-x**2 / sigma_inh**2)

def relu(x):
    return np.maximum(0, x)