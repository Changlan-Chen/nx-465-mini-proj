"""Shared helpers for NX-465 Mini-project 2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import convolve1d


def mexican_hat_1d(
    x,
    A_exc=0.3,
    sigma_exc=5.0,
    A_inh=0.2,
    sigma_inh=10.0,
):
    """1D Mexican-hat kernel from Exercise 0."""

    x = np.asarray(x)
    return A_exc * np.exp(-(x**2) / sigma_exc**2) - A_inh * np.exp(
        -(x**2) / sigma_inh**2
    )


def relu(x):
    """Rectified firing-rate nonlinearity."""

    return np.maximum(0, x)


@dataclass(frozen=True)
class IntegratorParams:
    """Numerical and model parameters for Exercise 1."""

    m: int = 100
    tau_ms: float = 10.0
    dt_ms: float = 1.0
    A_exc: float = 0.3
    sigma_exc: float = 5.0
    A_inh: float = 0.2
    sigma_inh: float = 10.0
    B0: float = 1.5
    alpha: float = 0.5
    weight_shift: float = 1.0


def circular_distances(m):
    """Signed circular distances for a ring with m neurons."""

    distances = np.arange(m, dtype=float)
    return np.where(distances <= m / 2, distances, distances - m)


def shifted_kernels(params):
    """Kernels from the right- and left-shifting populations."""

    d = circular_distances(params.m)
    W_right = mexican_hat_1d(
        d - params.weight_shift,
        params.A_exc,
        params.sigma_exc,
        params.A_inh,
        params.sigma_inh,
    )
    W_left = mexican_hat_1d(
        d + params.weight_shift,
        params.A_exc,
        params.sigma_exc,
        params.A_inh,
        params.sigma_inh,
    )
    return W_right, W_left


def periodic_convolve1d(rate, weights):
    """Periodic 1D convolution using scipy.ndimage.convolve1d."""

    return convolve1d(rate, weights, mode="wrap")


class TwoPopulationVelocityIntegrator:
    """Two-population ring attractor with velocity-modulated inputs."""

    def __init__(self, params=None):
        self.params = params or IntegratorParams()
        self.W_right, self.W_left = shifted_kernels(self.params)

    def simulate(self, velocity, seed=0, initial_state=None):
        """Run forward Euler integration for a supplied velocity trace."""

        p = self.params
        velocity = np.asarray(velocity, dtype=float)
        rng = np.random.default_rng(seed)

        if initial_state is None:
            s_right = rng.uniform(0.0, 0.1, p.m)
            s_left = rng.uniform(0.0, 0.1, p.m)
        else:
            s_right, s_left = (np.array(arr, dtype=float, copy=True) for arr in initial_state)

        n_steps = len(velocity)
        rate_right = np.zeros((n_steps, p.m))
        rate_left = np.zeros((n_steps, p.m))
        rate_total = np.zeros((n_steps, p.m))
        potential_right = np.zeros((n_steps, p.m))
        potential_left = np.zeros((n_steps, p.m))

        for t, v_t in enumerate(velocity):
            r_right = relu(s_right)
            r_left = relu(s_left)

            rate_right[t] = r_right
            rate_left[t] = r_left
            rate_total[t] = r_right + r_left
            potential_right[t] = s_right
            potential_left[t] = s_left

            recurrent = periodic_convolve1d(r_right, self.W_right) + periodic_convolve1d(
                r_left, self.W_left
            )
            B_right = p.B0 * (1.0 + p.alpha * v_t)
            B_left = p.B0 * (1.0 - p.alpha * v_t)

            s_right += (p.dt_ms / p.tau_ms) * (-s_right + recurrent + B_right)
            s_left += (p.dt_ms / p.tau_ms) * (-s_left + recurrent + B_left)

        return {
            "velocity": velocity,
            "rate_right": rate_right,
            "rate_left": rate_left,
            "rate_total": rate_total,
            "potential_right": potential_right,
            "potential_left": potential_left,
            "final_state": (s_right, s_left),
        }


def step_velocity_protocol(
    total_steps=1000,
    moving_start=300,
    moving_end=700,
    velocity_value=1.0,
):
    """Velocity protocol for Exercise 1.1."""

    velocity = np.zeros(total_steps, dtype=float)
    velocity[moving_start:moving_end] = velocity_value
    return velocity


def sampled_velocity_protocol(seed=7, n_velocities=10, segment_steps=300, warmup_steps=300):
    """Sample velocities from [-2, -0.5] union [0.5, 2]."""

    rng = np.random.default_rng(seed)
    signs = np.array([-1.0] * (n_velocities // 2) + [1.0] * (n_velocities - n_velocities // 2))
    rng.shuffle(signs)
    magnitudes = rng.uniform(0.5, 2.0, n_velocities)
    sampled = signs * magnitudes

    velocity = np.concatenate([np.zeros(warmup_steps), np.repeat(sampled, segment_steps)])
    starts = warmup_steps + np.arange(n_velocities) * segment_steps
    return velocity, sampled, starts


def dominant_spatial_frequency(activity, window=None):
    """Find the strongest non-zero Fourier mode in a stable activity pattern."""

    data = activity if window is None else activity[window]
    template = data.mean(axis=0)
    spectrum = np.fft.rfft(template - template.mean())
    return int(np.argmax(np.abs(spectrum[1:])) + 1)


def fourier_bump_position(activity, k=None):
    """Track bump phase as a continuous position in neuron-index units."""

    activity = np.asarray(activity, dtype=float)
    _, m = activity.shape
    if k is None:
        k = dominant_spatial_frequency(activity)

    x = np.arange(m)
    basis = np.exp(-2j * np.pi * k * x / m)
    centered = activity - activity.mean(axis=1, keepdims=True)
    coeff = centered @ basis
    phase = np.unwrap(np.angle(coeff))
    position = -phase * m / (2 * np.pi * k)
    position -= position[0]
    return position, k


def estimate_segment_velocities(
    position,
    segment_starts,
    segment_steps=300,
    discard_steps=50,
    dt_ms=1.0,
):
    """Estimate bump velocity in each segment by linear regression."""

    estimates = []
    time = np.arange(len(position)) * dt_ms
    for start in segment_starts:
        fit_start = int(start + discard_steps)
        fit_end = int(start + segment_steps)
        slope, _ = np.polyfit(time[fit_start:fit_end], position[fit_start:fit_end], deg=1)
        estimates.append(slope)
    return np.array(estimates)


def linear_fit(x, y):
    """Return slope, intercept, and R^2 for y = slope*x + intercept."""

    slope, intercept = np.polyfit(x, y, deg=1)
    prediction = slope * x + intercept
    ss_res = np.sum((y - prediction) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot
    return float(slope), float(intercept), float(r_squared)

from scipy.ndimage import convolve


def mexican_hat_2d(
    x,
    y,
    A_exc=1.0,
    sigma_exc=4.8,
    A_inh=1.0,
    sigma_inh=5.0,
):
    """2D Mexican-hat kernel for Exercise 2."""
    x = np.asarray(x)
    y = np.asarray(y)
    r2 = x**2 + y**2
    return A_exc * np.exp(-r2 / sigma_exc**2) - A_inh * np.exp(
        -r2 / sigma_inh**2
    )


@dataclass(frozen=True)
class Grid2DParams:
    """Numerical and model parameters for Exercise 2."""

    m: int = 40
    tau_ms: float = 10.0
    dt_ms: float = 1.0

    A_exc: float = 1.0
    sigma_exc: float = 4.8
    A_inh: float = 1.0
    sigma_inh: float = 5.0

    B0: float = 1.0
    alpha: float = 0.1
    weight_shift: float = 1.0


def centered_2d_coordinates(m):
    """2D coordinate grid centered around zero."""
    coords = np.arange(-m // 2, m // 2, dtype=float)
    X, Y = np.meshgrid(coords, coords, indexing="xy")
    return X, Y


def shifted_kernels_2d(params):
    """Shifted 2D kernels for North, South, East, and West populations."""
    X, Y = centered_2d_coordinates(params.m)
    l = params.weight_shift

    W_N = mexican_hat_2d(
        X,
        Y - l,
        params.A_exc,
        params.sigma_exc,
        params.A_inh,
        params.sigma_inh,
    )
    W_S = mexican_hat_2d(
        X,
        Y + l,
        params.A_exc,
        params.sigma_exc,
        params.A_inh,
        params.sigma_inh,
    )
    W_E = mexican_hat_2d(
        X - l,
        Y,
        params.A_exc,
        params.sigma_exc,
        params.A_inh,
        params.sigma_inh,
    )
    W_W = mexican_hat_2d(
        X + l,
        Y,
        params.A_exc,
        params.sigma_exc,
        params.A_inh,
        params.sigma_inh,
    )

    return W_N, W_S, W_E, W_W


def periodic_convolve2d(rate, weights):
    """Fast periodic 2D convolution using FFT with centered kernel."""
    kernel = np.fft.ifftshift(weights)
    return np.fft.ifft2(np.fft.fft2(rate) * np.fft.fft2(kernel)).real


class FourPopulationGridIntegrator:
    """Four-population 2D continuous attractor model for grid cells."""

    def __init__(self, params=None):
        self.params = params or Grid2DParams()
        self.W_N, self.W_S, self.W_E, self.W_W = shifted_kernels_2d(self.params)

    def simulate(self, velocity, seed=0, initial_state=None, store_history=True):
        """
        Run forward Euler integration for a supplied 2D velocity trace.

        velocity should have shape (n_steps, 2), where columns are vx and vy.
        """
        p = self.params
        velocity = np.asarray(velocity, dtype=float)

        if velocity.ndim != 2 or velocity.shape[1] != 2:
            raise ValueError("velocity must have shape (n_steps, 2), with columns vx and vy")

        rng = np.random.default_rng(seed)

        if initial_state is None:
            s_N = rng.uniform(0.0, 0.1, size=(p.m, p.m))
            s_S = rng.uniform(0.0, 0.1, size=(p.m, p.m))
            s_E = rng.uniform(0.0, 0.1, size=(p.m, p.m))
            s_W = rng.uniform(0.0, 0.1, size=(p.m, p.m))
        else:
            s_N, s_S, s_E, s_W = (
                np.array(arr, dtype=float, copy=True) for arr in initial_state
            )

        n_steps = len(velocity)

        if store_history:
            rate_total = np.zeros((n_steps, p.m, p.m))
        else:
            rate_total = None

        for t, (vx_t, vy_t) in enumerate(velocity):
            r_N = relu(s_N)
            r_S = relu(s_S)
            r_E = relu(s_E)
            r_W = relu(s_W)

            r_total_t = r_N + r_S + r_E + r_W

            if store_history:
                rate_total[t] = r_total_t

            recurrent = (
                periodic_convolve2d(r_N, self.W_N)
                + periodic_convolve2d(r_S, self.W_S)
                + periodic_convolve2d(r_E, self.W_E)
                + periodic_convolve2d(r_W, self.W_W)
            )

            B_N = p.B0 * (1.0 + p.alpha * vy_t)
            B_S = p.B0 * (1.0 - p.alpha * vy_t)
            B_E = p.B0 * (1.0 + p.alpha * vx_t)
            B_W = p.B0 * (1.0 - p.alpha * vx_t)

            s_N += (p.dt_ms / p.tau_ms) * (-s_N + recurrent + B_N)
            s_S += (p.dt_ms / p.tau_ms) * (-s_S + recurrent + B_S)
            s_E += (p.dt_ms / p.tau_ms) * (-s_E + recurrent + B_E)
            s_W += (p.dt_ms / p.tau_ms) * (-s_W + recurrent + B_W)

        r_N = relu(s_N)
        r_S = relu(s_S)
        r_E = relu(s_E)
        r_W = relu(s_W)
        final_rate_total = r_N + r_S + r_E + r_W

        return {
            "velocity": velocity,
            "rate_total": rate_total,
            "final_rate_total": final_rate_total,
            "final_rates": {
                "N": r_N,
                "S": r_S,
                "E": r_E,
                "W": r_W,
            },
            "final_state": (s_N, s_S, s_E, s_W),
        }