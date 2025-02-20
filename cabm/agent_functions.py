# cabm/agent_functions.py

import math
import warnings
import random
import numpy as np


def sample_normal_min(
    mean: float, std_dev: float = 1.0, min_value: float = 1.0, override: float = 0
) -> float:
    """
    Sample from a normal distribution, ignoring values < min_value.
    If override != 0, return override directly.
    """
    if override != 0:
        warnings.warn("Normal Sampler Override is in effect.")
        return override

    sample = np.random.normal(mean, std_dev)
    while sample < min_value:
        sample = np.random.normal(mean, std_dev)
    return sample


def sample_beta_min(
    alpha: float, beta: float, min_value: float = 0.05, override: float = 0
) -> float:
    """
    Sample from a beta distribution, ignoring values < min_value.
    If override != 0, return override.
    """
    if override != 0:
        warnings.warn("Beta Sampler Override is in effect.")
        return override

    sample = np.random.beta(alpha, beta)
    while abs(sample) < min_value:
        sample = np.random.beta(alpha, beta)
    return sample


def get_pantry_max(household_size: int, pantry_min: int) -> int:
    """
    Statistically assigns the maximum number of products a given household stocks.
    We do a normal distribution with mean=household_size, stdev=1, ensure >= pantry_min
    """
    pantry_max = math.ceil(np.random.normal(household_size, 1.0))
    if pantry_max < pantry_min:
        pantry_max = pantry_min
    return pantry_max


def assign_media_channel_weights(channels: list, prior_weights: list) -> dict:
    """
    Assign random channel weights around a prior distribution, e.g. [0.7, 0.3].
    This is used to produce agent-specific slight preference for, say, 'Web' vs. 'TV'.
    """
    # small random fluctuation
    fluctuations = [(random.random() - 0.5) * 0.2 for _ in channels]
    weights = []
    for w, f in zip(prior_weights, fluctuations):
        w_ = w * (1 + f)
        w_ = max(0.001, w_)
        weights.append(w_)

    s = sum(weights)
    if s == 0:
        # fallback
        n = len(channels)
        return {c: 1 / n for c in channels}

    normalized = [w / s for w in weights]
    return dict(zip(channels, normalized))
