"""Small numerical helpers used across the project."""
import numpy as np


def zscore(x, axis=None, eps=1e-8):
    """Standardize values to zero mean / unit variance along an axis."""
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True) + eps
    return (x - mu) / sd


def softmax(x):
    """Numerically stable softmax over a 1-D array."""
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()
