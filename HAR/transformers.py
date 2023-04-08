from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)
import numpy as np
from tqdm import tqdm
import pickle

from .rocket_functions import apply_kernels, generate_kernels


class CSIScaler(TransformerMixin, BaseEstimator):
    def __init__(self) -> None:
        super().__init__()
        self._mean = None

    def fit(self, *_, **__):
        return self

    def transform(self, X):
        X = np.swapaxes(X, 1, 2)

        n_samples, t_max, *_ = X.shape

        min_vec = np.min(X, axis=(0, 1))
        min_vec = np.expand_dims(min_vec, axis=(0, 1))
        X -= np.tile(min_vec, (n_samples, t_max, 1))
        max_vec = np.max(X, axis=(0, 1))
        max_vec = np.expand_dims(max_vec, axis=(0, 1))
        X /= np.tile(max_vec, (n_samples, t_max, 1))

        X = np.swapaxes(X, 1, 2)

        return X


class Rocket(TransformerMixin, BaseEstimator):
    def __init__(self, n_kernels=10_000) -> None:
        super().__init__()

        self.n_kernels = n_kernels
        self._kernels = None

    def fit(self, X, y=None):
        n_samples, n_sc, t_max = X.shape

        if self._kernels is None:
            print("=== Generating Kernels ===")
            self._kernels = generate_kernels(t_max, self.n_kernels)
        return self

    def transform(self, X):
        n_samples, n_sc, t_max = X.shape

        print("=== Kernel Transform ===")
        Xr = np.zeros((X.shape[0], n_sc, 2 * self.n_kernels))
        for isample in tqdm(range(n_samples)):
            Xr[isample, :, :] = apply_kernels(X[isample, :, :], self._kernels)

        return Xr

    def dump_kernels(self, outfile):
        with open(outfile, "wb") as f:
            pickle.dump(self._kernels, f, protocol=4)
        print(f"Kernels saved at {outfile}")

    def load_kernels(self, infile):
        print(f"Loading kernels from {infile}")
        with open(infile, "rb") as f:
            self._kernels = pickle.load(f)

        return self
