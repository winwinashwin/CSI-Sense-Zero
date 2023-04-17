from tqdm import tqdm
import pickle
import logging

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

from .rocket_functions import generate_kernels, apply_kernels

logger = logging.getLogger(__name__)


class CSIMinMaxScaler(TransformerMixin, BaseEstimator):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, *_, **__):
        return self

    def transform(self, X, y=None):
        ## Expected input shape: (N_SAMPLES, N_SUBCARRIERS, TIME_WINDOW)
        X = np.swapaxes(X, 1, 2)  ## Swap time and subcarrier axis

        n_samples, t_win, *_ = X.shape

        # compute minimum across entire data for each subcarrier
        min_vec = np.min(X, axis=(0, 1))
        min_vec = np.expand_dims(min_vec, axis=(0, 1))
        X -= np.tile(min_vec, (n_samples, t_win, 1))

        # compute maximum across entire data for each subcarrier
        max_vec = np.max(X, axis=(0, 1))
        max_vec = np.expand_dims(max_vec, axis=(0, 1))
        X /= np.tile(max_vec, (n_samples, t_win, 1))

        # restore original input shape
        X = np.swapaxes(X, 1, 2)

        return X


class Rocket(TransformerMixin, BaseEstimator):
    def __init__(self, n_kernels=10_000, batch_size=64, show_progress=True) -> None:
        super().__init__()

        self.n_kernels = n_kernels
        self.batch_sz = batch_size
        self.show_progress = show_progress

        self._kernels = None

    def fit(self, X, y=None):
        n_samples, n_sc, t_win = X.shape

        if self._kernels is None:
            self._kernels = generate_kernels(t_win, self.n_kernels)
        return self

    def transform(self, X, y=None):
        n_samples, n_sc, t_win = X.shape

        ts_sz = n_samples * n_sc  # total number of timeseries

        # Holds extracted features
        Xf = np.zeros((ts_sz, 2 * self.n_kernels))
        X = X.reshape(ts_sz, t_win)

        print("Kernel Transform")
        for i in tqdm(range(0, ts_sz, self.batch_sz), disable=not self.show_progress):
            j = i + self.batch_sz
            Xf[i:j, :] = apply_kernels(X[i:j, :], self._kernels)

        Xf = Xf.reshape(n_samples, n_sc, 2 * self.n_kernels)

        return Xf

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self._kernels, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Kernels saved at {path}")

    def load(self, path):
        logger.info(f"Loading kernels from {path}")
        with open(path, "rb") as f:
            self._kernels = pickle.load(f)
