from sklearn.base import ClassifierMixin, BaseEstimator
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import pickle
import logging
from sklearn.linear_model import RidgeClassifierCV

logger = logging.getLogger(__name__)


class PC2VarBinaryClassifer(ClassifierMixin, BaseEstimator):
    def __init__(self, threshold) -> None:
        super().__init__()

        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def predict(self, X):
        # X.shape = (n_samples, n_subcarriers, time_win)
        # SVD in "stacked" mode, SVD is applied to each combination of last two indices
        U, _, _ = np.linalg.svd(X)

        # 1. Noise induced by internal state changes are present in CSI streams. Due to high correlation,
        # these noises are captured in the first principal component along with movement signal.
        # 2. Phase of a subcarrier is a linear combination of two orthogonal components (sine and cosine).
        # PCA components are uncorrelated, first princ. component contains only one of these orthogonal
        # components and the other is retained in the rest. Hence we can safely discard PC1 and focus on PC2.

        # For each sample, take projection along second principal component
        # https://ajcr.net/Basic-guide-to-einsum/
        pc2 = np.einsum("ijk,ij->ik", X, U[:, :, 1])
        variances = np.var(pc2, axis=1)

        clf = variances > self.threshold

        return clf


class RidgeVotingClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_classes, show_progress=True) -> None:
        super().__init__()

        self.n_classes = n_classes
        self.show_progress = show_progress

        self._models = None

    def fit(self, X, y):
        n_samples, n_sc, t_win = X.shape

        # Train models
        if self._models is None:
            self._models = Parallel(n_jobs=-2, backend="threading")(
                delayed(self._train_clf)(X[:, m_, :], y)
                for m_ in tqdm(range(n_sc), disable=not self.show_progress)
            )
        return self

    def transform(self, X):
        return X

    def predict(self, X):
        n_samples, n_sc, *_ = X.shape
        final_predictions = np.zeros((n_samples,))
        for isample in range(n_samples):
            # Get predictions from each model followed by majority voting
            predictions = Parallel(n_jobs=1, backend="threading")(
                delayed(self._score)(
                    self._models[m_], np.expand_dims(X[isample, m_, :], axis=0)
                )
                for m_ in range(n_sc)
            )

            unique, counts = np.unique(predictions, return_counts=True)

            final_predictions[isample] = unique[np.argmax(counts)]
        return final_predictions

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self._models, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Models saved at {path}")

    def load(self, path):
        logger.info(f"Loading models from {path}")
        with open(path, "rb") as f:
            self._models = pickle.load(f)

    def _train_clf(self, X, y):
        model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        model.fit(X, y)
        return model

    def _score(self, model, X):
        p = model.predict(X)
        return p
