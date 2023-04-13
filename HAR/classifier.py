from joblib import Parallel, delayed
from tqdm import tqdm
import logging
import pickle

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import RidgeClassifierCV


logger = logging.getLogger(__name__)


class ActivityIndicatorClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, threshold) -> None:
        super().__init__()

        self.threshold = threshold

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X

    def predict(self, X):
        n_samples, n_sc, *_ = X.shape

        final_predictions = np.zeros((n_samples,))

        # For each sample, take the second principal component and compute its
        # variance. Activity detected if variance is greater than threshold.
        for isample in range(n_samples):
            U, _, _ = np.linalg.svd(X[isample, :, :])

            var = np.var(np.dot(U[:, 1], X[isample, :, :]))

            final_predictions[isample] = var > self.threshold

        return final_predictions


class RidgeVotingClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_classes) -> None:
        super().__init__()

        self.n_classes = n_classes

        self._models = None

    def fit(self, X, y):
        n_samples, n_sc, t_max = X.shape

        # Train models
        if self._models is None:
            self._models = Parallel(n_jobs=-2, backend="threading")(
                delayed(self._train_clf)(X[:, m_, :], y) for m_ in tqdm(range(n_sc))
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

    def dump_models(self, outfile):
        with open(outfile, "wb") as f:
            pickle.dump(self._models, f, protocol=4)

        logger.info(f"Models saved at {outfile}")

    def load_models(self, infile):
        logger.info(f"Loading models from {infile}")
        with open(infile, "rb") as f:
            self._models = pickle.load(f)

        return self

    def _train_clf(self, X, y):
        model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        model.fit(X, y)
        return model

    def _score(self, model, X):
        p = model.predict(X)
        return p
