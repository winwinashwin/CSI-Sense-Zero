from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
)
from sklearn.linear_model import RidgeClassifierCV
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
import pickle


class RidgeVotingClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_classes) -> None:
        super().__init__()

        self.n_classes = n_classes

        self._models = None

    def fit(self, X, y):
        n_samples, n_sc, t_max = X.shape

        if self._models is None:
            print("=== Classifer Training ===")
            self._models = Parallel(n_jobs=-2, backend="threading")(
                delayed(self._train_clf)(X[:, m_, :], y) for m_ in tqdm(range(n_sc))
            )
        return self

    def transform(self, X):
        return X

    def predict(self, X):
        n_samples, n_sc, *_ = X.shape
        final_predictions = []
        for isample in range(n_samples):
            predictions = Parallel(n_jobs=1, backend="threading")(
                delayed(self._score)(
                    self._models[m_], np.expand_dims(X[isample, m_, :], axis=0)
                )
                for m_ in range(n_sc)
            )

            unique, counts = np.unique(predictions, return_counts=True)

            final_predictions.append(unique[np.argmax(counts)])
        return np.array(final_predictions)

    def dump_models(self, outfile):
        with open(outfile, "wb") as f:
            pickle.dump(self._models, f, protocol=4)

        print(f"Models saved at {outfile}")

    def load_models(self, infile):
        print(f"Loading models from {infile}")
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
