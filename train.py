import pathlib
import logging

import scipy.io
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from HAR.transformers import CSIScaler, Rocket
from HAR.classifier import RidgeVotingClassifier


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(infile):
    mat = scipy.io.loadmat(infile)
    X = mat["csi"].T
    nsamples = mat["nsamples"].flatten()
    dim = mat["dim"].flatten()
    classnames = list(map(lambda s: s.strip().title(), mat["classnames"]))
    y = []
    for i in range(len(classnames)):
        y += [i] * nsamples[i]
    y = np.array(y)
    return X, y, nsamples, classnames, dim


class CSIHARGym:
    # Constants for activity detection
    ACTIVITY_CLASSES = ["idle", "walk", "jump"]

    # Constants for classifier
    N_KERNELS = 10_000
    N_CLASSES = len(ACTIVITY_CLASSES)

    def __init__(self, main_set, hold_set, train_size, params_dest) -> None:
        self.main_set = main_set
        self.hold_set = hold_set
        self.train_size = train_size
        self.params_dest = pathlib.Path(params_dest)
        self.params_dest.mkdir(parents=True, exist_ok=True)

        self.pipe = Pipeline(
            [
                ("scaler", CSIScaler()),
                (
                    "feature_selector",
                    Rocket(n_kernels=self.N_KERNELS, progress=True),
                ),
                (
                    "classifier",
                    RidgeVotingClassifier(n_classes=self.N_CLASSES),
                ),
            ]
        )

    def run(self):
        X, y, _, _, dim = load_dataset(self.main_set)
        X = X.reshape(X.shape[0], *dim)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=self.train_size, stratify=y
        )

        print("> Training phase")
        self._train(X_train, y_train)

        print("> Testing phase 1")
        self._test(X_test, y_test)

        X, y, _, _, dim = load_dataset(self.hold_set)
        X = X.reshape(X.shape[0], *dim)

        print("> Testing phase 2")
        self._test(X, y)

        self._dump_params()

    def _train(self, X, y):
        self.pipe.fit_transform(X, y)

    def _test(self, X, y):
        y_pred = self.pipe.predict(X)

        print(f"\n> Test Accuracy: {accuracy_score(y, y_pred)*100:.4f}%")
        print("\n> Confusion Matrix: ")
        print(confusion_matrix(y, y_pred))
        print("\n> Classification Report :")
        print(classification_report(y, y_pred, target_names=self.ACTIVITY_CLASSES))

    def _dump_params(self):
        self.pipe[1].dump_kernels(self.params_dest / "kernels.pkl")
        self.pipe[2].dump_models(self.params_dest / "models.pkl")


def main(args):
    har = CSIHARGym(args.main_set, args.hold_set, args.train_size, args.dump)
    har.run()


if __name__ == "__main__":
    import argparse

    def between_zero_and_one(value):
        fvalue = float(value)
        if fvalue < 0 or fvalue > 1:
            raise argparse.ArgumentTypeError(f"{value} is not between 0 and 1")
        return fvalue

    parser = argparse.ArgumentParser(description="CSI HAR training")

    parser.add_argument(
        "--main-set",
        help="Dataset used to train and test (phase 1)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--hold-set",
        help="Dataset used to test (phase 2)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--train-size",
        help="Fraction of main set to use for training",
        type=between_zero_and_one,
        default=0.8,
    )
    parser.add_argument(
        "--dump",
        help="Destination dir to dump trained parameters",
        type=str,
        required=True,
    )

    args = parser.parse_args()
    main(args)