from sklearn.pipeline import Pipeline
import pathlib
from .transformers import CSIMinMaxScaler, Rocket
from .classifiers import RidgeVotingClassifier, PC2VarBinaryClassifer


class CSIActivityIndicatorPipeline(Pipeline):
    def __init__(
        self,
        threshold,
        normalize_input=True,
        memory=None,
        verbose=False,
    ):
        self.threshold = threshold
        self.normalize_input = normalize_input
        self.memory = memory
        self.verbose = verbose

        self.steps = []

        if self.normalize_input:
            self.steps += [("scaler", CSIMinMaxScaler())]

        self.steps += [("clf", PC2VarBinaryClassifer(threshold=self.threshold))]

        self._validate_steps()


class CSIActivityRecognitionPipeline(Pipeline):
    def __init__(
        self,
        n_classes,
        n_kernels,
        batch_size=64,
        normalize_input=True,
        show_progress=True,
        memory=None,
        verbose=False,
    ):
        self.n_classes = n_classes
        self.n_kernels = n_kernels
        self.batch_size = batch_size
        self.normalize_input = normalize_input
        self.show_progress = show_progress
        self.memory = memory
        self.verbose = verbose

        self.steps = []

        if self.normalize_input:
            self.steps += [("scaler", CSIMinMaxScaler())]

        self.steps += [
            (
                "rocket",
                Rocket(
                    n_kernels=n_kernels,
                    batch_size=self.batch_size,
                    show_progress=self.show_progress,
                ),
            ),
            (
                "clf",
                RidgeVotingClassifier(
                    n_classes=n_classes, show_progress=self.show_progress
                ),
            ),
        ]

        self._validate_steps()

    def save(self, path):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self["rocket"].save(path / "kernels.pkl")
        self["clf"].save(path / "models.pkl")

    def load(self, path):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        self["rocket"].load(path / "kernels.pkl")
        self["clf"].load(path / "models.pkl")

        return self
