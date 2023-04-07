import scipy.io
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from transformers import CSIScaler, Rocket
from classifier import RidgeVotingClassifier

pipe = Pipeline(
    [
        ("scaler", CSIScaler()),
        (
            "feature_selector",
            # Rocket(n_kernels=1_000).load_kernels("blobs/kernels.pkl"),
            Rocket(n_kernels=1_000),
        ),
        (
            "classifier",
            # RidgeVotingClassifier(n_classes=3).load_models("blobs/model.pkl"),
            RidgeVotingClassifier(n_classes=3),
        ),
    ]
)


mat = scipy.io.loadmat("../dataset/rCSI-300")

X = mat["csi"].T
nsamples = mat["nsamples"].flatten()
dim = mat["dim"].flatten()
classnames = list(map(lambda s: s.strip().title(), mat["classnames"]))
y = []
for i in range(len(classnames)):
    y += [i] * nsamples[i]
y = np.array(y, dtype=int)

X = X.reshape(X.shape[0], *dim)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y)

pipe.fit_transform(X_train, y_train)

pipe[1].dump_kernels("blobs/kernel.pkl")
pipe[2].dump_models("blobs/models.pkl")

y_pred = pipe.predict(X_test)

print(f"\n> Test Accuracy: {accuracy_score(y_test, y_pred)*100:.4f}%")
print("\n> Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))
print("\n> Classification Report :")
print(classification_report(y_test, y_pred, target_names=classnames))


mat = scipy.io.loadmat("../dataset/rCSI-100")

X = mat["csi"].T
nsamples = mat["nsamples"].flatten()
dim = mat["dim"].flatten()
classnames = list(map(lambda s: s.strip().title(), mat["classnames"]))
y = []
for i in range(len(classnames)):
    y += [i] * nsamples[i]
y = np.array(y, dtype=int)

X = X.reshape(X.shape[0], *dim)

y_pred = pipe.predict(X)

print(f"\n> Test Accuracy: {accuracy_score(y, y_pred)*100:.4f}%")
print("\n> Confusion Matrix: ")
print(confusion_matrix(y, y_pred))
print("\n> Classification Report :")
print(classification_report(y, y_pred, target_names=classnames))
