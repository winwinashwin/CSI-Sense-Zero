from sklearn.pipeline import Pipeline
from datetime import datetime
from io import StringIO
import pandas as pd
import numpy as np
import logging
import json
import asyncio

from HAR.constants import CSI_COL_NAMES, NULL_SUBCARRIERS
from HAR.classifier import RidgeVotingClassifier, ActivityIndicatorClassifier
from HAR.io import read_nonblocking, WebsocketBroadcastServer
from HAR.transformers import CSIScaler, Rocket

CSIFIFO = "/tmp/csififo"
WINSIZE = 256
KM_VERSION = 2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_next_sample():
    buf = read_nonblocking(CSIFIFO, 235_000, 0)

    firstvalid = next(i for i, v in enumerate(buf) if v.startswith("CSI_DATA"))

    try:
        df = pd.read_csv(
            StringIO("\n".join(buf[firstvalid : firstvalid + WINSIZE])),
            header=None,
            names=CSI_COL_NAMES,
            on_bad_lines="skip",
        )

        df = df.loc[df["sig_mode"] == 1]
        csi_data = np.array(
            [
                np.fromstring(csi_record.strip("[]"), dtype=int, sep=",")
                for csi_record in df["data"].copy()
            ]
        )

        csi_data = np.delete(csi_data.T, NULL_SUBCARRIERS, 0).T
        csi_amp = np.array(
            [np.sqrt(data[::2] ** 2 + data[1::2] ** 2) for data in csi_data]
        ).T

        diff = WINSIZE - csi_amp.shape[1]

        if diff > 0:
            logger.warning(
                "Bad lines detected during CSI parse. Broadcasting last column for compatibility"
            )
            csi_amp = np.hstack(
                (
                    csi_amp,
                    np.broadcast_to(csi_amp[:, -1][:, None], (csi_amp.shape[0], diff)),
                )
            )

        return csi_amp
    except Exception as e:
        logger.error("Error during CSI parse", exc_info=True)
        return None


activity_detection_pipeline = Pipeline(
    [
        ("scaler", CSIScaler()),
        ("classifier", ActivityIndicatorClassifier(threshold=0.3)),
    ]
)

activity_classification_pipeline = Pipeline(
    [
        ("scaler", CSIScaler()),
        (
            "feature_selector",
            Rocket(n_kernels=10_000, progress=False).load_kernels(
                f"artifacts/{KM_VERSION}/kernel.pkl"
            ),
        ),
        (
            "classifier",
            RidgeVotingClassifier(n_classes=3).load_models(
                f"artifacts/{KM_VERSION}/models.pkl"
            ),
        ),
    ]
)


"""## matplotlib live CSI animation

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
sns.set_theme()

fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(1, 1, 1)
ax1.title.set_text("CSI LLTF Live")

scaler = CSIScaler()

def animate(i):
    csi = get_next_sample()
    if csi is None:
        return
    X = scaler.fit_transform(csi.reshape(1, *csi.shape))

    U, _, _ = np.linalg.svd(X[0, :, :])
    s = np.dot(U[:, 1], X[0, :, :])


    # print(csi.min(), csi.max())
    ax1.clear()
    # ax1.imshow(csi, aspect="auto", cmap="hsv", vmin=0, vmax=100)
    ax1.plot(s)
    ax1.set_ylim((-1, 1))
    ax1.grid(0)


ani = animation.FuncAnimation(fig, animate, interval=100, cache_frame_data=False)
plt.show()

exit(0)
"""

msg = {
    "timestamp": datetime.utcnow().isoformat(),
    "hypothesis": 0,
    "classnames": ["idle", "walk", "jump"],
}


def make_prediction():
    X = get_next_sample()
    if X is None:
        return None

    msg["timestamp"] = datetime.utcnow().isoformat()

    X = X.reshape(1, *X.shape)

    h = int(activity_detection_pipeline.predict(X)[0])
    if h:
        h = int(activity_classification_pipeline.predict(X)[0])

    msg["hypothesis"] = h
    return json.dumps(msg)


if __name__ == "__main__":
    server = WebsocketBroadcastServer(
        host="localhost",
        port=9999,
        message_generator=make_prediction,
        broadcast_frequency=2,
    )
    asyncio.run(server.run())
