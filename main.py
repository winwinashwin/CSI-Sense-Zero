from websocket_server import WebsocketServer
from sklearn.pipeline import Pipeline
from datetime import datetime
from io import StringIO
import pandas as pd
import numpy as np
import threading
import logging
import errno
import json
import time
import os

from HAR.transformers import CSIScaler, Rocket
from HAR.classifier import RidgeVotingClassifier

CSI_COL_NAMES = [
    "type",
    "id",
    "mac",
    "rssi",
    "rate",
    "sig_mode",
    "mcs",
    "bandwidth",
    "smoothing",
    "not_sounding",
    "aggregation",
    "stbc",
    "fec_coding",
    "sgi",
    "noise_floor",
    "ampdu_cnt",
    "channel",
    "secondary_channel",
    "local_timestamp",
    "ant",
    "sig_len",
    "rx_state",
    "len",
    "first_word",
    "data",
]
NULL_SUBCARRIERS = list(range(0, 10)) + list(range(118, 256))

CSIFIFO = "/tmp/csififo"
WINSIZE = 256
KM_VERSION = 1


logger = logging.getLogger(__name__)


def read_nonblocking(path, bufferSize=100, timeout=0.100):
    """
    Implementation of a non-blocking read
    works with a named pipe or file

    errno 11 occurs if pipe is still written too, wait until some data
    is available
    """
    grace = True
    result = []
    try:
        pipe = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        while True:
            try:
                buf = os.read(pipe, bufferSize)
                if not buf:
                    break
                else:
                    content = buf.decode("utf-8")
                    line = content.split("\n")
                    result.extend(line)
            except OSError as e:
                if e.errno == 11 and grace:
                    # grace period, first write to pipe might take some time
                    # further reads after opening the file are then successful
                    time.sleep(timeout)
                    grace = False
                else:
                    break

    except OSError as e:
        if e.errno == errno.ENOENT:
            pipe = None
        else:
            raise e

    return result


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
            csi_amp = np.hstack(
                (
                    csi_amp,
                    np.broadcast_to(csi_amp[:, -1][:, None], (csi_amp.shape[0], diff)),
                )
            )

        return csi_amp
    except Exception as e:
        print(e)
        return None


pipe = Pipeline(
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


def animate(i):
    csi = get_next_sample()
    if csi is None:
        return
    # print(csi.min(), csi.max())
    ax1.clear()
    ax1.imshow(csi, aspect="auto", cmap="hsv", vmin=0, vmax=100)
    ax1.grid(0)


ani = animation.FuncAnimation(fig, animate, interval=100, cache_frame_data=False)
plt.show()
"""

if __name__ == "__main__":
    server = WebsocketServer("0.0.0.0", 9999, logging.INFO)

    t = threading.Thread(target=server.run_forever, daemon=True)
    t.start()

    msg = {
        "timestamp": datetime.utcnow().isoformat(),
        "hypothesis": 100,
        "classnames": ["idle", "walk", "jump"],
    }
    try:
        while True:
            X = get_next_sample()
            X = X.reshape(1, *X.shape)
            y = pipe.predict(X)

            msg["timestamp"] = datetime.utcnow().isoformat()
            msg["hypothesis"] = int(y[0])

            logger.info("Broadcasting hypothesis to clients")
            server.send_message_to_all(json.dumps(msg))

    except KeyboardInterrupt:
        pass
