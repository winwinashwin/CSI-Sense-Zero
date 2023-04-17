import asyncio
import json
import logging
from datetime import datetime
from io import StringIO
from typing import Optional

import numpy as np
import pandas as pd

from HAR import (
    CSIMinMaxScaler,
    CSIActivityRecognitionPipeline,
    CSIActivityIndicatorPipeline,
)
from HAR.constants import CSI_COL_NAMES, NULL_SUBCARRIERS
from HAR.io import WebsocketBroadcastServer, read_nonblocking


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CSIHAR:
    CSIFIFO = "/tmp/csififo"
    WINSIZE = 256

    # Constants for activity detection
    ACTIVITY_THRESHOLD = 0.3
    ACTIVITY_CLASSES = ["idle", "walk", "jump"]

    # Constants for classifier
    N_KERNELS = 10_000
    N_CLASSES = len(ACTIVITY_CLASSES)

    def __init__(self, params_dir) -> None:
        self.params_dir = params_dir

        self.scaler = CSIMinMaxScaler()
        self.activity_detection = CSIActivityIndicatorPipeline(
            self.ACTIVITY_THRESHOLD, normalize_input=False
        )
        self.activity_recognition = CSIActivityRecognitionPipeline(
            self.N_CLASSES, self.N_KERNELS, normalize_input=False, show_progress=False
        ).load(self.params_dir)

        self.msg = {
            "timestamp": None,
            "hypothesis": None,
            "classnames": self.ACTIVITY_CLASSES,
        }

    def make_prediction(self) -> Optional[str]:
        """
        Generates a prediction based on the next CSI sample from the CSIFIFO, using a pipeline of transformers and classifiers.
        Returns a JSON string containing the timestamp and predicted activity, or None if the sample could not be retrieved.

        Returns:
            A JSON string containing the timestamp and predicted activity, or None.
        """
        X = self.get_next_sample()
        if X is None:
            return None

        self.msg["timestamp"] = datetime.utcnow().isoformat()

        X = X.reshape(1, *X.shape)
        X = self.scaler.fit_transform(X)

        h = int(self.activity_detection.predict(X)[0])
        if h:
            h = int(self.activity_recognition.predict(X)[0])

        self.msg["hypothesis"] = h
        return json.dumps(self.msg)

    def get_next_sample(self) -> np.ndarray:
        """
        Reads the next sample from the CSI FIFO and processes it to extract
        the CSI amplitudes.

        Returns:
            np.ndarray: A 2D array of shape (n_antennas, n_subcarriers) containing
            the CSI amplitudes.
        """
        buf = read_nonblocking(self.CSIFIFO, 235_000, 0)

        firstvalid = next(i for i, v in enumerate(buf) if v.startswith("CSI_DATA"))

        try:
            df = pd.read_csv(
                StringIO("\n".join(buf[firstvalid : firstvalid + self.WINSIZE])),
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

            diff = self.WINSIZE - csi_amp.shape[1]

            if diff > 0:
                logger.warning(
                    "Bad lines detected during CSI parse. Broadcasting last column for compatibility"
                )
                csi_amp = np.hstack(
                    (
                        csi_amp,
                        np.broadcast_to(
                            csi_amp[:, -1][:, None], (csi_amp.shape[0], diff)
                        ),
                    )
                )

            return csi_amp
        except Exception as e:
            logger.error("Error during CSI parse", exc_info=True)
            return None


def main(args):
    har = CSIHAR(args.load)

    server = WebsocketBroadcastServer(
        host=args.host,
        port=args.port,
        message_generator=har.make_prediction,
        broadcast_frequency=args.frequency,
    )

    asyncio.run(server.run())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Perform realtime HAR using Wi-Fi CSI and broadcast predictions via websockets"
    )

    parser.add_argument(
        "--load", help="Path to model parameters", type=str, required=True
    )

    parser.add_argument(
        "--host",
        help="IP or hostname server will bind to",
        type=str,
        default="localhost",
    )
    parser.add_argument(
        "--port", help="Port server will listen on", type=int, default=9999
    )
    parser.add_argument(
        "--frequency",
        help="Message broadcast frequency (Hz)",
        type=float,
        default=2.0,
    )

    args = parser.parse_args()
    main(args)
