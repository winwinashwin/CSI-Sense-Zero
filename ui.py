"""
This script creates a simple GUI to visualize HAR predictions received over a WebSocket connection.
class name in a large font on a black background. The prediction is expected as a JSON
message over the WebSocket connection, and is extracted and displayed on the GUI.


Command line arguments:
    --host: Host address of the WebSocket server. Default is "localhost".
    --port: Port number of the WebSocket server. Default is 9999.
"""

import sys
import asyncio
import websockets
import json
import logging

from PyQt5.QtCore import Qt, QObject, pyqtSignal, pyqtSlot, QThread
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtGui import QFont

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HAR-GUI")


class WebSocketClient(QObject):
    message_received = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.ws_url = None

    async def connect(self):
        async with websockets.connect(self.ws_url) as websocket:
            async for message in websocket:
                self.message_received.emit(message)


class WebSocketThread(QThread):
    def __init__(self):
        super().__init__()

        self.client = WebSocketClient()

    def run(self):
        asyncio.run(self.client.connect())


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CSI Sense Zero")
        self.label = QLabel(self)
        font = QFont("Arial", 200, QFont.Bold)
        self.label.setFont(font)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet(
            "color: #E50914; background-color: #000000; border: none;"
        )

        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.setStyleSheet("background-color: #000000;")

        self.thread = WebSocketThread()
        self.thread.client.message_received.connect(self.on_message_received)
        self.thread.start()

    @pyqtSlot(str)
    def on_message_received(self, message):
        msg = json.loads(message)

        data = msg["classnames"][msg["hypothesis"]].title()
        self.label.setText(data)


def main(args):
    ws_url = f"ws://{args.host}:{args.port}"
    logger.info(f"Connecting to server {ws_url}")

    app = QApplication(sys.argv)
    window = MainWindow()
    window.thread.client.ws_url = ws_url
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple GUI to visualise predictions")

    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9999)

    args = parser.parse_args()
    main(args)
