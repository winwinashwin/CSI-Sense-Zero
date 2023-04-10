import sys
import asyncio
import websockets
from PyQt5.QtCore import Qt, QObject, pyqtSignal, pyqtSlot, QThread
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtGui import QFont
import json


class WebSocketClient(QObject):
    message_received = pyqtSignal(str)

    async def connect(self, url):
        async with websockets.connect(url) as websocket:
            async for message in websocket:
                self.message_received.emit(message)


class WebSocketThread(QThread):
    def __init__(self):
        super().__init__()

        self.client = WebSocketClient()

    def run(self):
        asyncio.run(self.client.connect("ws://localhost:9999"))


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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
