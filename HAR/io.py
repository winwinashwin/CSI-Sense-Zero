import errno
import time
import os
from typing import List

import numpy as np
import scipy.io

import websockets
import asyncio


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


def read_nonblocking(path, bufferSize=100, timeout=0.100) -> List[str]:
    """
    Implementation of a non-blocking read, works with a named pipe or file.
    errno 11 occurs if pipe is still written too, wait until some data is available

    Adapted from: https://gist.github.com/conrad784/862b9d050e5018104d2e7ea900d057b8

    Args:
    path (str): The path to the file to be read.
    bufferSize (int, optional): The size of the buffer used to read the file. Defaults to 100.
    timeout (float, optional): The time to wait for data to become available in the pipe. Defaults to 0.100.

    Returns:
    list: A list of lines read from the file located at path.
    """
    result = []
    try:
        pipe = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        grace = True
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

    if pipe is not None:
        os.close(pipe)

    return result


class WebsocketBroadcastServer:
    """
    Class for a websocket server that broadcasts messages to all connected clients.

    Attributes:
        CLIENTS (set): A set of all currently connected clients.
    """

    CLIENTS = set()

    def __init__(self, host, port, message_generator, broadcast_frequency) -> None:
        """
        Initializes a new instance of the WebsocketBroadcastServer class.

        Args:
            host (str): The IP address or hostname that the server will bind to.
            port (int): The port that the server will listen on.
            message_generator (callable): A function that generates messages to broadcast.
            broadcast_frequency (float): The frequency at which to broadcast messages, in Hz.
        """
        self.host = host
        self.port = port
        self.message_generator = message_generator
        self.broadcast_frequency = broadcast_frequency

    async def run(self):
        """
        Starts the websocket server and broadcast loop.
        """
        async with websockets.serve(self._client_handler, self.host, self.port):
            await self._broadcast_loop()

    async def _client_handler(self, websocket):
        """
        Handler for new client connections.

        Args:
            websocket: The websocket connection object.
        """
        self.CLIENTS.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.CLIENTS.remove(websocket)

    async def _send(self, websocket, message):
        """
        Sends a message to a single client.

        Args:
            websocket: The websocket connection object.
            message: The message to send.
        """
        try:
            await websocket.send(message)
        except websockets.ConnectionClosed:
            pass

    async def _broadcast(self, message):
        """
        Broadcasts a message to all connected clients.

        Args:
            message: The message to broadcast.
        """
        if message is None:
            return
        for websocket in self.CLIENTS:
            asyncio.create_task(self._send(websocket, message))

    async def _broadcast_loop(self):
        """
        The main loop for broadcasting messages.
        """
        while True:
            await asyncio.gather(
                self._broadcast(self.message_generator()),
                asyncio.sleep(1.0 / self.broadcast_frequency),  # throttle
            )
