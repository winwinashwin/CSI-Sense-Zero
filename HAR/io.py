import time
import os
import errno
import websockets
import asyncio


def read_nonblocking(path, bufferSize=100, timeout=0.100):
    """
    Implementation of a non-blocking read, works with a named pipe or file

    errno 11 occurs if pipe is still written too, wait until some data is available
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

    if pipe is not None:
        os.close(pipe)
    return result


class WebsocketBroadcastServer:
    CLIENTS = set()

    def __init__(self, host, port, message_generator, broadcast_frequency) -> None:
        self.host = host
        self.port = port
        self.message_generator = message_generator
        self.broadcast_frequency = broadcast_frequency

    async def run(self):
        async with websockets.serve(self._client_handler, self.host, self.port):
            await self._broadcast_loop()

    async def _client_handler(self, websocket):
        self.CLIENTS.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.CLIENTS.remove(websocket)

    async def _send(self, websocket, message):
        try:
            await websocket.send(message)
        except websockets.ConnectionClosed:
            pass

    async def _broadcast(self, message):
        if message is None:
            return
        for websocket in self.CLIENTS:
            asyncio.create_task(self._send(websocket, message))

    async def _broadcast_loop(self):
        while True:
            await asyncio.gather(
                self._broadcast(self.message_generator()),
                asyncio.sleep(1.0 / self.broadcast_frequency),  # throttle
            )
