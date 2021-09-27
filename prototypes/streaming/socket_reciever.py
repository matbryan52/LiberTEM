import zmq
import numpy as np


def reconstruct_ds_slice(ds_slice):
    values = []
    for a in ds_slice:
        if isinstance(a, int):
            values.append(a)
        else:
            values.append(np.s_[a[0]:a[1]:a[2]])
    return tuple(values)


class SocketReciever(object):
    def __init__(self, ctx=None):
        if ctx is None:
            ctx = zmq.Context.instance()
        self.ctx = ctx
        self.n_sub = 0
        self.socket = self.ctx.socket(zmq.SUB)

    def connect(self, address='127.0.0.1', port=5555, protocol='tcp', filter=b''):
        connection = self.socket.connect(f'{protocol}://{address}:{port}')
        self.socket.setsockopt(zmq.SUBSCRIBE, filter)
        self.n_sub += 1
        return connection

    def multi_connect(self, ports, **kwargs):
        for port in ports:
            self.connect(port=port, **kwargs)

    def recieve(self, flags=0):
        metadata = self.socket.recv_json(flags=flags)
        if not metadata['state']:
            return False
        msg = self.socket.recv(flags=flags)
        buf = memoryview(msg)
        frame_flat = np.frombuffer(buf, dtype=metadata['dtype'])
        ds_slice = reconstruct_ds_slice(metadata['ds_slice'])
        return ds_slice, frame_flat.reshape(metadata['shape'])

    def poll(self):
        n_finished = 0
        while True:
            _ = self.socket.poll()
            retval = self.recieve()
            if not retval:
                n_finished += 1
                if n_finished >= self.n_sub:
                    break
            else:
                yield retval

    def close(self):
        self.socket.close()


if __name__ == '__main__':
    recvd = 0
    with zmq.Context.instance() as ctx:
        reciever = SocketReciever(ctx=ctx)
        reciever.multi_connect([5555, 5556, 5557])

        for ds_slice, frame in reciever.poll():
            print(ds_slice, frame.shape, frame.dtype)
            recvd += 1

        reciever.close()

    print(f'Recieved {recvd}')
