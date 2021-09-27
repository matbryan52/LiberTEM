import time
import zmq
import pathlib


def split_ds_slice(ds_slice):
    values = []
    for a in ds_slice:
        if isinstance(a, int):
            values.append(a)
        else:
            values.append((a.start, a.stop, a.step))
    return values


class SocketStreamer(object):
    def __init__(self, ctx=None):
        if ctx is None:
            ctx = zmq.Context.instance()
        self.ctx = ctx
        self.socket = self.ctx.socket(zmq.PUB)

    def connect(self, address='127.0.0.1', port=5555, protocol='tcp', delay=0.5):
        retval = self.socket.bind(f'{protocol}://{address}:{port}')
        if delay > 0:
            time.sleep(delay)
        return retval

    def send_from_dataset(self, data_reader, flags=0):
        for ds_slice, frame in data_reader.generate_tiles():

            metadata = {
                'state': True,
                'dtype': frame.dtype.name,
                'shape': frame.shape,
                'ds_slice': split_ds_slice(ds_slice)
            }

            self.socket.send_json(metadata, flags=flags | zmq.SNDMORE)
            self.socket.send(frame, flags=flags)

        self.socket.send_json({'state': False}, flags=flags)

    def close(self):
        self.socket.close()

from dataset_reader import DatasetReader
def stream(ctx, start_frame, port, nframes=1000):
    rawfile = pathlib.Path('../../ray_testing/random.raw')
    sig_shape = (256, 256)
    p_slice = slice(start_frame, start_frame+nframes)
    reader = DatasetReader(rawfile, sig_shape, partition_slice=p_slice)
    reader.open()

    streamer = SocketStreamer(ctx=ctx)
    streamer.connect(port=port)
    streamer.send_from_dataset(reader)
    streamer.close()


if __name__ == '__main__':
    from concurrent.futures import ThreadPoolExecutor
    with zmq.Context.instance() as ctx, ThreadPoolExecutor() as p:
        for start_frame, port in zip([1000, 3000, 5000], [5555, 5556, 5557]):
            p.submit(stream, ctx, start_frame, port)


    # from dataset_reader import DatasetReader
    # rawfile = pathlib.Path('../../ray_testing/random.raw')
    # sig_shape = (256, 256)
    # p_slice = slice(1000, 2000)
    # reader = DatasetReader(rawfile, sig_shape, partition_slice=p_slice)
    # reader.open()

    # with zmq.Context.instance() as ctx:
    #     streamer = SocketStreamer(ctx=ctx)
    #     streamer.connect(port=5556)
    #     streamer.send_from_dataset(reader)
    #     streamer.close()