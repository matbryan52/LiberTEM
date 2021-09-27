import time
import warnings
import numpy as np
import pyarrow.plasma as plasma
import pyarrow as pa
import zmq
import asyncio
from zmq.asyncio import Context

from tracing import Tracer


class StreamingSender(object):
    def __init__(self, endpoint_dict, tracer=None, ctx=None):
        self.ctx = ctx
        self.own_context = ctx is None
        self.endpoint_dict = endpoint_dict
        self.sockets = {}
        self.tracer = tracer

    @property
    def control_port(self):
        return int(self.endpoint_dict['control'].split(':')[-1])

    @property
    def context(self):
        if self.ctx is None:
            self.ctx = zmq.Context.instance()
        else:
            return self.ctx

    @property
    def control_socket(self):
        try:
            return self.sockets['control']
        except KeyError:
            self.sockets['control'] = self.context.socket(zmq.REP)
            return self.control_socket

    @property
    def notify_socket(self):
        try:
            return self.sockets['notify']
        except KeyError:
            self.sockets['notify'] = self.context.socket(zmq.PUB)
            return self.notify_socket

    def connect(self, endpoint_dict):
        self.control_socket.connect(endpoint_dict['control'])
        self.notify_socket.bind(endpoint_dict['notify'])

    def disconnect(self):
        for socket in self.sockets.values():
            if not socket.closed:
                socket.close()

    def notify(self, message_dict, flags=0):
        self.notify_socket.send_json(message_dict, flags=flags)

    def __enter__(self):
        if self.endpoint_dict is not None:
            self.connect(self.endpoint_dict)
        return self

    def __exit__(self, *args):
        self.disconnect()
        if self.own_context:
            if not self.context.closed:
                self.context.close()

    def run(self):
        while True:
            _ = self.control_socket.poll()
            msg = self.control_socket.recv_json()
            if msg is None:
                raise
            self.control_socket.send_json({'recieved': True})
            if msg.get('start', False):
                break

        print(f'Worker start: {self.control_port}')
        time.sleep(0.1)

        while True:
            message_dict = random_msg(self.control_port)
            self.notify(message_dict)

            poll_res = self.control_socket.poll(timeout=1)
            if poll_res:
                msg = self.control_socket.recv_json()
                self.control_socket.send_json({'recieved': True})
                if msg.get('stop', False):
                    break
        print(f'Worker stopped: {self.control_port}')

    def interpret_control(self, message):
        tasks = []
        if 'start' in message:
            tasks.append(asyncio.create_task(self.iterate_tiles()))
        if 'delay' in message:
            self.delay = message['delay']
        if 'stop' in message:
            self.stop = message['stop']
        return tasks

    def setup_async(self):
        self.delay = None
        self.stop = False

    async def async_notify(self, message_dict, flags=0):
        await self.notify_socket.send_json(message_dict, flags=flags)
        return True

    def put_tile(self):
        message_dict = random_msg(self.control_port)
        return message_dict

    async def monitor_control(self):
        tasks = []
        while True:
            tasks = [t for t in tasks if not t.done()]

            with self.tracer.span('control_message'):
                _ = await self.control_socket.poll()
                control_message = await self.control_socket.recv_json()
                await self.control_socket.send_json({'recieved': True, 'worker': self.control_port})
            tasks.extend(self.interpret_control(control_message))
            if self.stop:
                break

    async def iterate_tiles(self):
        run = True
        while run:
            if self.delay is not None:
                # print('Delaying')
                await asyncio.sleep(self.delay)

            with self.tracer.span('next_tile'):
                # message_dict = await asyncio.to_thread(self.dataset.next)
                message_dict = self.dataset.next()
            # print(message_dict)
            if message_dict.get('finished', False):
                message_dict['worker'] = self.control_port
                run = False
            with self.tracer.span('tile_notify'):
                await self.async_notify(message_dict)

    async def run_async(self):
        await self.monitor_control()
        print('End run_async')

    def add_dataset(self, dataset):
        self.dataset = dataset


def random_msg(worker=None):
    di = {'time': time.time()}
    if worker is not None:
        di['worker'] = worker
    return di


class ShmDataset(object):
    def __init__(self, dataset, iteration_params, plasma_addr, tracer=None):
        self.dataset = dataset
        self.iteration_params = iteration_params
        self.plasma_addr = plasma_addr
        self.client = None
        self._ds_iterator = None
        self.tracer = tracer
        self.tsize_cache = {}

    @property
    def is_connected(self):
        return self.client is not None

    def connect(self):
        if not self.is_connected:
            self.client = plasma.connect(self.plasma_addr)

    def disconnect(self):
        if self.is_connected:
            self.client.disconnect()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()

    def ds_iterator(self, reset=False):
        if reset or self._ds_iterator is None:
            self._ds_iterator = self.dataset.generate_tiles(**self.iteration_params)
        return self._ds_iterator

    def next(self):
        with self.tracer.span('iter_tile'):
            try:
                frame_slice, frame = next(self.ds_iterator())
            except StopIteration:
                return {'finished': True}
        with self.tracer.span('put_shm'):
            object_id = self.to_shm(frame)
        slice_tuple = split_ds_slice(frame_slice)
        return {'slice': slice_tuple,
            'object_id': object_id.binary().hex()}

    def to_shm(self, data):
        object_id = plasma.ObjectID.from_random()
        tensor = pa.Tensor.from_numpy(data)
        sizekey = data.shape + (data.nbytes, data.dtype)
        with self.tracer.span('tensor_size'):
            if sizekey in self.tsize_cache:
                tensor_size = self.tsize_cache[sizekey]
            else:
                tensor_size = pa.ipc.get_tensor_size(tensor)
                self.tsize_cache[sizekey] = tensor_size
        with self.tracer.span('get_buffer'):
            while True:
                try:
                    buf = self.client.create(object_id, tensor_size)
                    break
                except plasma.PlasmaStoreFull:
                    print('Full, sleeping')
                    time.sleep(0.001)
        with self.tracer.span('get_writer'):
            stream = pa.FixedSizeBufferWriter(buf)
            stream.set_memcopy_threads(6)
        with self.tracer.span('write'):
            pa.ipc.write_tensor(tensor, stream)
        with self.tracer.span('seal'):
            self.client.seal(object_id)
        return object_id


def split_ds_slice(ds_slice):
    values = []
    for a in ds_slice:
        if isinstance(a, int):
            values.append(a)
        else:
            values.append((a.start, a.stop, a.step))
    return values


def send_frames(endpoint_dict, plasma_addr, ds_meta, iteration_params, partition_idx):
    ds = ds_meta.to_dataset()
    ds = ds.partition(len(endpoint_dict))[partition_idx]

    ed = {k: v[partition_idx] for k, v in endpoint_dict.items()}
    with Context.instance() as ctx:
        with StreamingSender(ed, ctx=ctx) as streamer:
            with ShmDataset(ds, iteration_params, plasma_addr) as shm_dataset:
                streamer.setup_async()
                streamer.add_dataset(shm_dataset)
                asyncio.run(streamer.run_async())


if __name__ == '__main__':
    import sys
    import pathlib
    try:
        eid = int(sys.argv[1])
    except IndexError:
        warnings.warn('Must specify endpoint number, using 0')
        eid = 0

    # import pathlib
    # from dataset_reader import DatasetMeta
    # rawfile = pathlib.Path('../../ray_testing/random.raw')
    # sig_shape = (256, 256)
    # nav_shape = (10, 300)
    # ds_meta = DatasetMeta(rawfile, nav_shape, sig_shape)

    from run_streaming import ds_meta, endpoint_dict, plasma_addr, iteration_params
    ds = ds_meta.to_dataset()
    n_partitions = len(endpoint_dict['notify'])
    ds = ds.partition(n_partitions)[eid]

    ed = {k: v[eid] for k, v in endpoint_dict.items()}
    tracer = Tracer(f'Streamer', f'{eid}')
    with Context.instance() as ctx:
        with StreamingSender(ed, tracer=tracer, ctx=ctx) as streamer:
            with ShmDataset(ds, iteration_params, plasma_addr, tracer) as shm_dataset:
                streamer.setup_async()
                streamer.add_dataset(shm_dataset)
                asyncio.run(streamer.run_async())

    savepath = pathlib.Path(f'./tracing/streaming-{eid}.json')
    savepath.parent.mkdir(exist_ok=True)
    tracer.store(pathlib.Path(f'./tracing/streaming-{eid}.json'))
