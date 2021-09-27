from asyncio.tasks import as_completed
import time
import pathlib
import warnings
import numpy as np
import pyarrow as pa
import pyarrow.plasma as plasma
import asyncio
import zmq
from zmq.asyncio import Context
import ray


from tracing import Tracer

class StreamingReciever(object):
    def __init__(self, endpoint_dict, ctx=None, tracer=None):
        self.ctx = ctx
        self.own_context = ctx is None
        self.endpoint_dict = endpoint_dict
        self.sockets = {'control': None}
        self.subscription_filter = b''
        self.tracer=tracer

        _ = self.control_sockets
        _ = self.notify_socket

    @property
    def n_control(self):
        return len(self.endpoint_dict['control'])

    @property
    def context(self):
        if self.ctx is None:
            self.ctx = zmq.Context.instance()
        else:
            return self.ctx

    @property
    def notify_socket(self):
        try:
            return self.sockets['notify']
        except KeyError:
            with self.tracer.span('create notify socket'):
                self.sockets['notify'] = self.context.socket(zmq.SUB)
                self.notify_socket.setsockopt(zmq.SUBSCRIBE, self.subscription_filter)
            return self.notify_socket

    @property
    def control_sockets(self):
        if self.sockets['control'] is None:
            with self.tracer.span('create control sockets'):
                self.sockets['control'] = [self.context.socket(zmq.REQ) for _ in range(self.n_control)]
        return self.sockets['control']

    def connect(self):
        print('Connecting control sockets:')
        for socket, control_addr in zip(self.control_sockets, self.endpoint_dict['control']):
            print(f'\t{control_addr}')
            socket.bind(control_addr)
        print('Connecting notify socket:')
        for notify_addr in self.endpoint_dict['notify']:
            print(f'\t{notify_addr}')
            self.notify_socket.connect(notify_addr)

    def disconnect(self):
        for name, slist in self.sockets.items():
            if not isinstance(slist, list):
                sockets = [slist]
            else:
                sockets = slist
            for socket in sockets:
                if not socket.closed:
                    socket.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()
        if self.own_context:
            if not self.context.closed:
                self.context.close()

    async def control_message(self, message):
        coros = [self._control_message(socket, message) for socket in self.control_sockets]
        return await asyncio.gather(*coros)

    async def _control_message(self, socket, message):
        await socket.send_json(message)
        return await socket.recv_json()

    async def iterate_workers(self):

        # Start workers
        with self.tracer.span('start_streamers'):
            replies = await self.control_message({'start': True})
            for r in replies:
                print(r)

        # Recieve messages
        running = {r['worker']: True for r in replies}
        tasks = []
        
        self.dispatcher.run = True
        # future_task = asyncio.create_task(self.dispatcher.check_futures())
        while any(running.values()):
            with self.tracer.span('recieve_tile'):
                _ = await self.notify_socket.poll()
                msg = await self.notify_socket.recv_json()
            if msg.get('finished', False):
                running[msg['worker']] = False
            else:
                with self.tracer.span('dispatch_work'):
                    tasks.append(asyncio.create_task(self.dispatcher.process_message(msg)))
                    # await self.dispatcher.process_message(msg)
        
        with self.tracer.span('stop_streamers'):
            await asyncio.create_task(self.stop_workers())

        self.dispatcher.run = False
        for task in asyncio.as_completed(tasks):
            await task

    async def run_async(self):
        await self.iterate_workers()

    async def stop_workers(self):
        # Stop workers
        replies = await self.control_message({'stop': True})
        for r in replies:
            print(r)

    def add_dispatcher(self, dispatcher):
        self.dispatcher = dispatcher


class ShmWorkOrchestrator(object):
    def __init__(self, plasma_addr, worker_pool, tracer=None):
        self.plasma_addr = plasma_addr
        self.client = None
        self.worker_pool = worker_pool
        self.run = True

        self.buffers = {}
        self.futures = {}
        self.tracer = tracer

    def warmup(self):
        obj_id = self.np_to_shm(np.ones((5, 5)))
        _ = self.np_from_oid(obj_id.binary().hex())

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
        self.warmup()
        return self

    def __exit__(self, *args):
        self.disconnect()

    async def process_message(self, msg):
        # with self.tracer.span('check_futures'):
        #     await self.check_futures()

        with self.tracer.span('construct_args'):
            obj_id_string = msg['object_id']
            object_id = plasma.ObjectID(bytes.fromhex(obj_id_string))
            self.buffers[obj_id_string] = self.client.get_buffers([object_id])[0]
            dataset_slice = reconstruct_ds_slice(msg['slice'])

        with self.tracer.span('submit_work'):
            future = self.worker_pool.submit(dataset_slice, obj_id_string)
            # self.futures[future] = obj_id_string
        with self.tracer.span('await_result'):
            await future
        with self.tracer.span('popping_buffer'):
            _ = self.buffers.pop(obj_id_string, None)

        self.tracer.frame()
            # print(f'N-Futures: {len(self.futures)}')

    async def check_futures(self, timeout_ms=0.1):
        while self.run:
            if len(self.futures) == 0:
                # print('No futures!')
                await asyncio.sleep(0.001)
                continue
            with self.tracer.span('wait_on_f'):
                done, pending = ray.wait([*self.futures.keys()],
                               timeout=timeout_ms*0.001, num_returns=1)
                # done, pending = await asyncio.wait([*self.futures.values()],
                #                                 timeout=timeout_ms*0.001,
                #                                 return_when=asyncio.FIRST_COMPLETED)

            # print('Done, pending', len(done), len(pending))
            for fut in done:
                with self.tracer.span('pop_buffer'):
                    # obj_id = await task
                    obj_id = self.futures.pop(fut, None)
                    buf = self.buffers.pop(obj_id, None)
                    # print('Popping', obj_id, fut, buf)
        
        # self.buffers = {}

    def np_from_oid(self, obj_id_string):
        object_id = plasma.ObjectID(bytes.fromhex(obj_id_string))
        buffer = self.client.get_buffers([object_id])[0]
        buf_reader = pa.BufferReader(buffer)
        tensor = pa.ipc.read_tensor(buf_reader)
        return tensor.to_numpy()

    def np_to_shm(self, data):
        object_id = plasma.ObjectID.from_random()
        tensor = pa.Tensor.from_numpy(data)
        tensor_size = pa.ipc.get_tensor_size(tensor)
        buf = self.client.create(object_id, tensor_size)
        stream = pa.FixedSizeBufferWriter(buf)
        stream.set_memcopy_threads(6)
        pa.ipc.write_tensor(tensor, stream)
        self.client.seal(object_id)
        return object_id

        # with self.tracer.span('wait_done'):
        #     done, unfinished = await asyncio.wait([*self.futures.keys()],
        #                                     timeout=timeout_ms*0.001)
        #     async for future in asyncio.as_completed([*self.futures.keys()], timeout=timeout_ms*0.001):
        #         obj_id = await future
        #     done, unfinished = ray.wait([*self.futures.keys()],
        #                         num_returns=nfutures,
        #                         timeout=timeout_ms*0.001,
        #                         fetch_local=False)
        # with self.tracer.span('pop_buffers'):
        #     _ = [self.buffers.pop(self.futures[f], None) for f in done]
        #     self.futures = {f: self.futures[f] for f in unfinished}

    # def check_futures(self):
    #     while True:
    #         with self.tracer.span('check_next'):
    #             if self.worker_pool.has_next():
    #                 with self.tracer.span('get_next'):
    #                     obj_id = self.worker_pool.get_next()
    #                 with self.tracer.span('pop_buffer'):
    #                     _ = self.buffers.pop(obj_id, None)
    #             else:
    #                 break
        # nfutures = len(self.futures)
        # if nfutures > 0:
        #     done, self.futures = await ray.wait(self.futures, num_returns=nfutures, timeout=0.1)
        #     async for obj_id in ray.get(done):
        #         _ = self.buffers.pop(obj_id, None)


def reconstruct_ds_slice(ds_slice):
    values = []
    for a in ds_slice:
        if isinstance(a, int):
            values.append(a)
        else:
            values.append(np.s_[a[0]:a[1]:a[2]])
    return tuple(values)


def make_addr(port, addr='127.0.0.1', protocol='tcp'):
    return f'{protocol}://{addr}:{port}'


def recieve_frames(endpoint_dict, plasma_addr, worker_pool, tracer=None):
    if tracer is None:
        tracer = Tracer('TileDispatcher', '0')
    with Context.instance() as ctx:
        with StreamingReciever(endpoint_dict, ctx=ctx, tracer=tracer) as reciever:
            with ShmWorkOrchestrator(plasma_addr, worker_pool, tracer=tracer) as shm_dispatcher:
                reciever.add_dispatcher(shm_dispatcher)
                asyncio.run(reciever.run_async())
    return tracer


# import pathlib
# from dataset_reader import DatasetMeta

# n_partitions = 2
# start_control = 50005
# start_notify = 51005
# endpoint_dict = {'control': [make_addr(p) for p in range(start_control, start_control + n_partitions)],
#                 'notify': [make_addr(p) for p in range(start_notify, start_notify + n_partitions)]}
# plasma_addr = "/tmp/plasma"

# rawfile = pathlib.Path('../../ray_testing/random.raw')
# sig_shape = (256, 256)
# nav_shape = (10, 300)
# ds_meta = DatasetMeta(rawfile, nav_shape, sig_shape)
# iteration_params = {'sig_split_yx': (4, 1), 'depth': 16}

if __name__ == '__main__':

    # from zmq import Context
    from zmq.asyncio import Context
    with Context.instance() as ctx:
        with StreamingReciever(endpoint_dict, ctx=ctx) as reciever:
            with ShmWorkOrchestrator(plasma_addr) as shm_dispatcher:
                reciever.add_dispatcher(shm_dispatcher)
                asyncio.run(reciever.run_async())
