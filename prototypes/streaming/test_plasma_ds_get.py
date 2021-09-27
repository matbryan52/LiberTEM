import pathlib
from dataset_reader import DatasetReader
import numpy as np
import pyarrow as pa
import pyarrow.plasma as plasma
import time
from collections import deque
import itertools

TSIZE = None

def write_tensor(client, data):
    if True:
        global TSIZE
        object_id = plasma.ObjectID.from_random()
        tensor = pa.Tensor.from_numpy(data)
        if TSIZE is None:
            TSIZE = pa.ipc.get_tensor_size(tensor)
            print(f'{TSIZE / 2**20} MB')
        buf = client.create(object_id, TSIZE)
        stream = pa.FixedSizeBufferWriter(buf)
        stream.set_memcopy_threads(6)
        pa.ipc.write_tensor(tensor, stream)
        client.seal(object_id)
    else:
        bt = data.tobytes()
        object_id = client.put_raw_buffer(bt)
    return object_id


def benchmark_timings(reader, client, f_copy=False):
    global TSIZE

    timings = {}
    for depth in range(1, 128, 4):
        TSIZE = None
        timings[depth] = []
        maxref = 100
        obj_ids = deque(maxlen=maxref)
        for idx, (frame_slice, frame) in enumerate(reader.generate_tiles(sig_split_yx, depth)):
            if f_copy:
                frame = frame.copy()
            a = time.time()
            object_id = write_tensor(client, frame)
            timings[depth].append((time.time() - a) * 1000)

            while len(obj_ids) > obj_ids.maxlen:
                _del_id = obj_ids.popleft()
                client.delete([_del_id])
                del _del_id

            obj_ids.append(object_id)
    return timings



if __name__ == '__main__':
    rawfile = pathlib.Path('../../ray_testing/random.raw')
    sig_shape = (256, 256)
    p_slice = slice(0, 4000)
    # depth = 32
    sig_split_yx = (1, 1)
    reader = DatasetReader(rawfile, sig_shape, partition_slice=p_slice)
    reader.open()
    client = plasma.connect("/tmp/plasma")

    # mem_timings = benchmark_timings(reader, client)
    # copy_timings = benchmark_timings(reader, client, f_copy=True)

    nobjects = 10
    n_repeat = 10

    all_timings = {}
    for depth in range(1, 128, 4):
        TSIZE = None
        all_timings[depth] = []
        maxref = 100
        client = plasma.connect("/tmp/plasma")
        obj_ids = []
        for idx, (frame_slice, frame) in enumerate(reader.generate_tiles(sig_split_yx, depth)):
            if idx > nobjects:
                break
            object_id = write_tensor(client, frame)
            obj_ids.append(object_id)

        for idx, obj_id in enumerate(itertools.cycle(obj_ids)):
            if idx > nobjects * n_repeat:
                break
            a = time.time()
            buf = client.get_buffers([obj_id])[0]
            buf_reader = pa.BufferReader(buf)
            tensor = pa.ipc.read_tensor(buf_reader)
            array = tensor.to_numpy()
            all_timings[depth].append((time.time() - a) * 1000)


    import matplotlib.pyplot as plt
    display_num = 100
    fig, axs = plt.subplots(1, 2)
    ax = axs[0]
    for depth, timings in all_timings.items():
        itemsize = np.prod(sig_shape) * np.dtype(np.float32).itemsize * depth // 2**20
        timings = np.random.choice(np.asarray(timings), size=min(display_num, len(timings)), replace=False)
        itemsize = np.ones_like(timings).astype(int) * itemsize
        if depth == 1:
            ax.plot(itemsize, timings, 'kx', alpha=0.05, label='memmap')
        else:
            ax.plot(itemsize, timings, 'kx', alpha=0.05)
    ax.set_xlabel('Object size (MB)')
    ax.set_ylabel('Get time (ms)')
    ax.legend()
    ax.set_title('Object get time with size')

    ax = axs[1]
    for depth, timings in all_timings.items():
        itemsize = np.prod(sig_shape) * np.dtype(np.float32).itemsize * depth // 2**20
        timings = np.random.choice(np.asarray(timings), size=min(display_num, len(timings)), replace=False)
        itemsize = np.ones_like(timings).astype(int) * itemsize
        if depth == 1:
            ax.plot(itemsize, (itemsize / timings), 'kx', alpha=0.05, label='memmap')
        else:
            ax.plot(itemsize, (itemsize / timings), 'kx', alpha=0.05)
    ax.set_xlabel('Object size (MB)')
    ax.set_ylabel('Get rate (GB/s)')
    ax.legend()
    ax.set_title('Object get rate with size')

    plt.show()    

