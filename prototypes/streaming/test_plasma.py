import numpy as np
import pyarrow as pa
import pyarrow.plasma as plasma
import time

def main():
    client = plasma.connect("/tmp/plasma")
    # for nb in np.logspace(5, 8, num=10):
    nb = 10000000
    data = np.random.randn(int(nb))
    timings = []
    for nb in range(100):

        a = time.time()
        tensor = pa.Tensor.from_numpy(data)
        object_id = plasma.ObjectID.from_random()
        buf = client.create(object_id, pa.ipc.get_tensor_size(tensor))
        stream = pa.FixedSizeBufferWriter(buf)
        stream.set_memcopy_threads(4)
        pa.ipc.write_tensor(tensor, stream)
        client.seal(object_id)

        client.delete([object_id])
        del object_id
        del buf

        t = time.time() - a
        timings.append(t)
        mb = data.nbytes / 1e6
        avg = sum(timings) / len(timings)
        print(f'{mb} MB, Writing took {t}, Speed {mb / (1000 * avg):.1f} GB/s')

if __name__ == '__main__':
    main()
