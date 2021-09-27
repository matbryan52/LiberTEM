import pathlib
import asyncio
from collections import deque
import numpy as np

from dataset_reader import DatasetReader


async def main():
    rawfile = pathlib.Path('../../ray_testing/random.raw')
    sig_shape = (256, 256)
    p_slice = slice(1000, 2000)
    reader = DatasetReader(rawfile, sig_shape, partition_slice=p_slice)
    reader.open()

    sig_split_yx = (16, 1)
    depth = 16

    que = deque(maxlen=10)
    async for ret in reader.generate_tiles(sig_split_yx, depth):
        que.append(ret)



if __name__ == '__main__':
    asyncio.run(main())