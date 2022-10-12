import pathlib
import tempfile
import contextlib
import numpy as np
import time
import tqdm
import click

import libertem.api as lt
from libertem.common.math import prod
from libertem.io.dataset.raw import RawFileDataSet
from libertem.io.dataset.dm4 import DM4DataSet
from libertem.udf.sumsigudf import SumSigUDF
from libertem.udf.base import UDF, UDFRunner

from bench_utils import warmup_cache, drop_cache


class SumFrameUDF(UDF):
    def get_result_buffers(self):
        return {'intensity': self.buffer(kind='nav')}

    def process_frame(self, frame):
        self.results.intensity[:] = frame.sum()


class RawDM4Like(RawFileDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._array_c_ordered = False

    def get_partitions(self):
        return DM4DataSet.get_partitions(self)

    def adjust_tileshape(self, *args, **kwargs):
        return DM4DataSet.adjust_tileshape(self, *args, **kwargs)

    def need_decode(self, *args, **kwargs):
        return DM4DataSet.need_decode(self, *args, **kwargs)


@contextlib.contextmanager
def get_data(nav_shape, sig_shape, dtype):
    nav_size = prod(nav_shape)
    sig_size = prod(sig_shape)
    with tempfile.TemporaryDirectory() as tempdir:
        path = pathlib.Path(tempdir) / 'data.raw'
        vec: np.ndarray = np.linspace(0., 1., num=nav_size, endpoint=True, dtype=dtype)
        with path.open('wb') as fp:
            for _ in range(sig_size):
                vec.tofile(fp)
        yield path


def get_ds_shape(ds_size_mb, sig_size_mb, dtype):
    itemsize = np.dtype(dtype).itemsize
    sig_dim = np.sqrt(sig_size_mb * 2**20 / itemsize)
    sig_shape = (max(1, np.floor(sig_dim).astype(int).item()),
                 max(1, np.ceil(sig_dim).astype(int).item()))
    sig_size = prod(sig_shape) * itemsize
    nav_dim = np.sqrt(ds_size_mb * 2**20 / sig_size)
    nav_shape = (max(1, np.floor(nav_dim).astype(int).item()),
                 max(1, np.ceil(nav_dim).astype(int).item()))
    true_size_mb = prod(nav_shape + sig_shape) * itemsize / 2**20
    return nav_shape, sig_shape, true_size_mb


@click.command()
@click.option('-d', '--ds_size_mb', help='dataset size',
              default=1024, type=int, show_default=True)
@click.option('-s', '--sig_size_mb', help='signal size',
              default=1, type=int)
@click.option('-r', '--repeats', help='number of runs',
              default=10, type=int)
@click.option('--warm',
              help="warm cache",
              default=False, is_flag=True)
def main(ds_size_mb, sig_size_mb, repeats, warm):
    ctx = lt.Context.make_with('inline')
    dtype = np.float32
    roi = None
    corrections = None
    udf_class = SumSigUDF

    drop_caches = not warm

    nav_shape, sig_shape, true_size_mb = get_ds_shape(ds_size_mb, sig_size_mb, dtype)
    print(f'{nav_shape}, {sig_shape}, {str(np.dtype(np.float32))}, {true_size_mb:.1f} MB), warm caches: {warm}')
    print('Creating data...', end='', flush=True)
    tstart = time.perf_counter()

    with get_data(nav_shape, sig_shape, dtype) as path:
        print(f'Done ({true_size_mb / (time.perf_counter() - tstart):.1f} MB/s')
        ds = RawDM4Like(path=path, nav_shape=nav_shape, sig_shape=sig_shape, dtype=dtype)
        ds.initialize(ctx.executor)
        udf = udf_class()

        tasks, params = UDFRunner([udf])._prepare_run_for_dataset(ds, ctx.executor, roi, corrections, None, False)
        print(f'Num partitions {len(tasks)}, {udf.__class__.__name__}, {ctx.executor.__class__.__name__}')
        print(f'{params.tiling_scheme}, {params.tiling_scheme.intent}')

        runs = []
        for _ in tqdm.trange(repeats):
            if drop_caches:
                drop_cache([str(path)])
            else:
                warmup_cache([str(path)])
            tstart = time.perf_counter()
            res = ctx.run_udf(dataset=ds, udf=udf, roi=roi, corrections=corrections)
            runs.append(time.perf_counter() - tstart)

        assert res['intensity'].data[0, 0] == 0

    runs = np.asarray(runs)
    print(f'Average of {repeats} runs, mean {runs.mean():.2f} s, '
          f'min-max ({runs.min():.2f}, {runs.max():.2f}) s')
    print(f'Processing speed (mean) {true_size_mb / runs.mean():.1f} MB/s, '
          f'cache: {"cold" if drop_caches else "warm"}')
    return runs


if __name__ == '__main__':
    main()
