import os
os.environ["KMP_WARNINGS"] = "off"

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
from libertem.io.dataset.base.backend_fortran import FortranReader
from libertem.udf.sumsigudf import SumSigUDF
from libertem.udf.base import UDF, UDFRunner

from bench_utils import warmup_cache, drop_cache


class SumFrameUDF(UDF):
    def get_result_buffers(self):
        return {'intensity': self.buffer(kind='nav')}

    def process_frame(self, frame):
        self.results.intensity[:] = frame.sum()


class RawDM4Like(RawFileDataSet):
    def __init__(self, *args, num_part=None, tileshape=None, max_io=1, **kwargs):
        super().__init__(*args, **kwargs)
        self._array_c_ordered = False
        self._num_part = num_part
        self._tileshape = tileshape
        self._max_io = max_io

    def get_partitions(self):
        return DM4DataSet.get_partitions(self)

    def adjust_tileshape(self, *args, **kwargs):
        if self._tileshape is None:
            return DM4DataSet.adjust_tileshape(self, *args, **kwargs)
        if isinstance(self._tileshape, str):
            tileshape = eval(self._tileshape)
        else:
            tileshape = self._tileshape
        shape = []
        max_depth = max(s.shape.nav.size for s, _, _ in self.get_slices())
        if isinstance(tileshape, tuple):
            assert len(tileshape) == 3
            for dim_idx, (tile_dim, shape_dim) in enumerate(zip(tileshape, self.shape.flatten_nav())):
                if tile_dim is None:
                    if dim_idx == 0:
                        shape.append(max_depth)
                    else:
                        shape.append(shape_dim)
                elif isinstance(tile_dim, float):
                    if dim_idx == 0:
                        shape.append(max(1, int(tile_dim * max_depth)))
                    else:
                        shape.append(max(1, int(tile_dim * shape_dim)))
                else:
                    shape.append(tile_dim)
            return tuple(shape)
        else:
            assert isinstance(tileshape, (int, float))
            if isinstance(tileshape, float):
                tileshape = max(1, int(max_depth * tileshape))
            depth = min(max_depth, tileshape)
            max_io_size = self.get_max_io_size()
            itemsize = np.dtype(self.meta.raw_dtype).itemsize
            sig_pix = max_io_size // (depth * itemsize)
            if sig_pix == 0:
                depth = max_io_size // itemsize
                sig_pix = 1
            rows, cols = self.shape.sig
            if sig_pix < cols:
                shape = (depth, 1, sig_pix)
            else:
                shape = (depth, max(sig_pix // rows, 1), cols)
            return shape

    def need_decode(self, *args, **kwargs):
        return DM4DataSet.need_decode(self, *args, **kwargs)

    def get_num_partitions(self) -> int:
        if self._num_part is not None:
            return self._num_part
        else:
            return super().get_num_partitions()

    def get_max_io_size(self):
        if self._max_io is not None:
            return self._max_io * 2**20
        else:
            return self._max_io


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


@contextlib.contextmanager
def adapt_param(obj, param, value):
    oldval = getattr(obj, param)
    setattr(obj, param, value)
    yield
    setattr(obj, param, oldval)


@contextlib.contextmanager
def adapt_params(obj, mods):
    with contextlib.ExitStack() as stack:
        for param, value in mods.items():
            if value is None:
                continue
            stack.enter_context(adapt_param(obj, param, value))
        yield


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
@click.option('--roi',
              help="use roi",
              default=False, is_flag=True)
@click.option('-n', '--num_part', help='number of partitions',
              default=None, type=int)
@click.option('--max_io', help='max_io_size mb',
              default=1, type=int)
@click.option('--combine', default=None, type=int)
@click.option('--memmap', default=None, type=int)
@click.option('--buffer', default=None, type=int)
@click.option('--tileshape', default=None, type=str)
def main(ds_size_mb, sig_size_mb, repeats, warm, num_part,
         combine, memmap, buffer, tileshape, max_io, roi):
    ctx = lt.Context.make_with('inline')
    dtype = np.float32
    corrections = None
    udf_class = SumSigUDF

    mods = {
        'MAX_MEMMAP_SIZE': memmap,
        'BUFFER_SIZE': buffer,
        'THRESHOLD_COMBINE': combine,
    }
    for key in mods.keys():
        if 'SIZE' in key and mods[key] is not None:
            mods[key] = mods[key] * 2 ** 20

    drop_caches = not warm

    nav_shape, sig_shape, true_size_mb = get_ds_shape(ds_size_mb, sig_size_mb, dtype)
    print(f'{nav_shape}, {sig_shape}, {str(np.dtype(np.float32))}, {true_size_mb:.1f} MB), warm caches: {warm}')
    print('Creating data...', end='', flush=True)
    tstart = time.perf_counter()

    if roi:
        roi_a = np.random.choice([True, False], size=nav_shape)
    else:
        roi_a = None

    with get_data(nav_shape, sig_shape, dtype) as path:
        print(f'Done ({true_size_mb / (time.perf_counter() - tstart):.1f} MB/s)')
        with adapt_params(FortranReader, mods):
            ds = RawDM4Like(path=path, nav_shape=nav_shape, sig_shape=sig_shape,
                            dtype=dtype, num_part=num_part, tileshape=tileshape,
                            max_io=max_io)
            ds.initialize(ctx.executor)
            udf = udf_class()

            tasks, params = UDFRunner([udf])._prepare_run_for_dataset(ds, ctx.executor, roi_a, corrections, None, False)
            print(f'Num partitions {len(tasks)}, {udf.__class__.__name__}, {ctx.executor.__class__.__name__}')
            print(f'{params.tiling_scheme}, {params.tiling_scheme.intent}')
            # Warmup .pyc / numba etc
            res = ctx.run_udf(dataset=ds, udf=udf, roi=roi_a, corrections=corrections)

            runs = []
            for _ in tqdm.trange(repeats):
                if drop_caches:
                    drop_cache([str(path)])
                else:
                    warmup_cache([str(path)])
                tstart = time.perf_counter()
                res = ctx.run_udf(dataset=ds, udf=udf, roi=roi_a, corrections=corrections)
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
