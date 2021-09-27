import warnings
import pathlib
import numpy as np
from dataclasses import dataclass

from libertem.common.shape import Shape


def pairwise(iterable):
    for current in iterable:
        try:
            yield previous, current
        except NameError:
            pass
        previous = current


def divide_n_iterator(x, n):
    if n == 1:
        return [None, None]
    step = x // n
    return [min(a, x) for a in range(0, x + step, step)]


@dataclass
class DatasetMeta(object):
    filepath: pathlib.Path
    nav_shape: tuple
    sig_shape: tuple
    dtype: np.dtype = np.float32
    partition_slice: slice = None

    def __post_init__(self):
        if self.partition_slice is None:
            self.partition_slice = self.full_ds_slice

    @property
    def sig_dims(self):
        return len(self.sig_shape)

    @property
    def full_ds_slice(self):
        return slice(0, self.ds_nframes)

    @property
    def is_partitioned(self):
        return self.partition_slice != self.full_ds_slice

    @property
    def nframes(self):
        return self.partition_nframes

    @property
    def ds_nframes(self):
        return np.prod(self.nav_shape)

    @property
    def partition_nframes(self):
        return self.partition_slice.stop - self.partition_slice.start

    @property
    def shape(self):
        return Shape(self.nav_shape + self.sig_shape, self.sig_dims)

    @property
    def flat_shape(self):
        """This will always be the dataset flat shape for LT buffer compat"""
        return (self.ds_nframes,) + self.sig_shape

    @property
    def true_flat_shape(self):
        """This will give the real flat shape taking into account partitioning"""
        return (self.nframes,) + self.sig_shape

    @property
    def size(self):
        """Number of elements in (partitioned) datacube"""
        return np.prod(self.true_flat_shape)

    @property
    def size_bytes(self):
        return self.size * np.dtype(self.dtype).itemsize

    def partition_gen(self, n):
        """
        Yields new DSMeta objects with the partition slice attr set

        It is able to re-partition a partition
        """
        for start, stop in pairwise(divide_n_iterator(self.nframes, n)):
            yield self.__class__(self.filepath,
                                self.nav_shape,
                                self.sig_shape,
                                dtype=self.dtype,
                                partition_slice=slice(start, stop))

    def partition(self, n):
        assert n >= 1 and isinstance(n, int)
        if n == 1:
            return self
        return tuple([*self.partition_gen(n)])

    def to_dataset(self):
        return DatasetReader(self)


class DatasetReader(object):
    def __init__(self, ds_meta: DatasetMeta):
        self.ds_meta = ds_meta
        self._mmap = None

    def __repr__(self):
        ds = f'{self.__class__.__name__}(MemMap={"Open" if self.is_open else "Closed"})'
        return ds + '\n => ' + repr(self.ds_meta)

    @property
    def path(self):
        return pathlib.Path(self.ds_meta.filepath).absolute()

    @property
    def dtype(self):
        return self.ds_meta.dtype

    @property
    def sig_shape(self):
        return self.ds_meta.sig_shape

    @property
    def partition_slice(self):
        return self.ds_meta.partition_slice

    @property
    def is_open(self):
        return self._mmap is not None

    @property
    def data(self):
        if self.is_open:
            return self._mmap
        else:
            try:
                self.open()
                return self._mmap
            except Exception as e:
                e.message = e.message + \
                    '\nTried to access mmap before explicitly opening, auto-open failed'
                raise

    @property
    def filesize(self):
        return self.path.stat().st_size

    @property
    def frame_px(self):
        return np.prod(self.sig_shape)

    @property
    def frame_bytes(self):
        return self.frame_px * self.dtype_bytes

    @property
    def dtype_bytes(self):
        return np.dtype(self.dtype).itemsize

    def _mmap_offset(self):
        return self.first_frame * self.frame_bytes

    @property
    def nframes(self):
        return self.last_frame - self.first_frame

    @property
    def first_frame(self):
        return self.partition_slice.start

    @property
    def last_frame(self):
        return self.partition_slice.stop

    @property
    def shape(self):
        return (self.nframes,) + tuple(self.sig_shape)

    def open(self, mode='r'):
        if not self.is_open:
            self._mmap = np.memmap(self.path, dtype=self.dtype, mode=mode,
                                offset=self._mmap_offset(), shape=self.shape)
        else:
            warnings.warn('Trying to open dataset which is already open')

    def close(self):
        if self.is_open:
            self._mmap.flush()
        self._mmap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def generate_frames(self):
        """
        Generates individual frames with shape self.sig_dims
        and a dataset_slice which is (int, :, :)
        """
        for frame, frame_idx in zip(self.data, range(self.first_frame, self.last_frame)):
            yield np.s_[frame_idx, :, :], frame

    def generate_tiles(self, sig_split_yx, depth):
        """
        Generates stacked tiles of given depth where the sig dimensions are
        split into sig_split_yx==(ny, nx) tiles
        dataset_slice will be (frame_idx:frame_idx+depth, sigy:sigy+a, sigx:sigx+b)
        """
        first_frame = self.first_frame
        nframes = self.nframes

        slice_cache = [*self._tile_sig_slices(sig_split_yx)]
        for start_frame in range(0, nframes, depth):
            end_frame = min(start_frame + depth, nframes)
            for _, frame_slice in slice_cache:
                memmap_slice = (np.s_[start_frame:end_frame],) + frame_slice
                dataset_slice = (np.s_[first_frame+start_frame:
                                    first_frame+end_frame],) + frame_slice
                data = self.data[memmap_slice]
                yield dataset_slice, data

    def _tile_sig_slices(self, sig_split_yx):
        """Generates (sig_y, sig_x) slices to split frames into tiles"""
        h, w = self.sig_shape
        ny, nx = sig_split_yx

        slice_idx_map = {}
        xlims = divide_n_iterator(w, nx)
        ylims = divide_n_iterator(h, ny)
        for y0, y1 in pairwise(ylims):
            for x0, x1 in pairwise(xlims):
                try:
                    slice_idx = slice_idx_map[(y0, y1, x0, x1)]
                except KeyError:
                    try:
                        slice_idx = max(slice_idx_map.values()) + 1
                    except ValueError:
                        slice_idx = 0
                    slice_idx_map[(y0, y1, x0, x1)] = slice_idx
                yield slice_idx, (slice(y0, y1), slice(x0, x1))

    def partition_gen(self, n):
        for ds_meta in self.ds_meta.partition_gen(n):
            yield self.__class__(ds_meta)

    def partition(self, n):
        if n == 1:
            return self
        return tuple([*self.partition_gen(n)])


if __name__ == '__main__':
    rawfile = pathlib.Path('../../ray_testing/random.raw')
    sig_shape = (256, 256)
    nav_shape = (10, 300)
    ds_meta = DatasetMeta(rawfile, nav_shape, sig_shape)
    ds = ds_meta.to_dataset()
    part_0, part_1 = ds.partition(2)

    sig_split_yx = (4, 1)  # could add sig_split and depth to ds_meta
    depth = 16
    with part_1:
        for frame_slice, frame in part_1.generate_tiles(sig_split_yx, depth):
            print(frame_slice, frame.shape)
