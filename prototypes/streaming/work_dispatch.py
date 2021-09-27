import numpy as np

from libertem.udf.base import UDF, UDFMeta
from libertem.common.slice import Slice
from libertem.common.shape import Shape
from tile_wrapper import TileReciever
from dataset_reader import DatasetMeta

import time

import ray

import pyarrow as pa
import pyarrow.plasma as plasma
from tracing import Tracer

PLASMA_ADDR = "/tmp/plasma"


def set_plasma_addr(addr):
    global PLASMA_ADDR
    PLASMA_ADDR = addr


class ShmMixin(object):
    def __init__(self, plasma_addr=None):
        if plasma_addr is None:
            plasma_addr = PLASMA_ADDR
        self.plasma_addr = plasma_addr
        self.client = None

        self.buffers = {}

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
        return self

    def __exit__(self, *args):
        self.disconnect()

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


class BothSumUDF(UDF):
    def get_result_buffers(self):
        return {'sig_intensity': self.buffer(kind='sig', dtype=self.meta.input_dtype),
                'nav_intensity': self.buffer(kind='nav', dtype=self.meta.input_dtype)}

    def process_tile(self, tile):
        time.sleep(0.001)
        self.results.sig_intensity[:] += np.sum(tile, axis=0)
        self.results.nav_intensity[:] += np.sum(tile, axis=(1, 2))

    def merge(self, dest, src):
        dest.sig_intensity[:] += src.sig_intensity
        dest.nav_intensity[:] += src.nav_intensity


class UDFCoordinatorBase(object):
    def __init__(self, udfs, ds_meta, corrections=None, roi=None, worker_id=None):
        self.corrections = corrections
        self.roi = roi
        self.ds_meta = ds_meta
        # This is here for compatibility with existing UDF buffer architecture
        self._partition = FakePartition(ds_meta.nframes, ds_meta.sig_shape)
        self.tracer = Tracer('Worker', worker_id)

        if not isinstance(udfs, (list, tuple)):
            udfs = [udfs]
        with self.tracer.span('Instantiate UDFs'):
            self.udfs = self.instantiate_udfs(udfs)
        self.slice_history = []
        self.wrapper = TileReciever(ds_meta)

        with self.tracer.span('Create SHM connection'):
            self.shm = ShmMixin()
            self.shm.connect()
            self.shm.warmup()


    def instantiate_udfs(self, udfs):
        valid_udfs = []
        for udf in udfs:
            if isinstance(udf, UDF):
                pass
            elif isinstance(udf, dict):
                udf_class = udf['class']
                udf_args = udf['args']
                udf_kwargs = udf['kwargs']
                udf = udf_class(*udf_args, **udf_kwargs)
            else:
                raise ValueError(f'Unrecognized UDF format in {self.__class__}')

            meta = UDFMeta(
                partition_shape=self.ds_meta.nav_shape,#self._partition.slice.adjust_for_roi(self.roi).shape,
                dataset_shape=self.ds_meta.shape,
                roi=self.roi,
                dataset_dtype=self.ds_meta.dtype,
                input_dtype=self.ds_meta.dtype,
                tiling_scheme=None,
                corrections=self.corrections,
                device_class='cpu',
                threads_per_worker=1,
            )
            udf.set_meta(meta)
            udf.init_result_buffers()
            udf.allocate_for_full(self.ds_meta, self.roi)
            udf.init_task_data()
            valid_udfs.append(udf)
        return valid_udfs

    def store_tracer(self):
        self.tracer.store(f'./tracing/worker-{self.tracer.tid}.json')


class FakePartition(object):
    def __init__(self, nframes, sig_shape):
        self.slice = Slice(origin=(0, 0, 0), shape=Shape((nframes,)+sig_shape, len(sig_shape)))


class UDFRunner(UDFCoordinatorBase):
    def process_tile(self, frame_slice, frame):
        retval = None
        with self.tracer.span('Deserialize frame'):
            if isinstance(frame, (str, bytes)):
                retval = frame
                frame = self.shm.np_from_oid(frame)

        """Can in principle run different UDFs on partitions, frames and tiles"""
        with self.tracer.span('Wrap tile'):
            datatile = self.wrapper.wrap_tile(frame_slice, frame)

        with self.tracer.span('Run UDFs'):
            for udf in self.udfs:
                udf.set_contiguous_views_for_tile(self._partition, datatile)
                udf.set_slice(datatile.tile_slice)
                with self.tracer.span('process_tile'):
                    udf.process_tile(datatile)

        self.slice_history.append(datatile.tile_slice)
        return retval

    def iterate_tiles(self, ds_meta: DatasetMeta, iteration_params):
        ds = ds_meta.to_dataset()
        for frame_slice, frame in ds.generate_tiles(**iteration_params):
            self.process_tile(frame_slice, frame)
        return True

    def warmup(self, arg):
        with self.tracer.span('warmup'):
            if arg:
                return np.zeros((5, 5))
            else:
                return np.ones((5, 5))

    def wrapup_udfs(self):
        for udf in self.udfs:
            udf.flush()
            if hasattr(udf, 'postprocess'):
                udf.clear_views()
                udf.postprocess()

            udf.cleanup()
            udf.clear_views()
            udf.export_results()

    def get_results(self):
        return self.slice_history, tuple(udf.results for udf in self.udfs)


UDFRunner_r = ray.remote(UDFRunner)
UDFRunner_r.options(num_cpus=1.0)


class UDFMerger(UDFCoordinatorBase):
    def __init__(self, runners, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.runners = runners

    def merge_runners(self):
        results_by_udf = [[] for _ in self.udfs]
        slice_histories = []
        nav_masks = []
        for runner in self.runners:
            runner.wrapup_udfs()
            slice_history, udf_results = runner.get_results()
            slice_histories.append(slice_history)
            nav_masks.append(self.nav_mask_from_slices(slice_history))
            for res, udf_res in zip(udf_results, results_by_udf):
                udf_res.append(res)

        for runner_results, udf in zip(results_by_udf, self.udfs):
            udf.set_views_for_partition(self._partition)
            for part_res, nav_mask in zip(runner_results, nav_masks):
                # makes a copy of nav-type buffers when applying the mask
                masked_part_res = part_res.get_proxy_masked(nav_mask)
                masked_udf_res = udf.results.get_proxy_masked(nav_mask)
                udf.merge(dest=masked_udf_res, src=masked_part_res)
                # Merged into masked_udf_res, now will update valid values in udf.results
                self.reassign_masked(udf.results, masked_udf_res, nav_mask)
            udf.clear_views()

    def nav_mask_from_slices(self, slice_history):
        mask = np.zeros(self.ds_meta.nav_shape, dtype=bool)
        for slice in slice_history:
            nav_origin = slice.origin[0]
            depth = slice.shape[0]
            np_slice = np.s_[nav_origin:nav_origin + depth, ...]
            mask.ravel()[np_slice] = True
        return mask

    def reassign_masked(self, udf_results, masked_results, nav_mask):
        for udf_res_name in udf_results.keys():
            udf_res = udf_results.get_buffer(udf_res_name)
            masked_res = getattr(masked_results, udf_res_name)
            if udf_res.kind == 'sig':
                continue
            else:
                udf_res.data[nav_mask] = masked_res

    @property
    def results(self):
        return tuple(udf.results.finalize() for udf in self.udfs)


class RunnerProxy(object):
    def __init__(self, actor):
        self.actor = actor

    def get_results(self):
        return ray.get(self.actor.get_results.remote())

    def wrapup_udfs(self):
        return ray.get(self.actor.wrapup_udfs.remote())


if __name__ == '__main__':
    nworkers = 2
    use_ray = False
    if use_ray:
        ray.init(address='auto', _redis_password='5241590000000000')

    import pathlib
    import itertools
    import matplotlib.pyplot as plt
    from dataset_reader import DatasetMeta

    rawfile = pathlib.Path('../../ray_testing/random.raw')
    sig_shape = (256, 256)
    nav_shape = (20, 300)
    ds_meta = DatasetMeta(rawfile, nav_shape, sig_shape)
    ds = ds_meta.to_dataset()
    sig_split_yx = (4, 1)
    depth = 16

    shm = ShmMixin()
    shm.connect()

    udfs = [{'class': BothSumUDF, 'args': [], 'kwargs': {}}]
    if use_ray:
        actors = [UDFRunner_r.remote(udfs, ds_meta) for _ in range(nworkers)]
        pool = ray.util.ActorPool(actors)
    else:
        runners = [UDFRunner(udfs, ds_meta) for _ in range(nworkers)]

    if use_ray:
        futures = pool.map_unordered(lambda a, v: a.process_tile.remote(*v),
                                ds.generate_tiles(sig_split_yx, depth))
        _ = [*futures]
        runners = [RunnerProxy(a) for a in actors]
    else:
        for (frame_slice, frame), runner in zip(ds.generate_tiles(sig_split_yx, depth),
                                                itertools.cycle(runners)):
            oid = shm.np_to_shm(frame)
            oid_string = oid.binary().hex()
            runner.process_tile(frame_slice, oid_string)

    shm.disconnect()

    merger = UDFMerger(runners, udfs, ds_meta)
    merger.merge_runners()

    res = merger.results[0]
    fig, axs = plt.subplots(1, 2)
    for k, ax in zip(res.keys(), axs):
        ax.imshow(res.get_buffer(k))
        ax.set_title(k)
    plt.show()
