import os
import shutil

import numpy as np
from numpy.testing import assert_allclose
import pytest
import zarr
from zarr.errors import PathNotFoundError
import zarr.convenience
import zarr.hierarchy

from libertem.io.dataset.zarr import ZarrDataSet
from libertem.io.dataset.base import TilingScheme
from libertem.common import Shape
from libertem.analysis.sum import SumAnalysis
from libertem.udf.sumsigudf import SumSigUDF
from libertem.udf.auto import AutoUDF
from libertem.udf import UDF
from libertem.io.dataset.base import Negotiator

from utils import _mk_random, PixelsumUDF, _naive_mask_apply


def get_or_create_zarr(tmpdir_factory, filename, **kwargs):
    datadir = tmpdir_factory.mktemp('data')
    filename = os.path.join(datadir, filename)
    try:
        yield zarr.convenience.open(filename, mode='r')
    except (OSError, PathNotFoundError):
        root = zarr.hierarchy.open_group(filename, "w")
        root.array("data", **kwargs)
        yield zarr.convenience.open(filename, mode='r')


@pytest.fixture(scope='module')
def zarr_4d(tmpdir_factory):
    yield from get_or_create_zarr(tmpdir_factory, "zarr-test.zarr", data=np.ones((5, 5, 16, 16)))


@pytest.fixture(scope='module')
def zarr_2d(tmpdir_factory):
    yield from get_or_create_zarr(tmpdir_factory, "zarr-test-2d.zarr", data=np.ones((16, 16)))


@pytest.fixture(scope='module')
def zarr_3d(tmpdir_factory):
    yield from get_or_create_zarr(tmpdir_factory, "zarr-test-3d.zarr", data=np.ones((17, 16, 16)))


@pytest.fixture(scope='module')
def zarr_5d(tmpdir_factory):
    yield from get_or_create_zarr(tmpdir_factory, "zarr-test-5d.zarr",
                                  data=np.ones((3, 5, 9, 16, 16)))


@pytest.fixture(scope='module')
def random_zarr_large_sig(tmpdir_factory):
    yield from get_or_create_zarr(tmpdir_factory, "zarr-test-random.zarr",
                                  data=np.random.randn(16, 16, 512, 512))


@pytest.fixture(scope='module')
def random_zarr_4d(tmpdir_factory):
    yield from get_or_create_zarr(tmpdir_factory, "zarr-test-random.zarr",
                                  data=np.random.randn(5, 5, 16, 16))


@pytest.fixture(scope='module')
def zarr_4d_data():
    data = np.random.randn(2, 10, 26, 26).astype("float32")
    yield data


@pytest.fixture(scope='module')
def zarr_same_data_3d(tmpdir_factory, zarr_4d_data):
    data = zarr_4d_data.reshape((20, 26, 26))
    yield from get_or_create_zarr(tmpdir_factory, "zarr-test-reshape-3d.zarr", data=data)


@pytest.fixture(scope='module')
def zarr_same_data_4d(tmpdir_factory, zarr_4d_data):
    yield from get_or_create_zarr(tmpdir_factory, "zarr-test-reshape-4d.zarr", data=zarr_4d_data)


@pytest.fixture(scope='module')
def zarr_same_data_5d(tmpdir_factory, zarr_4d_data):
    data = zarr_4d_data.reshape((2, 2, 5, 26, 26))
    yield from get_or_create_zarr(tmpdir_factory, "zarr-test-reshape-5d.zarr", data=data)


@pytest.fixture(scope='module')
def zarr_same_data_1d_sig(tmpdir_factory, zarr_4d_data):
    data = zarr_4d_data.reshape((2, 10, 676))
    yield from get_or_create_zarr(tmpdir_factory, "zarr-test-reshape-1d-sig.zarr", data=data)


@pytest.fixture(scope='module')
def shared_random_data():
    return _mk_random(size=(16, 16, 256, 256), dtype='float32')


@pytest.fixture
def zarr_ds_1(zarr_4d, inline_executor):
    ds = ZarrDataSet(
        path=os.path.join(zarr_4d.store.path, "data")
    )
    ds = ds.initialize(inline_executor)
    return ds


@pytest.fixture
def zarr_ds_3d(zarr_3d, inline_executor):
    ds = ZarrDataSet(
        path=os.path.join(zarr_3d.store.path, "data")
    )
    ds = ds.initialize(inline_executor)
    return ds


@pytest.fixture
def zarr_ds_5d(zarr_5d, inline_executor):
    ds = ZarrDataSet(
        path=os.path.join(zarr_5d.store.path, "data")
    )
    ds = ds.initialize(inline_executor)
    return ds


def test_read_1(lt_ctx, zarr_4d):
    ds = ZarrDataSet(
        path=os.path.join(zarr_4d.store.path, "data"),
    )
    ds = ds.initialize(lt_ctx.executor)
    tileshape = Shape(
        (16,) + tuple(ds.shape.sig),
        sig_dims=ds.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds.shape,
    )
    for p in ds.get_partitions():
        for t in p.get_tiles(tiling_scheme=tiling_scheme):
            print(t.tile_slice)


def test_read_2(lt_ctx, zarr_4d):
    ds = ZarrDataSet(
        path=os.path.join(zarr_4d.store.path, "data"),
    )
    ds = ds.initialize(lt_ctx.executor)
    tileshape = Shape(
        (16,) + tuple(ds.shape.sig),
        sig_dims=ds.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds.shape,
    )
    for p in ds.get_partitions():
        for t in p.get_tiles(tiling_scheme=tiling_scheme):
            print(t.tile_slice)


def test_read_3(lt_ctx, random_zarr_4d):
    # try with smaller partitions:
    ds = ZarrDataSet(
        path=os.path.join(random_zarr_4d.store.path, "data"),
    )
    ds = ds.initialize(lt_ctx.executor)
    tileshape = Shape(
        (16,) + tuple(ds.shape.sig),
        sig_dims=ds.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds.shape,
    )
    for p in ds.get_partitions():
        for t in p.get_tiles(tiling_scheme=tiling_scheme):
            print(t.tile_slice)


def test_pick(random_zarr_4d, lt_ctx):
    ds = ZarrDataSet(
        path=os.path.join(random_zarr_4d.store.path, "data"),
    )
    ds = ds.initialize(lt_ctx.executor)
    assert len(ds.shape) == 4
    print(ds.shape)
    pick = lt_ctx.create_pick_analysis(dataset=ds, x=2, y=3)
    res = lt_ctx.run(pick)
    pick_frame = res.intensity.raw_data
    assert_allclose(random_zarr_4d["data"][3, 2, ...], pick_frame)


def test_roi_2(random_zarr_4d, lt_ctx):
    ds = ZarrDataSet(
        path=os.path.join(random_zarr_4d.store.path, "data"),
    )
    ds = ds.initialize(lt_ctx.executor)

    roi = {
        "shape": "disk",
        "cx": 2,
        "cy": 2,
        "r": 1,
    }
    analysis = SumAnalysis(dataset=ds, parameters={
        "roi": roi,
    })

    print(analysis.get_roi())

    results = lt_ctx.run(analysis)

    # let's draw a circle!
    mask = np.full((5, 5), False)
    mask[1, 2] = True
    mask[2, 1:4] = True
    mask[3, 2] = True

    print(mask)

    assert mask.shape == (5, 5)
    assert mask.dtype == bool

    # Zarr does not support indexing with a mask
    # unless it has the same shape as the dataset,
    # so we need numpy to mask-slice only the nav dims
    data = np.asarray(random_zarr_4d["data"])

    # applying the mask flattens the first two dimensions, so we
    # only sum over axis 0 here:
    expected = data[mask, ...].sum(axis=(0,))

    assert expected.shape == (16, 16)
    assert results.intensity.raw_data.shape == (16, 16)

    # is not equal to results without mask:
    assert not np.allclose(results.intensity.raw_data, data.sum(axis=(0, 1)))
    # ... but rather like `expected`:
    assert np.allclose(results.intensity.raw_data, expected)


@pytest.mark.parametrize('chunks', [
    (1, 3, 16, 16),
    (1, 6, 16, 16),
    (1, 4, 16, 16),
    (1, 16, 16, 16),
])
def test_chunked(lt_ctx, tmpdir_factory, chunks):
    datadir = tmpdir_factory.mktemp('data')
    filename = os.path.join(datadir, 'zarr-test-chunked.zarr')
    data = _mk_random((16, 16, 16, 16), dtype=np.float32)

    root = zarr.hierarchy.open_group(filename, "w")
    root.array("data", data=data, chunks=chunks)

    ds = lt_ctx.load("zarr", path=os.path.join(filename, "data"))
    udf = PixelsumUDF()
    res = lt_ctx.run_udf(udf=udf, dataset=ds)['pixelsum']
    assert np.allclose(
        res,
        np.sum(data, axis=(2, 3))
    )


@pytest.mark.parametrize('udf', [
    SumSigUDF(),
    AutoUDF(f=lambda frame: frame.sum()),
])
@pytest.mark.parametrize('chunks', [
    (3, 3, 32, 32),
    (3, 6, 32, 32),
    (3, 4, 32, 32),
    (1, 4, 32, 32),
    (1, 16, 32, 32),

    (3, 3, 256, 256),
    (3, 6, 256, 256),
    (3, 4, 256, 256),
    (1, 4, 256, 256),
    (1, 16, 256, 256),

    (3, 3, 128, 256),
    (3, 6, 128, 256),
    (3, 4, 128, 256),
    (1, 4, 128, 256),
    (1, 16, 128, 256),

    (3, 3, 32, 128),
    (3, 6, 32, 128),
    (3, 4, 32, 128),
    (1, 4, 32, 128),
    (1, 16, 32, 128),
])
def test_chunked_weird(lt_ctx, tmpdir_factory, chunks, udf, shared_random_data):
    datadir = tmpdir_factory.mktemp('data')
    filename = os.path.join(datadir, 'weirdly-chunked-256-256.zarr')
    data = shared_random_data

    root = zarr.hierarchy.open_group(filename, "w")
    root.array("data", data=data, chunks=chunks)
    ds = lt_ctx.load("zarr", path=os.path.join(filename, "data"))

    base_shape = ds.get_base_shape(roi=None)
    print(base_shape)

    res = lt_ctx.run_udf(dataset=ds, udf=udf)
    assert len(res) == 1
    res = next(iter(res.values()))
    assert_allclose(
        res,
        np.sum(data, axis=(2, 3))
    )

    shutil.rmtree(filename)


@pytest.mark.parametrize('in_dtype', [
    np.float32,
    np.float64,
    np.uint16,
])
@pytest.mark.parametrize('read_dtype', [
    np.float32,
    np.float64,
    np.uint16,
])
@pytest.mark.parametrize('use_roi', [
    True, False
])
def test_zarr_result_dtype(lt_ctx, tmpdir_factory, in_dtype, read_dtype, use_roi):
    datadir = tmpdir_factory.mktemp('data')
    filename = os.path.join(datadir, 'result-dtype-checks.zarr')
    data = _mk_random((2, 2, 4, 4), dtype=in_dtype)

    root = zarr.hierarchy.open_group(filename, "w")
    root.array("data", data=data)
    ds = lt_ctx.load("zarr", path=os.path.join(filename, "data"))

    if use_roi:
        roi = np.zeros(ds.shape.nav, dtype=bool).reshape((-1,))
        roi[0] = 1
    else:
        roi = None
    udfs = [SumSigUDF()]  # need to have at least one UDF
    p = next(ds.get_partitions())
    neg = Negotiator()
    tiling_scheme = neg.get_scheme(
        udfs=udfs,
        dataset=ds,
        approx_partition_shape=p.shape,
        read_dtype=read_dtype,
        roi=roi,
        corrections=None,
    )
    tile = next(p.get_tiles(tiling_scheme=tiling_scheme, roi=roi, dest_dtype=read_dtype))
    assert tile.dtype == np.dtype(read_dtype)


class UDFWithLargeDepth(UDF):
    def process_tile(self, tile):
        pass

    def get_tiling_preferences(self):
        return {
            "depth": 128,
            "total_size": UDF.TILE_SIZE_BEST_FIT,
        }


def test_zarr_tileshape_negotation(lt_ctx, tmpdir_factory):
    # try to hit the third case in _get_subslices:
    datadir = tmpdir_factory.mktemp('data')
    filename = os.path.join(datadir, 'tileshape-neg-test.zarr')
    data = _mk_random((4, 100, 256, 256), dtype=np.uint16)

    root = zarr.hierarchy.open_group(filename, "w")
    root.array("data", data=data, chunks=(2, 32, 32, 32))
    ds = lt_ctx.load("zarr", path=os.path.join(filename, "data"))

    udfs = [UDFWithLargeDepth()]
    p = next(ds.get_partitions())
    neg = Negotiator()
    tiling_scheme = neg.get_scheme(
        udfs=udfs,
        dataset=ds,
        approx_partition_shape=p.shape,
        read_dtype=np.float32,
        roi=None,
        corrections=None,
    )
    assert len(tiling_scheme) > 1
    next(p.get_tiles(tiling_scheme=tiling_scheme, roi=None, dest_dtype=np.float32))


def test_scheme_too_large(zarr_ds_1):
    partitions = zarr_ds_1.get_partitions()
    p = next(partitions)
    depth = p.shape[0]

    # we make a tileshape that is too large for the partition here:
    tileshape = Shape(
        (depth + 1,) + tuple(zarr_ds_1.shape.sig),
        sig_dims=zarr_ds_1.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=zarr_ds_1.shape,
    )

    tiles = p.get_tiles(tiling_scheme=tiling_scheme)
    t = next(tiles)
    assert t.tile_slice.shape[0] <= zarr_ds_1.shape[0]


def test_zarr_macrotile(lt_ctx, tmpdir_factory):
    datadir = tmpdir_factory.mktemp('data')
    filename = os.path.join(datadir, 'macrotile-1.zarr')
    data = _mk_random((128, 128, 4, 4), dtype=np.float32)

    root = zarr.hierarchy.open_group(filename, "w")
    root.array("data", data=data)
    ds = lt_ctx.load("zarr", path=os.path.join(filename, "data"))

    ds.set_num_cores(4)

    partitions = ds.get_partitions()
    p0 = next(partitions)
    m0 = p0.get_macrotile()
    assert m0.tile_slice.origin == (0, 0, 0)
    assert m0.tile_slice.shape == p0.shape

    p1 = next(partitions)
    m1 = p1.get_macrotile()
    assert m1.tile_slice.origin == (p0.shape[0], 0, 0)
    assert m1.tile_slice.shape == p1.shape


def test_zarr_macrotile_roi(lt_ctx, zarr_ds_1):
    roi = np.random.choice(size=zarr_ds_1.shape.flatten_nav().nav, a=[True, False])
    data = zarr_ds_1.get_array(load=True)
    expected = data.reshape(zarr_ds_1.shape.flatten_nav())[roi]
    partitions = zarr_ds_1.get_partitions()
    p0 = next(partitions)
    m0 = p0.get_macrotile(roi=roi)
    assert_allclose(
        m0.data,
        expected
    )


def test_zarr_macrotile_empty_roi(lt_ctx, zarr_ds_1):
    roi = np.zeros(zarr_ds_1.shape.flatten_nav().nav, dtype=bool)
    partitions = zarr_ds_1.get_partitions()
    p0 = next(partitions)
    m0 = p0.get_macrotile(roi=roi)
    assert m0.shape == (0, 16, 16)
    assert_allclose(
        m0.data,
        0,
    )


def test_zarr_apply_masks_1(lt_ctx, zarr_ds_1):
    mask = _mk_random(size=(16, 16))
    data = zarr_ds_1.get_array(load=True)
    expected = _naive_mask_apply([mask], data)
    analysis = lt_ctx.create_mask_analysis(
        dataset=zarr_ds_1, factories=[lambda: mask]
    )
    results = lt_ctx.run(analysis)

    assert np.allclose(
        results.mask_0.raw_data,
        expected
    )


def test_zarr_3d_apply_masks(lt_ctx, zarr_ds_3d):
    mask = _mk_random(size=(16, 16))
    data = zarr_ds_3d.get_array(load=True)
    expected = _naive_mask_apply([mask], data.reshape((1, 17, 16, 16)))
    analysis = lt_ctx.create_mask_analysis(
        dataset=zarr_ds_3d, factories=[lambda: mask]
    )
    results = lt_ctx.run(analysis)

    assert np.allclose(
        results.mask_0.raw_data,
        expected
    )


def test_zarr_5d_apply_masks(lt_ctx, zarr_ds_5d):
    mask = _mk_random(size=(16, 16))
    data = zarr_ds_5d.get_array(load=True)
    expected = _naive_mask_apply([mask], data.reshape((1, 135, 16, 16))).reshape((3, 5, 9))
    analysis = lt_ctx.create_mask_analysis(
        dataset=zarr_ds_5d, factories=[lambda: mask]
    )
    results = lt_ctx.run(analysis)

    print(results.mask_0.raw_data.shape, expected.shape)

    assert np.allclose(
        results.mask_0.raw_data,
        expected
    )
