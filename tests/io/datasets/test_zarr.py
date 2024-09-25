import os

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
def random_zarr(tmpdir_factory):
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


def test_read_3(lt_ctx, random_zarr):
    # try with smaller partitions:
    ds = ZarrDataSet(
        path=os.path.join(random_zarr.store.path, "data"),
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


def test_pick(random_zarr, lt_ctx):
    ds = ZarrDataSet(
        path=os.path.join(random_zarr.store.path, "data"),
    )
    ds = ds.initialize(lt_ctx.executor)
    assert len(ds.shape) == 4
    print(ds.shape)
    pick = lt_ctx.create_pick_analysis(dataset=ds, x=2, y=3)
    res = lt_ctx.run(pick)
    pick_frame = res.intensity.raw_data
    assert_allclose(random_zarr["data"][3, 2, ...], pick_frame)


def test_roi_2(random_zarr, lt_ctx):
    ds = ZarrDataSet(
        path=os.path.join(random_zarr.store.path, "data"),
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
    data = np.asarray(random_zarr["data"])

    # applying the mask flattens the first two dimensions, so we
    # only sum over axis 0 here:
    expected = data[mask, ...].sum(axis=(0,))

    assert expected.shape == (16, 16)
    assert results.intensity.raw_data.shape == (16, 16)

    # is not equal to results without mask:
    assert not np.allclose(results.intensity.raw_data, data.sum(axis=(0, 1)))
    # ... but rather like `expected`:
    assert np.allclose(results.intensity.raw_data, expected)
