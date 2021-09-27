import numpy as np

from libertem.common.slice import Slice
from libertem.common.shape import Shape
from libertem.io.dataset.base.tiling import DataTile


def slice_to_tl(slice_obj):
    return tuple(s.start if s.start is not None else 0 for s in slice_obj)


class TileReciever(object):
    def __init__(self, ds_meta):
        self.ds_meta = ds_meta

    def wrap_tile(self, frame_slice, frame):
        scheme_idx = 0
        tile_slice = Slice(slice_to_tl(frame_slice), shape=Shape(frame.shape, self.ds_meta.sig_dims))
        return DataTile(frame, tile_slice, scheme_idx)
