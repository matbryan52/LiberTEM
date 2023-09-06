from libertem.udf.base import UDF
from libertem.io.dataset.raw import RawFileDataSet
import libertem

import pyfftw

class NeedsFrame(UDF):
    def get_result_buffers(self):
        return {}
    
    def process_frame(self, frame):
        pass


class DoesFFTStack(UDF):
    def get_backends(self):
        return (UDF.BACKEND_NUMPY, UDF.BACKEND_CUPY)

    def get_result_buffers(self):
        return {}
    
    def process_tile(self, tile):
        try:
            if self._processed_tiles < 0:
                raise RuntimeError('Multiple tiles with less than stack_size')
            self._processed_tiles += 1
            assert tile.shape[0] == self.params.get('stack_size')
        except AttributeError:
            self._processed_tiles = 1
            assert tile.shape[0] == self.params.get('stack_size')
        except AssertionError:
            # Should be last tile, will not be called again
            assert self._processed_tiles > 0
            self._processed_tiles = -1
        if self.params.get('use_pyfftw', False):
            stack_fft = pyfftw.interfaces.scipy_fftpack.fft2(tile)
        else:
            stack_fft = self.xp.fft.fft2(tile)
        try:
            stack_fft.get()
        except AttributeError:
            pass


class DoesFFTFrame(UDF):
    def get_backends(self):
        return (UDF.BACKEND_NUMPY, UDF.BACKEND_CUPY)

    def get_result_buffers(self):
        return {}
    
    def process_frame(self, frame):
        if self.params.get('use_pyfftw', False):
            stack_fft = pyfftw.interfaces.scipy_fftpack.fft2(frame)
        else:
            stack_fft = self.xp.fft.fft2(frame)
        try:
            stack_fft.get()
        except AttributeError:
            pass


class RawDataSetMod(RawFileDataSet):
    def __init__(self, *args, tile_depth=1, **kwargs):
        super().__init__(*args, **kwargs)
        self._tile_depth = tile_depth
        
    def adjust_tileshape(self, tileshape, roi):
        return (self._tile_depth,) +  self.meta.shape.sig.to_tuple()


libertem.io.dataset.filetypes['raw_m'] = RawDataSetMod
