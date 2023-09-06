from libertem.udf.base import UDF
from libertem.io.dataset.raw import RawFileDataSet
import libertem


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
