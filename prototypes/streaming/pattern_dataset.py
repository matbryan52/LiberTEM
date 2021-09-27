import pathlib
import numpy as np
import matplotlib.pyplot as plt

def make_frame(frame_size, nstripes, start_val=0., inc_val=0.1, dtype=np.float32):
    frame = np.zeros((frame_size, frame_size), dtype=dtype)
    assert frame_size % nstripes == 0
    skip = frame_size // nstripes
    val = start_val
    for yidx in range(0, frame_size, 2*skip):
        for xidx in range(0, frame_size, 2*skip):
            frame[yidx:yidx+skip, xidx:xidx+skip] = val
            val += inc_val
    return frame


if __name__ == '__main__':
    nav_shape = (20, 150)
    nframes = np.prod(nav_shape)

    dtype = np.float32
    frame_size = 256
    nstripes = 8
    base_frame = make_frame(frame_size, nstripes, dtype=dtype)

    with pathlib.Path('./stripes.raw').open('wb') as fp:
        for fidx in range(nframes):
            frame = base_frame.copy() * float(fidx)
            fp.write(frame.astype(dtype).data)
            print(fidx)
