import numpy as np

class MaskedArray(np.ndarray):
    def __new__(cls, array, mask):
        obj = np.asarray(array).view(cls)
        obj.mask = mask
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.mask = getattr(obj, 'mask', None)


if __name__ == '__main__':
    array = np.arange(64).reshape(8, 8)
    mask = np.ones_like(array, dtype=bool)
    mask[:, 2] = False

    marr = np.ma.masked_array(array, mask=mask)