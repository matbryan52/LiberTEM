import os
import warnings
import pathlib
import numpy as np
import pandas as pd
import mmap
from scipy.interpolate import interp1d

from libertem.web.messages import MessageConverter
from libertem.common import Shape
from .raw import RawFileDataSet
from .base import DataSetMeta, LocalFile, FileSet, DataSetException


"""
Assuming constant structure of PrecessionNotes.txt
Assuming files are in C-order i.e. framewise horizontal increasing fastest
Assuming bin files are C-order, 10 byte header
Assuming binfile header is 10 = (2 bytes) + (2x int32) encoding height, width
Assuming frames are all identical size
Not handling variability of scan coordinates
"""


class CEAPrecessionDatasetParams(MessageConverter):
    SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "http://libertem.org/CEAPrecessionDatasetParams.schema.json",
        "title": "CEAPrecessionDatasetParams",
        "type": "object",
        "properties": {
            "type": {"const": "CEAPREC"},
            "nav_shape": {
                "type": "array",
                "items": {"type": "number", "minimum": 1},
                "minItems": 2,
                "maxItems": 2
            },
            "sig_shape": {
                "type": "array",
                "items": {"type": "number", "minimum": 1},
                "minItems": 2,
                "maxItems": 2
            },
            "sync_offset": {"type": "number"},
            "enable_direct": {
                "type": "boolean"
            },
        },
        "required": ["type", "path", "nav_shape", "sig_shape", "dtype"]
    }

    def convert_to_python(self, raw_data):
        print('REACHED CONVERT')
        data = {
            k: raw_data[k]
            for k in ["path", "dtype", "nav_shape", "sig_shape", "enable_direct"]
        }
        if "sync_offset" in raw_data:
            data["sync_offset"] = raw_data["sync_offset"]
        return data


class PrecessionNotes(object):
    def __init__(self, filepath):
        self.root = filepath.parent
        self.headers, self.coordinates = self.load_image_notes(filepath)
        self.infer_grid_idcs()

    def __getitem__(self, idx):
        return self._fname_to_fpath(self.coordinates.fname.iloc[idx])

    def __iter__(self):
        for _, row in self.coordinates.iterrows():
            yield self._fname_to_fpath(row.fname)

    def __len__(self):
        return len(self.coordinates.index)

    def _fname_to_fpath(self, fname):
        return self.root / f'{fname}.bin'

    @property
    def nav_xdim(self):
        return self.headers['Map size x']

    @property
    def nav_ydim(self):
        return self.headers['Map size y']

    @property
    def nav_shape(self):
        return (self.nav_ydim, self.nav_xdim)

    @property
    def pixel_size(self):
        return self.headers['extra'][3]  # 'Image pixel size (m-1)'

    @property
    def exposure_time(self):
        return self.headers['extra'][2]  # 'Exposure time (s)'

    def load_image_notes(self, filepath):
        header_rows = 11
        dataframe = pd.read_csv(filepath, sep='\t', header=None,
                                names=['fname', 'xpos', 'ypos'],
                                skipinitialspace=True, skiprows=header_rows)
        dataframe = dataframe[:-1]
        with filepath.open() as fp:
            headers = [next(fp) for _ in range(header_rows)]

        headers = self.parse_headers(headers)
        return headers, dataframe

    def parse_headers(self, headers):
        headers = [h.replace('\n', '') for h in headers]
        headers = [h.strip() for h in headers]
        headers = [h for h in headers if len(h) > 0]
        headers = [h.replace('\t', '=') for h in headers]
        odict = {'extra': []}

        for h in headers:
            hsplit = h.split('=')
            if len(hsplit) > 1:
                odict[hsplit[0].strip()] = hsplit[1].strip()
            else:
                odict['extra'].append(h)

        for k, v in odict.items():
            try:
                if float(v) % 1 == 0:
                    odict[k] = int(v)
                else:
                    odict[k] = float(v)
            except (ValueError, TypeError):
                pass

        return odict

    def infer_grid_idcs(self):
        """
        Assumes we have a cartesian scan.
        Points will be labelled such that
        xidx increases horizontally right and
        yidx increases vertically down

        Dataframe is re-sorted to ensure a C-ordering of files
        This ensures any UDF results follow the correct
        indexing as the FileSet is constructed directly
        from the ordering of the coordinates dataframe
        """
        ny, nx = self.nav_shape
        xmin, ymin, xmax, ymax = self.scan_bounds
        sample_x_coords = np.linspace(xmin, xmax, num=nx, endpoint=True)
        sample_y_coords = np.linspace(ymin, ymax, num=ny, endpoint=True)
        xinterp = interp1d(sample_x_coords, np.arange(nx), kind='nearest')
        yinterp = interp1d(sample_y_coords, np.arange(ny), kind='nearest')
        self.coordinates['xidx'] = xinterp(self.coordinates.xpos.array).astype(int)
        self.coordinates['yidx'] = yinterp(self.coordinates.ypos.array).astype(int)
        self.coordinates = self.coordinates.sort_values(by=['yidx', 'xidx'], ignore_index=True)
        if not (self.coordinates.groupby(['xidx', 'yidx']).size() == 1).all():
            warnings.warn(('Assigning array indices to scan points found '
                           'duplicate coordinates. Consider inspecting the '
                           'coordinate dataframe.'))

class CEAPrecessionDataset(RawFileDataSet):
    _bin_header_bytes = 10

    def __init__(self, path, dtype=np.int32, nav_shape=None, sig_shape=None,
                io_backend=None, enable_direct=False, sync_offset=0,):
        super(RawFileDataSet, self).__init__(io_backend=io_backend)

        self._path = pathlib.Path(path)
        self._dtype = np.dtype(dtype).type
        self._sync_offset = sync_offset
        self._enable_direct = enable_direct

        self._nav_shape = None
        self._sig_shape = None
        self._sig_dims = None
        self._filesize = None

    def initialize(self, executor):
        self._prec_ds_meta = self._parse_meta(self._path)
        self._nav_shape = self._prec_ds_meta.nav_shape
        self._sig_shape = executor.run_function(self._infer_sig_shape, self._prec_ds_meta[0])
        self._sig_dims = len(self._sig_shape)
        self._filesize = executor.run_function(self._get_filesize, self._prec_ds_meta[0])

        self._nav_shape_product = int(np.prod(self._nav_shape))
        self._image_count = self._nav_shape_product
        self._sync_offset_info = self.get_sync_offset_info()
        shape = Shape(self._nav_shape + self._sig_shape, sig_dims=self._sig_dims)
        self._meta = DataSetMeta(
            shape=shape,
            raw_dtype=np.dtype(self._dtype),
            sync_offset=self._sync_offset,
            image_count=self._image_count,
        )
        return self

    @property
    def precession_meta(self):
        try:
            return self._prec_ds_meta
        except AttributeError:
            warnings.warn('Must initialize dataset before metadata is available')

    @classmethod
    def _parse_meta(cls, path):
        assert path.suffix == '.txt'
        return PrecessionNotes(path)

    @staticmethod
    def _split_files_by_filetype(filelist):
        files = {}
        for f in filelist:
            try:
                files[f.suffix].append(f)
            except KeyError:
                files[f.suffix.replace('.', '')] = [f]
        return files

    @classmethod
    def get_supported_extensions(cls):
        return {'txt', 'bin'}

    @classmethod
    def _infer_sig_shape(self, binpath):
        """
        Must infer from a frame header
        """
        with binpath.open('rb') as fp:
            header = fp.read(self._bin_header_bytes)
        # Can't quite figure out what header bytes [0, 1] represent
        # Assuming that frame_size is encoded as height, width but
        # I don't actually have any counterexamples, it could be w, h!
        sig_shape = np.frombuffer(header[2:], np.int32)
        return tuple(sig_shape.tolist())

    def _get_filesize(self, binpath):
        """
        Returns filesize of a single frame, but need to check
        where this value is actually used and how
        """
        with binpath.open('rb') as fp:
            bin_buffer = fp.read()
        return len(bin_buffer[self._bin_header_bytes:])

    def __repr__(self):
        return f"<{self.__class__.__name__} of {self.dtype} shape={self.shape}>"

    def _get_fileset(self):
        return CEAPrecessionFileSet([
            CEAPrecessionFile(
                path=f,
                start_idx=fidx,
                end_idx=fidx+1,
                sig_shape=self.shape.sig,
                native_dtype=self._meta.raw_dtype,
                frame_header=self._bin_header_bytes
            )
            for fidx, f in enumerate(self._prec_ds_meta)])

    def check_valid(self):
        if self._enable_direct and not hasattr(os, 'O_DIRECT'):
            raise DataSetException("LiberTEM currently only supports Direct I/O on Linux")
        try:
            fileset = self._get_fileset()
            for f in fileset:
                with f:
                    pass
            return True
        except (OSError, ValueError) as e:
            raise DataSetException("invalid dataset: %s" % e)

    @classmethod
    def detect_params(cls, path, executor):
        _path = pathlib.Path(path)
        if _path.name == 'PrecessionImageNotes.txt':
            prec_ds_meta = cls._parse_meta(_path)
            nav_shape = prec_ds_meta.nav_shape
            sig_shape = executor.run_function(cls._infer_sig_shape, prec_ds_meta[0])
        else:
            return False
        return {
            "parameters": {
                "path": str(path),
                "nav_shape": nav_shape,
                "sig_shape": sig_shape,
                "dtype": 'int32'
            },
            "info": {
                "image_count": len(prec_ds_meta),
                "native_sig_shape": sig_shape,
            }
        }

    @classmethod
    def get_msg_converter(cls):
        return CEAPrecessionDatasetParams


class CEAPrecessionFile(LocalFile):
    def open(self):
        f = open(self._path, "rb")
        self._file = f
        self._raw_mmap = mmap.mmap(
            fileno=f.fileno(),
            length=0,
            offset=0,
            access=mmap.ACCESS_READ,
        )
        # TODO: self._raw_mmap.madvise(mmap.MADV_HUGEPAGE) - benchmark this!
        self._raw_mmap = memoryview(self._raw_mmap)[self._frame_header:]
        self._mmap = self._mmap_to_array(self._raw_mmap)

    def _mmap_to_array(self, raw_mmap):
        """
        Create an array from the raw memory map, stripping away
        frame headers and footers

        Parameters
        ----------

        raw_mmap : np.memmap or memoryview
            The raw memory map, with the file header already stripped away

        start : int
            Number of items cut away at the start of each frame (frame_header // itemsize)

        stop : int
            Number of items per frame (something like start + np.prod(sig_shape))
        """
        return np.frombuffer(raw_mmap, dtype=self._native_dtype).reshape(self.num_frames, -1)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()


class CEAPrecessionFileSet(FileSet):
    pass
