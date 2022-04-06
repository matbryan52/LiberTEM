import warnings
import pathlib
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from libertem.web.messages import MessageConverter
from .raw_group import RawFileGroupDataSet
from .base import IOBackend


class PrecessionDatasetParams(MessageConverter):
    SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "http://libertem.org/PrecessionDatasetParams.schema.json",
        "title": "PrecessionDatasetParams",
        "type": "object",
        "properties": {
            "type": {"const": "PREC"},
            "path": {"type": "string"},
            "sync_offset": {"type": "number"},
            "io_backend": {
                "enum": IOBackend.get_supported(),
            },
        },
        "required": ["type", "path"]
    }


class PrecessionNotes:
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
    def paths(self):
        return [*self]

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

    @property
    def _origin_xy(self):
        try:
            return self._origin_xy_value
        except AttributeError:
            topleft = self.coordinates.groupby(['xidx', 'yidx']).get_group((0, 0))
            self._origin_xy_value = topleft.xpos.min(), topleft.ypos.min()
            return self._origin_xy

    @property
    def _scale_xy(self):
        try:
            return self._scale_xy_value
        except AttributeError:
            yscale = self.coordinates.groupby('xidx').ypos.diff().dropna().mean()
            xscale = self.coordinates.groupby('yidx').xpos.diff().dropna().mean()
            if np.isnan(xscale) and np.isnan(yscale):
                warnings.warn('Unexpected single frame scan? Cannot infer coordinate system')
                xscale = yscale = 1.
            elif np.isnan(yscale):
                yscale = xscale
            elif np.isnan(xscale):
                xscale = yscale
            self._scale_xy_value = xscale, yscale
            return self._scale_xy

    @staticmethod
    def _convert_metres(iterable, unit='nm'):
        unit_mapping = {'nm': 1e9, 'um': 1e6, 'μ': 1e6, 'a': 1e10, 'Å': 1e10}
        if unit not in unit_mapping.keys():
            raise ValueError(f'Unrecognized unit {unit}, accepted are {unit_mapping.keys()}')
        return tuple(map(lambda x: x * unit_mapping[unit], iterable))

    def origin_xy(self, unit='nm'):
        return self._convert_metres(self._origin_xy, unit), unit

    def scale_xy(self, unit='nm'):
        return self._convert_metres(self._scale_xy, unit), unit

    @property
    def scan_bounds(self):
        xmin, xmax = self.coordinates['xpos'].min(), self.coordinates['xpos'].max()
        ymin, ymax = self.coordinates['ypos'].min(), self.coordinates['ypos'].max()
        return xmin, ymin, xmax, ymax

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
        if nx > 1:
            sample_x_coords = np.linspace(xmin, xmax, num=nx, endpoint=True)
            xinterp = interp1d(sample_x_coords, np.arange(nx), kind='nearest')
            self.coordinates['xidx'] = xinterp(self.coordinates.xpos.array).astype(int)
        else:
            self.coordinates['xidx'] = 0
        if ny > 1:
            sample_y_coords = np.linspace(ymin, ymax, num=ny, endpoint=True)
            yinterp = interp1d(sample_y_coords, np.arange(ny), kind='nearest')
            self.coordinates['yidx'] = yinterp(self.coordinates.ypos.array).astype(int)
        else:
            self.coordinates['yidx'] = 0
        self.coordinates = self.coordinates.sort_values(by=['yidx', 'xidx'], ignore_index=True)
        if not (self.coordinates.groupby(['xidx', 'yidx']).size() == 1).all():
            warnings.warn('Assigning array indices to scan points found '
                          'duplicate coordinates. Consider inspecting the '
                          'coordinate dataframe.')


class PrecessionDataSet(RawFileGroupDataSet):
    _bin_header_bytes = 10

    def __init__(self, notes_path, sync_offset=0, io_backend=None):
        prec_ds_meta = self._parse_meta(notes_path)
        nav_shape = prec_ds_meta.nav_shape
        sig_shape = self._infer_sig_shape(prec_ds_meta[0])
        paths = prec_ds_meta.paths
        super().__init__(paths,
                         dtype=np.int32,
                         nav_shape=nav_shape,
                         sig_shape=sig_shape,
                         file_heade=self._bin_header_bytes,
                         sync_offset=sync_offset,
                         io_backend=io_backend)

    @classmethod
    def _parse_meta(cls, path):
        assert path.suffix == '.txt'
        return PrecessionNotes(path)

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

    def __repr__(self):
        return f"<{self.__class__.__name__} of {self.dtype} shape={self.shape}>"

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
        return PrecessionDatasetParams
