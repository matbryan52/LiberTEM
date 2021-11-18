import os
import pathlib
import typing
import logging

from ncempy.io.dm import fileDM
import numpy as np

from libertem.common.math import prod
from libertem.common import Shape
from libertem.web.messageconverter import MessageConverter
from .base import (
    DataSet, FileSet, BasePartition, DataSetException, DataSetMeta, File,
)
from libertem.io.dataset.base.backend_mmap import MMapFile, MMapBackend, MMapBackendImpl

log = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from numpy import typing as nt


def get_memmap(filepath, dataset_number=0, transpose_signav=True):
    fileobj = fileDM(filepath, on_memory=False)
    memmap = fileobj.getMemmap(dataset_number)
    if memmap.ndim != 4:
        raise DataSetException('No support for non-4D DM4 files yet')
    if transpose_signav:
        memmap = np.transpose(memmap, (2, 3, 0, 1))
    return memmap


class DM4MMapFile(MMapFile):
    """
    Override standard memmap creation, replace with ncempy.io implem
    """
    def open(self):
        filepath = self.desc.path
        ds_num = self.desc._dataset_number
        do_tranpose = self.desc._tranpose_signav
        self._mmap = get_memmap(filepath,
                                dataset_number=ds_num,
                                transpose_signav=do_tranpose)
        self._arr = self._mmap
        return self

    def close(self):
        del self._arr
        del self._mmap


class DM4Backend(MMapBackend):
    def get_impl(self):
        return DM4BackendImpl()


class DM4BackendImpl(MMapBackendImpl):
    FILE_CLS = DM4MMapFile


class DM4DatasetParams(MessageConverter):
    SCHEMA: typing.Dict = {}

    def convert_from_python(self, raw_data):
        return super().convert_from_python(raw_data)

    def convert_to_python(self, raw_data):
        return super().convert_to_python(raw_data)


class DM4File(File):
    def __init__(self, *args, dataset_number=0, transpose_signav=True, **kwargs):
        self._dataset_number = dataset_number
        self._transpose_signav = transpose_signav
        super().__init__(*args, **kwargs)


class DMFileSet(FileSet):
    pass


class DM4DataSet(DataSet):
    """
    Reader for stacks of DM3/DM4 files.

    Note
    ----
    This DataSet is not supported in the GUI yet, as the file dialog needs to be
    updated to `properly handle opening series
    <https://github.com/LiberTEM/LiberTEM/issues/498>`_.

    Note
    ----
    Single-file 4D DM files are not yet supported. The use-case would be
    to read DM4 files from the conversion of K2 STEMx data, but those data sets
    are actually transposed (nav/sig are swapped).

    That means the data would have to be transposed back into the usual shape,
    which is slow, or algorithms would have to be adapted to work directly on
    transposed data. As an example, applying a mask in the conventional layout
    corresponds to calculating a weighted sum frame along the navigation
    dimension in the transposed layout.

    Since the transposed layout corresponds to a TEM tilt series, support for
    transposed 4D STEM data could have more general applications beyond
    supporting 4D DM4 files. Please contact us if you have a use-case for
    single-file 4D DM files or other applications that process stacks of TEM
    files, and we may add support!

    Note
    ----
    You can use the PyPi package `natsort <https://pypi.org/project/natsort/>`_
    to sort the filenames by their numerical components, this is especially useful
    for filenames without leading zeros.

    Parameters
    ----------

    files : List[str]
        List of paths to the files that should be loaded. The order is important,
        as it determines the order in the navigation axis.

    nav_shape : Tuple[int] or None
        By default, the files are loaded as a 3D stack. You can change this
        by specifying the nav_shape, which reshapes the navigation dimensions.
        Raises a `DataSetException` if the shape is incompatible with the data
        that is loaded.

    sig_shape: Tuple[int], optional
        Signal/detector size (height, width)

    sync_offset: int, optional
        If positive, number of frames to skip from start
        If negative, number of blank frames to insert at start

    same_offset : bool
        When reading a stack of dm3/dm4 files, it can be expensive to read in
        all the metadata from all files, which we currently only use for
        getting the offsets and sizes of the main data in each file. If you
        absolutely know that the offsets and sizes are the same for all files,
        you can set this parameter and we will skip reading all metadata but
        the one from the first file.
    """
    def __init__(self, file=None, dataset_number=0, transpose_signav=True, io_backend=None):
        super().__init__(io_backend=io_backend)
        self._meta = None
        self._filesize = None
        if not isinstance(file, (str, pathlib.Path)):
            raise DataSetException("file argument must be path to a dm4 dataset")
        self._file = pathlib.Path(file)
        if self._file.suffix != '.dm4':
            raise DataSetException("file must be a dm4 file")
        self._dataset_number = dataset_number
        self._transpose_signav = transpose_signav
        self._fileset = None

    def _get_ds_info(self):
        memmap = get_memmap(self._file,
                            dataset_number=self._dataset_number,
                            transpose_signav=self._transpose_signav)
        raw_dtype = memmap.dtype
        return Shape(memmap.shape, sig_dims=2), raw_dtype

    def _get_fileset(self):
        f = DM4File(
            path=self._file,
            start_idx=0,
            end_idx=self._nav_shape_product,
            sig_shape=self._meta.shape.sig,
            native_dtype=self._meta.raw_dtype,
            file_header=0,
            dataset_number=self._dataset_number,
            transpose_signav=self._transpose_signav
        )
        return DMFileSet([f])

    def _get_filesize(self):
        return os.stat(self._file).st_size

    def initialize(self, executor):
        ds_shape, raw_dtype = self._get_ds_info()

        self._nav_shape = ds_shape.nav
        self._sig_shape = ds_shape.sig
        self._filesize = executor.run_function(self._get_filesize)

        self._nav_shape_product = int(prod(self._nav_shape))
        self._image_count = self._nav_shape_product
        self._sync_offset_info = self.get_sync_offset_info()
        self._meta = DataSetMeta(
            shape=ds_shape,
            raw_dtype=raw_dtype,
            sync_offset=self._sync_offset,
            image_count=self._image_count,
        )
        self._fileset = executor.run_function(self._get_fileset)
        return self

    @classmethod
    def get_supported_extensions(cls):
        return {"dm4"}

    @classmethod
    def get_msg_converter(cls) -> typing.Type[MessageConverter]:
        return DM4DatasetParams

    @classmethod
    def detect_params(cls, path, executor):
        # FIXME: this doesn't really make sense for file series
        pl = path.lower()
        if pl.endswith(".dm4"):
            return {
                "parameters": {
                    "file": path
                },
            }
        return False

    @property
    def dtype(self) -> "nt.DTypeLike":
        return self._meta.raw_dtype

    @property
    def shape(self):
        return self._meta.shape

    def check_valid(self):
        try:
            with fileDM(self._file, on_memory=False):
                pass
            return True
        except OSError as e:
            raise DataSetException("invalid dataset: %s" % e)

    def get_partitions(self):
        for part_slice, start, stop in self.get_slices():
            yield BasePartition(
                meta=self._meta,
                partition_slice=part_slice,
                fileset=self._fileset,
                start_frame=start,
                num_frames=stop - start,
                io_backend=self.get_io_backend(),
            )

    def __repr__(self):
        try:
            return f"<DM4DataSet of shape {self._meta.shape}>"
        except AttributeError:
            return "<Unitialized DM4DataSet>"
