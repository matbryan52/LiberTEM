import numpy as np

from libertem.api import Context
from libertem.executor.delayed import DelayedJobExecutor
from libertem.udf.sum import SumUDF  # No CUDA support in this one


if __name__ == '__main__':
    # Make a LiberTEM Dask cluster and set it as default for Dask
    ctx = Context.make_with("dask-make-default")
    del_ctx = Context(executor=DelayedJobExecutor())

    ds = ctx.load('memory', data=np.zeros((32, 32, 64, 64)))

    # No problem since scheduled properly
    res_sum = ctx.run_udf(dataset=ds, udf=SumUDF())

    # Scheduled also on CUDA and service workers.
    # CUDA will throw an error
    del_res_sum = del_ctx.run_udf(dataset=ds, udf=SumUDF())
    del_res_sum['intensity'].raw_data.compute()