import sys
import os
import time
import itertools
import subprocess
import pathlib
import ray
import numpy as np
from dataset_reader import DatasetMeta
from shm_worker import make_addr, recieve_frames
# from shm_streaming import send_frames
from work_dispatch import BothSumUDF, UDFRunner_r, RunnerProxy, UDFMerger, set_plasma_addr

n_partitions = 2
nworkers = 2
start_control = 50005
start_notify = 51005
endpoint_dict = {'control': [make_addr(p) for p in range(start_control, start_control + n_partitions)],
                'notify': [make_addr(p) for p in range(start_notify, start_notify + n_partitions)]}
plasma_addr = "/tmp/plasma"
# set_plasma_addr(plasma_addr)

if False:
    rawfile = pathlib.Path('./stripes.raw')
    nav_shape = (20, 150)
    sig_shape = (256, 256)
    iteration_params = {'sig_split_yx': (4, 1), 'depth': 16}
else:
    rawfile = pathlib.Path('../../ray_testing/random.raw')
    nav_shape = (50, 100)
    sig_shape = (512, 512)
    iteration_params = {'sig_split_yx': (8, 1), 'depth': 8}
dtype = np.float32
ds_meta = DatasetMeta(rawfile, nav_shape, sig_shape, dtype=dtype)


def run_streamer(eid):
    args = [sys.executable, 'shm_streaming.py', f'{eid}']
    return subprocess.Popen(args, cwd=os.getcwd())


class PoolWrapper:
    def __init__(self, pool):
        self.pool = pool
        self.actor_iterator = itertools.cycle(self.pool)

    def submit(self, dataset_slice, obj_id_string):
        future = next(self.actor_iterator).process_tile.remote(dataset_slice, obj_id_string)
        return future


if __name__ == '__main__':
    tracing_dir = pathlib.Path('./tracing')
    _ = [f.unlink() for f in tracing_dir.iterdir()]

    from tracing import merge_traces, Tracer
    tracer = Tracer('main', '0')

    print(f'DS-Size {np.prod(sig_shape) * np.prod(nav_shape) * np.dtype(dtype).itemsize // 2**20} MB')

    stream_delay = 1
    stream_create = time.time()
    with tracer.span('Create streamers'):
        streamers = [run_streamer(eid) for eid in range(n_partitions)]

    #ray.init(include_dashboard=False)
    # ray.init(address='auto', _redis_password='5241590000000000')
    with tracer.span('ray_init'):
        ray.init(include_dashboard=False)
    
    if time.time() - stream_create < stream_delay:
        time.sleep(time.time() - stream_create)

    with tracer.span('create_actors'):
        udfs = [{'class': BothSumUDF, 'args': [], 'kwargs': {}}]
        actors = [UDFRunner_r.remote(udfs, ds_meta, worker_id=str(i)) for i in range(nworkers)]
    
    with tracer.span('warmup_actors'):
        for a in actors:
            _ = ray.get(a.warmup.remote(True))

    with tracer.span('recieve_frames'):
        dispatch_tracer = recieve_frames(endpoint_dict, plasma_addr, PoolWrapper(actors), tracer=tracer)

    with tracer.span('merge_udfs'):
        runners = [RunnerProxy(a) for a in actors]
        merger = UDFMerger(runners, udfs, ds_meta)
        merger.merge_runners()

    with tracer.span('kill_streamers'):
        for s in streamers:
            # s.kill()
            outs, errs = s.communicate()
            print(outs)

    for a in actors:
        f = a.store_tracer.remote()
        ray.get(f)

    # savepath = pathlib.Path('./tracing/dispatch.json')
    # dispatch_tracer.store(savepath)

    savepath = pathlib.Path('./tracing/main.json')
    tracer.store(savepath)

    savepath = pathlib.Path('./tracing/ray-tracing.json')
    try:
        json_str = ray.timeline(savepath)
    except Exception:
        pass

    merged_path = savepath.parent / 'merged.json'
    merge_traces(savepath.parent, merged_path)

    import matplotlib.pyplot as plt
    res = merger.results[0]
    fig, axs = plt.subplots(1, 2)
    for k, ax in zip(res.keys(), axs):
        ax.imshow(res.get_buffer(k))
        ax.set_title(k)
    plt.show()
