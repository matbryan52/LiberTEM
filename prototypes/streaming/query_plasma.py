import time
import pathlib
import json
import pyarrow.plasma as plasma
from run_streaming import plasma_addr
import matplotlib.pyplot as plt
import numpy as np


def make_record(tval, num, etime):
    pid = 'plasma'
    tid = '0'
    return {'pid': pid,
            'tid': tid,
            'ts': tval,
            'ph': 'I',
            'cat': 'disabled-by-default-devtools.timeline',
            'name': 'UpdateCounters',
            'args': {'data': {'documents': num, 'test': 5}},
            'tts': etime,
            's': 't'}


def make_json(data):
    json_list = []
    for tval, num, etime in data:
        json_list.append(make_record(tval, num, etime))
    return json_list


if __name__ == '__main__':
    client = plasma.connect(plasma_addr)

    data = []
    while True:
        try:
            dtime = time.time() * 1e6
            objects = client.list()
            n_alive = sum([1 for o in objects.values() if o['ref_count'] > 0])
            etime = time.time() * 1e6 - dtime
            data.append((dtime, n_alive, etime))
            time.sleep(0.025)
        except KeyboardInterrupt:
            break

    json_list = make_json(data)
    savepath = pathlib.Path('./tracing/plasma.json')
    with savepath.open('w') as fp:
        json.dump(json_list, fp)

    from tracing import merge_traces
    merged_path = savepath.parent / 'merged.json'
    merge_traces(savepath.parent, merged_path)

    array = np.asarray(data)
    plt.plot(array[:, 0], array[:, 1], 'x-')
    plt.show()
