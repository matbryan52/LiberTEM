import pathlib
import json
import time
from collections import deque
import copy


def make_record(*, cat, name, pid, tid, tstart, duration):
    return {'cat': cat,
            'name': name,
            'pid': pid,
            'tid': tid,
            'ts': tstart,
            'dur': duration,
            'ph': 'X',
            'cname': '',
            'args': {}}


def _frame(msg, pid, tid, frame_id, tt=None, data=False, fid=True):
    if tt is None:
        tt = time.time() * 1e6
    frame_id = {'frameId': frame_id} if fid else {}
    args = {"layerTreeId": 3, 'data': frame_id} if data else {"layerTreeId": 3, **frame_id}
    return {"pid": pid,
            "tid": tid,
            "ts": tt,
            "ph": "I",
            "cat": "devtools.timeline.frame",
            "name": msg,
            "args": args,
            "tts": tt,
            "s": "t"}


def end_frame(pid, tid, frameId, tt=None):
    return _frame('DrawFrame', pid, tid, frameId, tt=tt, fid=False)


def start_frame(pid, tid, frameId, tt=None):
    return [_frame('BeginFrame', pid, tid, frameId, tt=tt),
            _frame('BeginMainThreadFrame', pid, tid, frameId, tt=tt+1, data=True),
            _frame('ActivateLayerTree', pid, tid, frameId, tt=tt+2, data=False)]


records = deque()


class Span():
    def __init__(self, params):
        self.params = params

    def __enter__(self, *args, **kwargs):
        self.tstart = time.time()
        return self

    def __exit__(self, *args):
        duration = (time.time() - self.tstart) * 1e6
        kwargs = {'tstart': self.tstart * 1e6, 'duration': duration, **self.params}
        records.append(make_record(**kwargs))


class Tracer:
    def __init__(self, pid, tid):
        self.pid = pid
        self.tid = tid
        self.frames = 0

    def span(self, name, cat=None):
        if cat is None:
            cat = name
        return Span({'cat': cat, 'name': name, 'pid': self.pid, 'tid': self.tid})

    def store(self, path):
        _ = pathlib.Path(path).unlink(missing_ok=True)
        _r = copy.deepcopy(records)
        with pathlib.Path(path).open('w') as fp:
            json.dump(list(_r), fp, indent=4)

    def frame(self):
        tt = time.time() * 1e6
        if self.frames == 0:
            records.extend(start_frame(self.pid, self.tid, self.frames, tt=tt))
        self.frames = True
        records.append(end_frame(self.pid, self.tid, self.frames, tt=tt))
        self.frames += 1
        records.extend(start_frame(self.pid, self.tid, self.frames, tt=tt+1))


def merge_traces(source_dir, dest):
    dest.unlink(missing_ok=True)
    source_paths = [f for f in source_dir.iterdir() if f.suffix == '.json']
    events = []
    for f in source_paths:
        events += json.load(f.open('r'))
    json.dump(events, dest.open('w'), indent=4)