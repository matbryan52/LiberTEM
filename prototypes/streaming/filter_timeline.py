import pathlib
import json

if __name__ == '__main__':
    tfile = pathlib.Path('~/Downloads/Profile-20210909T164230.json').expanduser()
    jlist = json.load(tfile.open('r'))

    cats = ['disabled-by-default-devtools.timeline.frame']
    names = ['LatencyInfo.Flow',
            'SetLayerTreeId',
            'UpdateLayerTree']

    flist = [r for r in jlist if r['cat'] in cats or r['name'] in names]
    ofile = pathlib.Path('./filtered.json')
    json.dump(flist, ofile.open('w'))