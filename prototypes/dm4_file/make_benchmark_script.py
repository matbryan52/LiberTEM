import itertools
import pathlib

if __name__ == '__main__':
    out_file = pathlib.Path(__file__).parent / 'run.sh'

    params = {
            'ds_size_mb': (1024, 4096, 1024 * 16),
            'max_io': (1, 32),
            'tileshape': (1/8, 1/4, 1/2),
            'part_size_mb': (512, 1024, 2048),
        }

    stub = 'python prototypes/dm4_file/benchmark_ds.py'
    iterators = tuple(zip(itertools.repeat(k), v) for k, v in params.items())
    with out_file.open('w') as fp:
        for args in itertools.product(*iterators):
            command = stub
            for name, val in args:
                command = command + f' --{name} {val}'
            fp.write(command + '\n')
