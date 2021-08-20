from copy import deepcopy
import functools
import logging

import ray

from .base import (
    JobExecutor, JobCancelledError,
    Environment,
)
from .scheduler import Worker, WorkerSet
from libertem.udf.base import UDFTask
from libertem.common.backend import set_use_cpu, set_use_cuda

log = logging.getLogger(__name__)


"""
I've tried to follow the structure of dask.py where practical,
but some of the concepts don't really transfer over, particularly
node/worker locations and resource allocation, not without quite
substantial refactoring.

As a result some of the functions don't really make sense
even though I've tried to convert them
"""

def worker_setup(resource, device):
    # Disable handling Ctrl-C on the workers for a local cluster
    # since the nanny restarts workers in that case and that gets mixed
    # with Ctrl-C handling of the main process, at least on Windows
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if resource == "CUDA":
        set_use_cuda(device)
    elif resource == "CPU":
        set_use_cpu(device)
    else:
        raise ValueError("Unknown resource %s, use 'CUDA' or 'CPU'", resource)


def cluster_spec(cpus, cudas, has_cupy, name='default', num_service=0, options=None):

    if options is None:
        options = {}
    if options.get("nthreads") is None:
        options["nthreads"] = 1
    if options.get("silence_logs") is None:
        options["silence_logs"] = logging.WARN

    workers_spec = {}

    cpu_options = deepcopy(options)
    cpu_options["resources"] = {"CPU": 1, "compute": 1, "ndarray": 1}
    cpu_base_spec = {
        # "cls": dd.Nanny,
        "options": cpu_options,
    }

    # Service workers not for computation

    service_options = deepcopy(options)
    service_options["resources"] = {}
    service_base_spec = {
        # "cls": dd.Nanny,
        "options": service_options
    }

    cuda_options = deepcopy(options)
    cuda_options["resources"] = {"CUDA": 1, "compute": 1}
    if has_cupy:
        cuda_options["resources"]["ndarray"] = 1
    cuda_base_spec = {
        # "cls": dd.Nanny,
        "options": cuda_options
    }

    for cpu in cpus:
        cpu_spec = deepcopy(cpu_base_spec)
        # cpu_spec['options']['preload'] = \
        #     'from libertem.executor.dask import worker_setup; ' + \
        #     f'worker_setup(resource="CPU", device={cpu})'
        workers_spec[f'{name}-cpu-{cpu}'] = cpu_spec

    for service in range(num_service):
        workers_spec[f'{name}-service-{service}'] = deepcopy(service_base_spec)

    for cuda in cudas:
        cuda_spec = deepcopy(cuda_base_spec)
        # cuda_spec['options']['preload'] = \
        #     'from libertem.executor.dask import worker_setup; ' + \
        #     f'worker_setup(resource="CUDA", device={cuda})'
        workers_spec[f'{name}-cuda-{cuda}'] = cuda_spec

    return workers_spec


def ray_as_completed(futures, num_returns=1, fetch_local=True, **kwargs):
    """
    A generator around a list of Ray futures which yields num_returns
    futures at a time, in as-completed order. The caller then needs to
    call ray.get to actually get the result from the future.

    - fetch_local : pre-fetches results onto the local node before
    declaring that a future has completed

    This fills the same role as dd.as_completed(futures) in Dask
    """
    unfinished = futures
    while unfinished:
        finished, unfinished = ray.wait(unfinished,
                                        num_returns=num_returns,
                                        fetch_local=fetch_local,
                                        **kwargs)
        yield finished


@ray.remote
def run_remote_wrapper(fn, *args, **kwargs):
    """
    A wrapper around a function to run it remotely

    The function will be serialized before being run each time
    with no holding of the serialied form on workers.

    This is a Ray anti-pattern but does work for
    the cases where it is needed (for example in executor.run_function)
    """
    return fn(*args, **kwargs)


@ray.remote
def ray_task_creator(*, udf_init, corrections, roi, backends,
                    partition_idx, partition, env, task_idx):
    """
    Executes the task as a remote job. This remotely does
    what UDFRunner._make_udf_tasks() does currently.

    It is executed for each partition as a remote function:
    UDFs are init'd, buffers allocated for the partition
    UDFTask created and then called

    This could be done better with a Ray-specific
    stateful UDFRunner with a method which creates a task,
    called once for each partition.

    I am hacking in the functionality of TaskProxy here
    because once a function has been decorated as remote
    it's inefficient to re-wrap its return values
    """
    udfs = [ud['class'](**ud['kwargs']) for ud in udf_init]

    for udf in udfs:
        udf.params.new_for_partition(partition, roi)

    task_result = UDFTask(
        partition=partition, idx=partition_idx, udfs=udfs, roi=roi, corrections=corrections,
        backends=backends)(env=env)

    return {
            "task_result": task_result,
            "task_id": task_idx,
            }

class CommonRayMixin:

    def _resources_available(self, workers, resources):
        """Placeholder function"""
        def filter_fn(worker):
            return all(worker.resources.get(key, 0) >= resources[key] for key in resources.keys())

        return len(workers.filter(filter_fn))

    def has_libertem_resources(self):
        """Placeholder function"""
        workers = self.get_available_workers()

        def has_resources(worker):
            r = worker.resources
            # FIXME
            return 'compute' in r and (('CPU' in r and 'ndarray' in r) or 'GPU' in r)

        return len(workers.filter(has_resources)) > 0

    def _get_futures(self, tasks, *args, **kwargs):
        '''
        Submit jobs to the cluster

        Tasks is either:
         - a list of dictionaries
            [{'callable':fn, 'args':[], 'kwargs'}]
        or
         - a list of callable
           in which case *args/**kwargs are passed to all
        '''
        available_workers = self.get_available_workers()
        if len(available_workers) == 0:
            raise RuntimeError("no workers available!")
        futures = []
        for task in tasks:
            if isinstance(task, dict):
                future = self.call_remote(task['callable'],
                                          *task.get('args', []),
                                          **task.get('kwargs', {}))
            elif callable(task):
                future = self.call_remote(task, *args, **kwargs)
            else:
                raise RuntimeError('Unrecognized task format (dict or callable)')
            futures.append(future)
        return futures

    def call_remote(self, fn, *args, **kwargs):
        """
        Call fn remote task with args and kwargs
        If fn was already decorated as a @ray.remote procedure
        then call this directly, otherwise use the wrapper
        to serialize fn and call it immediately
        """
        try:
            return fn.remote(*args, **kwargs)
        except AttributeError:
            return run_remote_wrapper.remote(fn, *args, **kwargs)

    def get_available_workers(self):
        return WorkerSet([
            Worker(
                name=worker['NodeID'],
                host=worker['NodeManagerAddress'],
                resources=worker['Resources']
            )
            for worker in ray.nodes()
        ])

    def get_resource_details(self):
        workers = self.get_available_workers()
        details = {}

        for worker in workers:
            host_name = worker.host

            r = worker.resources
            if "CPU" in r:
                resource = 'cpu'
            elif "GPU" in r:
                resource = 'cuda'
            else:
                resource = 'service'

            if host_name not in details.keys():
                details[host_name] = {
                                 'host': host_name,
                                 'cpu': 0,
                                 'cuda': 0,
                                 'service': 0,
                            }
            details[host_name][resource] += 1

        details_sorted = []
        for host in sorted(details.keys()):
            details_sorted.append(details[host])

        return details_sorted

    def _store_global_task_info(self, _udfs, corrections, roi, backends):
        """
        Put Task constant initialization variables into global shared
        memory and return a dict of ray.ObjectRef hashes
        """
        udf_init_info = []
        for udf in _udfs:
            udf_init_info.append({'class': udf.__class__, 'kwargs': udf._kwargs})
        udf_init_hash = self.put(udf_init_info)

        task_metadata = {'corrections': self.put(corrections),
                         'roi': self.put(roi),
                         'backends': self.put(backends)}

        return {'udf_init': udf_init_hash,
                **task_metadata}

    def _add_local_task_info(self, global_task_dict, partition_idx, partition):
        """
        Add the partition-specific items for running a UDFTask
        to a new dictionary with the global hashes, used to
        initialize the remote job
        """
        return {**global_task_dict, 'partition_idx': partition_idx, 'partition': partition}

    def _link_task_callable(self, task_dict):
        """
        Wrap the task definition dictionary with the callable
        ran as a remote task
        """
        return {'callable': ray_task_creator, 'kwargs': task_dict}


class TaskRecord(object):
    """
    Translation layer to allow downstream code
    to function without modification
    
    Unecessary after refactoring
    """
    def __init__(self, partition):
        self.partition = partition


class RayExecutor(CommonRayMixin, JobExecutor):
    def __init__(self, client, is_local=False, lt_resources=None):
        self.is_local = is_local
        self.client = client
        if lt_resources is None:
            lt_resources = self.has_libertem_resources()
        self.lt_resources = lt_resources
        self._futures = {}

    def put(self, obj):
        """
        Put an item into global shared memory and get its hash
        """
        obj_reference = ray.put(obj)
        return obj_reference

    def get(self, obj_reference):
        """
        Get one or more objects from shared memory

        obj_reference can be a list
        """
        objs = ray.get(obj_reference)
        return objs

    def run_tasks(self, tasks, cancel_id):
        """
        Run tasks in a way compatible with existing code

        Expects tasks to be a list of dictionaries of kwargs
        typically generated with self.__store_global_task_info
        and self._add_local_task_info together

        Yields (result, task) tuples compatible with downstream
        code, although we don't actually have the UDFTask object
        so the task is replaced with a TaskRecord instance with
        the partition attribute for compatibility
        """
        tasks = list(tasks)

        env = Environment(threads_per_worker=1)
        for idx, task_dict in enumerate(tasks):
            task_dict['env'] = env
            task_dict['task_idx'] = idx
        tasks = [self._link_task_callable(task_dict) for task_dict in tasks]

        def _id_to_task(task_id):
            return tasks[task_id]

        futures = self._get_futures(tasks)
        self._futures[cancel_id] = futures

        try:
            as_completed = ray_as_completed(futures)
            for completed_futures in as_completed:
                for c_future in completed_futures:
                    try:
                        result_wrap = ray.get(c_future)
                    except ray.exceptions.TaskCancelledError:
                        del self._futures[cancel_id]
                        raise JobCancelledError()
                    result = result_wrap['task_result']
                    task_dict = _id_to_task(result_wrap['task_id'])
                    task = TaskRecord(task_dict['kwargs']['partition'])
                    yield result, task
        finally:
            if cancel_id in self._futures:
                del self._futures[cancel_id]

    def cancel(self, cancel_id):
        if cancel_id in self._futures:
            futures = self._futures[cancel_id]
            for future in futures:
                ray.cancel(future)

    def run_each_partition(self, partitions, fn, all_nodes=False):
        """
        Run `fn` for all partitions. Yields results in order of completion.

        Parameters
        ----------

        partitions : List[Partition]
            List of relevant partitions.

        fn : callable
            Function to call, will get the partition as first and only argument.

        all_nodes : bool
            If all_nodes is True, run the function on all nodes that have this partition,
            otherwise run on any node that has the partition. If a partition has no location,
            the function will not be run for that partition if `all_nodes` is True, otherwise
            it will be run on any node.
        """
        items = [functools.partial(fn, p) for p in partitions]
        futures = self._get_futures(items)

        for completed_futures in ray_as_completed(futures):
            for c_future in completed_futures:
                try:
                    result = ray.get(c_future)
                except ray.exceptions.TaskCancelledError:
                    raise JobCancelledError()
            yield result

    def run_function(self, fn, *args, **kwargs):
        """
        run a callable `fn` on any worker
        """
        future = run_remote_wrapper.remote(fn, *args, **kwargs)
        return ray.get(future)

    def map(self, fn, iterable):
        """
        Run a callable `fn` for each element in `iterable`, on arbitrary worker nodes.

        Parameters
        ----------

        fn : callable
            Function to call. Should accept exactly one parameter.

        iterable : Iterable
            Which elements to call the function on.
        """
        remote_fn = ray.remote(fn)
        return ray.get([remote_fn.remote(it) for it in iterable])

    def close(self):
        ray.shutdown()

    @classmethod
    def connect(cls, *args, **kwargs):
        """
        Connect to a remote Ray scheduler

        Returns
        -------
        RayJobExecutor
            the connected JobExecutor
        """
        ray.init(*args, **kwargs)
        client = {'args': args, 'kwargs': kwargs}
        return cls(client, is_local=False, lt_resources=None)

    @classmethod
    def make_local(cls, default_cluster=False, spec=None, cluster_kwargs=None, client_kwargs=None):
        """
        Spin up a local Ray cluster

        interesting cluster_kwargs:
            threads_per_worker
            n_workers

        Returns
        -------
        RayJobExecutor
            the connected JobExecutor
        """
        if not default_cluster:
            if spec is None:
                from libertem.utils.devices import detect
                spec = cluster_spec(**detect())

            if cluster_kwargs is None:
                cluster_kwargs = {}
            if cluster_kwargs.get('silence_logs') is None:
                cluster_kwargs['silence_logs'] = logging.WARN

            ray_args = {
                'configure_logging': True,
                'logging_level': cluster_kwargs['silence_logs'],
                'num_cpus': sum([s['options']['resources'].get('CPU', 0) for s in spec.values()]),
                'num_gpus': sum([s['options']['resources'].get('CUDA', 0) for s in spec.values()]),
            }
        else:
            ray_args = {}

        if not ray.is_initialized():
            ray.init(**ray_args)
        else:
            log.warn('Ray has already been intialized, trying with existing cluster')
        return cls(client=None, is_local=True, lt_resources=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
