import inspect
from ext_module import get_platform_dir
from os.path import join, expanduser
import os
import shutil


def getsourcelines(obj):
    '''Given an object return the source code that defines it as a list of
    lines along with the starting line.
    '''
    try:
        return inspect.getsourcelines(obj)
    except Exception:
        if hasattr(obj, 'source'):
            return obj.source.splitlines(True), 0
        else:
            raise


def getsource(obj):
    '''Given an object return the source that defines it.
    '''
    try:
        return inspect.getsource(obj)
    except Exception:
        if hasattr(obj, 'source'):
            return obj.source
        else:
            raise


def identify_filesystem():
    # All procs should call this function
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    root = expanduser(join('~', '.compyle', 'filesystem'))

    work_dir = join(root, 'fsid%s' % rank)

    if not isdir(work_dir):
        os.makedirs(work_dir)

    comm.barrier()

    sharing_ranks = [x.replace('fsid', '') for x in os.listdir(root) \
                     if 'fsid' in x]

    comm.barrier()

    shutil.rmtree(root)

