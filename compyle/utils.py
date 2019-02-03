import inspect


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
