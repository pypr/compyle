import io
import os
import sys
from tempfile import mktemp


def get_ipython_capture():
    try:
        # This will work inside IPython but not outside it.
        name = get_ipython().__class__.__name__
        if name.startswith('ZMQ'):
            from IPython.utils.capture import capture_output
            return capture_output
        else:
            return None
    except NameError:
        return None


class CaptureStream(object):
    """A context manager which captures any errors on a given stream (like
    sys.stderr).  The stream is captured and the outputs can be used.

    We treat sys.stderr and stdout specially as very often these are
    overridden by nose or IPython.  We always wrap the underlying file
    descriptors in this case as this is the intent of this context manager.

    This is somewhat based on this question:
        http://stackoverflow.com/questions/7018879/disabling-output-when-compiling-with-distutils

    Examples
    --------

    See the tests in tests/test_capture_stream.py for example usage.
    """

    def __init__(self, stream=sys.stderr):
        self.stream = stream
        if stream is sys.stderr:
            self.fileno = 2
        elif stream is sys.stdout:
            self.fileno = 1
        else:
            self.fileno = stream.fileno()
        self.orig_stream = None
        self.tmp_stream = None
        self.tmp_path = ''
        self._cached_output = None

    def __enter__(self):
        if sys.platform.startswith('win32') and sys.version_info[:2] > (3, 5):
            return self
        self.orig_stream = os.dup(self.fileno)
        self.tmp_path = mktemp()
        self.tmp_stream = io.open(self.tmp_path, 'w+', encoding='utf-8')
        os.dup2(self.tmp_stream.fileno(), self.fileno)
        return self

    def __exit__(self, type, value, tb):
        if sys.platform.startswith('win32') and sys.version_info[:2] > (3, 5):
            return
        if self.orig_stream is not None:
            os.dup2(self.orig_stream, self.fileno)
        if self.tmp_stream is not None:
            self._cache_output()
            self.tmp_stream.close()
            os.remove(self.tmp_path)

    def _cache_output(self):
        if self._cached_output is not None:
            return
        tmp_stream = self.tmp_stream
        result = ''
        if tmp_stream is not None:
            tmp_stream.flush()
            tmp_stream.seek(0)
            result = tmp_stream.read()
        self._cached_output = result

    def get_output(self):
        """Return the captured output.
        """
        if self._cached_output is None:
            self._cache_output()
        return self._cached_output


class CaptureMultipleStreams(object):
    """This lets one capture multiple streams together.
    """
    def __init__(self, streams=None):
        streams = (sys.stdout, sys.stderr) if streams is None else streams
        self.streams = streams
        self.captures = [CaptureStream(x) for x in streams]
        cap = get_ipython_capture()
        if cap:
            self.jcap = cap(stdout=True, stderr=True, display=True)
        else:
            self.jcap = None
        self.joutput = None

    def __enter__(self):
        for capture in self.captures:
            capture.__enter__()
        if self.jcap:
            self.joutput = self.jcap.__enter__()
        return self

    def __exit__(self, type, value, tb):
        for capture in self.captures:
            capture.__exit__(type, value, tb)
        if self.jcap:
            self.jcap.__exit__(type, value, tb)

    def get_output(self):
        out = list(x.get_output() for x in self.captures)
        if self.joutput:
            out[0] += self.joutput.stdout
            out[1] += self.joutput.stderr
        return out
