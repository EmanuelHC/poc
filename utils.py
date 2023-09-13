import sys
import io

def get_log_streams():
    return io.StringIO(), io.StringIO()

stdout_stream, stderr_stream = get_log_streams()

class CaptureLogs:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = stdout_stream
        sys.stderr = stderr_stream

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
