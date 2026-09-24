"""Run a function in a disposable forked child.

Native code that reads untrusted input (libdarshan-util on a truncated log)
can kill the whole interpreter with SIGBUS or SIGABRT. Running the call in a
child turns such a death, a timeout, or an exception into ``ChildFailure``
with a one-line cause, and leaves the calling process untouched.
"""

from __future__ import annotations

import os
import pickle
import resource
import select
import signal
import time
from collections.abc import Callable

_PIPE_CHUNK = 1 << 16


class ChildFailure(RuntimeError):
    """The child produced no result: it raised, died, exited, or timed out."""


def run_in_child(fn: Callable, *args, timeout: float | None = None, **kwargs):
    """Return ``fn(*args, **kwargs)`` computed in a forked child.

    The child sends its pickled return value, or the text of the exception it
    raised, through a pipe and leaves with ``os._exit`` (no interpreter
    shutdown, so shutdown-time crashes of a C library cannot reach the
    caller). ``timeout`` seconds after the fork the child is killed with
    SIGKILL. Any outcome other than a returned value raises ``ChildFailure``.

    Core dumps are disabled in the child: its death is an expected, recorded
    outcome, and writing a core of a large process to the shared file system
    takes seconds (long enough to look like a timeout) and hundreds of MB.
    """
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            try:
                payload = pickle.dumps((True, fn(*args, **kwargs)))
            except BaseException as exc:  # the child reports every failure
                payload = pickle.dumps((False, f"{type(exc).__name__}: {str(exc)[:300]}"))
            view = memoryview(payload)
            while view:
                written = os.write(write_fd, view[:_PIPE_CHUNK])
                view = view[written:]
        finally:
            os._exit(0)

    os.close(write_fd)
    chunks = []
    deadline = None if timeout is None else time.monotonic() + timeout
    timed_out = False
    try:
        while True:
            remaining = None if deadline is None else max(deadline - time.monotonic(), 0.0)
            ready, _, _ = select.select([read_fd], [], [], remaining)
            if not ready:
                timed_out = True
                break
            chunk = os.read(read_fd, _PIPE_CHUNK)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(read_fd)
    if timed_out:
        os.kill(pid, signal.SIGKILL)
    _, status = os.waitpid(pid, 0)
    name = getattr(fn, '__name__', repr(fn))
    if timed_out:
        raise ChildFailure(f"timeout_after_{timeout:g}s in {name}")
    if os.WIFSIGNALED(status):
        signum = os.WTERMSIG(status)
        raise ChildFailure(
            f"child running {name} was killed by signal {signum} ({signal.Signals(signum).name})")
    if os.WEXITSTATUS(status) != 0:
        raise ChildFailure(f"child running {name} exited with status {os.WEXITSTATUS(status)}")
    try:
        ok, value = pickle.loads(b''.join(chunks))
    except Exception as exc:
        raise ChildFailure(f"child running {name} returned an unreadable result: {exc}") from exc
    if not ok:
        raise ChildFailure(value)
    return value
