"""``run_in_child`` turns a crash, an exception, or a timeout into a recorded failure."""
import os
import signal
import time

from src.utils.isolation import ChildFailure, run_in_child


def _square(value):
    return value * value


def _raise(text):
    raise ValueError(text)


def _die_by_signal():
    os.kill(os.getpid(), signal.SIGBUS)


def _sleep(seconds):
    time.sleep(seconds)
    return 'done'


def _large():
    return bytes(3_000_000)


def _expect_failure(fn, *args, text, **kwargs):
    try:
        run_in_child(fn, *args, **kwargs)
    except ChildFailure as exc:
        assert text in str(exc), str(exc)
        return str(exc)
    raise AssertionError('expected ChildFailure')


def test_value_and_large_result_round_trip():
    assert run_in_child(_square, 12) == 144
    assert len(run_in_child(_large)) == 3_000_000


def test_exception_signal_and_timeout_are_recorded_causes():
    assert _expect_failure(_raise, 'bad input', text='ValueError: bad input')
    before = set(os.listdir('.'))
    assert _expect_failure(_die_by_signal, text='SIGBUS')
    assert not [name for name in set(os.listdir('.')) - before if name.startswith('core')]
    started = time.monotonic()
    assert _expect_failure(_sleep, 30, timeout=0.5, text='timeout_after_0.5s')
    assert time.monotonic() - started < 5
    # the caller survives every failure and can keep working
    assert run_in_child(_square, 3) == 9
