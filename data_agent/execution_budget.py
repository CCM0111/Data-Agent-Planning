"""A task deadline independent of frozen model parameters and cache identities."""
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
import inspect
from pathlib import Path
import signal
import threading
import time


class TaskDeadlineExceeded(TimeoutError):
    """Execution was interrupted by the task budget, not by a method failure."""


class InfrastructureInterruption(OSError):
    """Worker/host failure with no confirmed method defect."""


def infrastructure_failure(error):
    if isinstance(error, (OSError, MemoryError, ImportError)):
        return True
    text = str(error)
    return text.startswith(('OSError:', 'IOError:', 'MemoryError:', 'PermissionError:',
        'FileNotFoundError:', 'ModuleNotFoundError:', 'ImportError:',
        'InfrastructureInterruption:', 'TaskDeadlineExceeded:'))


def _boot_id():
    path = Path('/proc/sys/kernel/random/boot_id')
    return path.read_text().strip() if path.exists() else None


def start_wall_accounting(state, started):
    state['wall_session'] = {'base_seconds': float(state.get('wall_seconds', 0)),
        'started_monotonic': started, 'started_unix': time.time() - (time.monotonic() - started),
        'boot_id': _boot_id()}


def account_wall_time(state, *, finish=False, recover=False):
    """Checkpoint active wall time, including abrupt exits that bypass finally.

    An unclosed session has no trustworthy exit timestamp. Recovery charges the
    unobserved interval conservatively, capped at the old task budget, and records
    that estimate separately. An explicit budget extension can grant more time.
    """
    session = state.get('wall_session')
    if not session:
        return
    if session.get('boot_id') is not None and session['boot_id'] == _boot_id():
        elapsed = max(0., time.monotonic() - session['started_monotonic'])
    else:
        elapsed = max(0., time.time() - session['started_unix'])
    accounted = max(float(state.get('wall_seconds', 0)), session['base_seconds'] + elapsed)
    if recover:
        accounted = max(float(state.get('wall_seconds', 0)), min(accounted,
            float(state.get('wall_budget_seconds', accounted))))
        state.setdefault('wall_recoveries', []).append({
            'last_checkpoint_seconds': state.get('wall_seconds', 0), 'charged_seconds': accounted,
            'basis': 'Unclosed session; conservative elapsed time including unobserved downtime'})
    state['wall_seconds'] = accounted
    if finish or recover:
        state.pop('wall_session', None)


_deadline = ContextVar('data_agent_execution_deadline', default=None)


@contextmanager
def deadline_scope(deadline):
    previous = _deadline.get()
    token = _deadline.set(min(previous, deadline) if previous is not None else deadline)
    try:
        yield
    finally:
        _deadline.reset(token)


def remaining_seconds():
    deadline = _deadline.get()
    return None if deadline is None else max(0., deadline - time.monotonic())


def check_deadline():
    remaining = remaining_seconds()
    if remaining is not None and remaining <= 0:
        raise TaskDeadlineExceeded('Task wall budget exhausted; resume with an execution budget extension')


def bounded_timeout(seconds):
    check_deadline()
    remaining = remaining_seconds()
    return seconds if remaining is None else min(float(seconds), remaining)


async def await_with_budget(awaitable, default_timeout=None):
    try:
        check_deadline()
        remaining = remaining_seconds()
        timeout = (remaining if default_timeout is None else
                   default_timeout if remaining is None else min(default_timeout, remaining))
    except BaseException:
        if inspect.iscoroutine(awaitable):
            awaitable.close()
        raise
    try:
        result = await asyncio.wait_for(awaitable, timeout=timeout)
        check_deadline()
        return result
    except asyncio.CancelledError:
        # An outer task-deadline wait_for can cancel this nested request first.
        # Preserve ordinary user cancellation, but keep deadline recovery typed.
        check_deadline()
        raise
    except asyncio.TimeoutError:
        check_deadline()
        raise


def call_with_budget(function, *args, **kwargs):
    """Bound synchronous CPU stages on the Linux coordinator's main thread.

    Do not wrap awaits or subprocess launches: those have cancellation/cleanup
    paths and use bounded_timeout instead. Other threads check at stage boundaries.
    """
    check_deadline()
    remaining = remaining_seconds()
    can_interrupt = (remaining is not None and hasattr(signal, 'setitimer') and
                     threading.current_thread() is threading.main_thread())
    if not can_interrupt:
        result = function(*args, **kwargs)
        check_deadline()
        return result
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    # Respect an existing timer owned by the caller; do not replace its semantics.
    if previous_timer[0] > 0:
        result = function(*args, **kwargs)
        check_deadline()
        return result
    def expired(signum, frame):
        raise TaskDeadlineExceeded('Task wall budget exhausted during a synchronous stage')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(remaining, 1e-6))
    try:
        result = function(*args, **kwargs)
        check_deadline()
        return result
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
