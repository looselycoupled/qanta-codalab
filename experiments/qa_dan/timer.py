# utils.timer
# Provides timing functionality.
#
# Created:
# Author:
#
# ID: timer.py [] allen.leis@gmail.com $

"""
Provides timing functionality for the Hecate application.
"""

##########################################################################
## Imports
##########################################################################

import time

from functools import wraps
from datetime import timedelta

from dateutil.relativedelta import relativedelta

##########################################################################
## Decorator
##########################################################################

def timeit(func, wall_clock=True):
    """
    Appends the return with a Timer object recording function execution time.
    """
    @wraps(func)
    def timer_wrapper(*args, **kwargs):
        """
        Inner function that uses the Timer context object
        """
        with Timer(wall_clock) as timer:
            result = func(*args, **kwargs)

        return result, timer
    return timer_wrapper


##########################################################################
## Timer functions
##########################################################################

def humanizedelta(*args, **kwargs):
    """
    Wrapper around dateutil.relativedelta (same construtor args) and returns
    a humanized string representing the delta in a meaningful way.
    """
    if 'milliseconds' in kwargs:
        sec  = kwargs.get('seconds', 0)
        msec = kwargs.pop('milliseconds')
        kwargs['seconds'] = sec + (float(msec) / 1000.0)

    delta = relativedelta(*args, **kwargs)
    attrs = ('years', 'months', 'days', 'hours', 'minutes', 'seconds')
    parts = [
        '%d %s' % (getattr(delta, attr), getattr(delta, attr) > 1 and attr or attr[:-1])
        for attr in attrs if getattr(delta, attr)
    ]

    return " ".join(parts)

class Timer(object):
    """
    A context object timer. Usage:
        >>> with Timer() as timer:
        ...     do_something()
        >>> print timer.elapsed
    """

    def __init__(self, wall_clock=True):
        """
        If wall_clock is True then use time.time() to get the number of
        actually elapsed seconds. If wall_clock is False, use time.clock to
        get the process time instead.
        """
        self.wall_clock = wall_clock
        self.time = time.time if wall_clock else time.clock

        # Stubs for serializing an empty timer.
        self.started  = None
        self.finished = None
        self.elapsed  = 0.0

    def __enter__(self):
        self.started  = self.time()
        return self

    def __exit__(self, type, value, tb):
        self.finished = self.time()
        self.elapsed  = self.finished - self.started

    def __str__(self):
        return humanizedelta(seconds=self.elapsed)

    @property
    def timedelta(self):
        return timedelta(seconds=self.elapsed)

    def serialize(self):
        return {
            'started':  self.started,
            'finished': self.finished,
            'elapsed':  humanizedelta(seconds=self.elapsed),
        }
