from enum import Enum
from collections import namedtuple, defaultdict
import re
import json
from itertools import *
from collections import Counter, defaultdict, namedtuple
from copy import deepcopy
from functools import reduce
import operator as op
import sys
from time import sleep
import hashlib
from math import *

from inspect import signature


DATA_FILENAME = 'data2015/{}.data'

# Problem Enum
Prb = Enum('Prb', 'a b')

# Position Tuple
P = namedtuple('P', 'x y')


###
##  UTIL
#

def run(fun):
    sig = signature(fun)
    no_parm = len(sig.parameters)
    if no_parm == 0:
        return fun()
    problem_name = fun.__name__
    if problem_name.endswith('_a') or problem_name.endswith('_b'):
        problem_name = problem_name[:-2]
    data = get_data(problem_name)
    return fun(data)


def get_data(problem_name):
    try:
        return get_file_contents(DATA_FILENAME.format(problem_name))
    except FileNotFoundError:
        print("Missing data file {}".format(
            DATA_FILENAME.format(problem_name)), file=sys.stderr)
        return


def get_file_contents(file_name):
    with open(file_name, encoding='utf-8') as f:
        return [line.strip('\n') for line in f.readlines()]


class Bar:
    @staticmethod
    def range(*args):
        bar = Bar(len(list(range(*args))))
        for i in range(*args):
            yield i
            bar.tick()

    @staticmethod
    def p(t):
        sys.stdout.write(t)
        sys.stdout.flush()

    def foreach(elements):
        bar = Bar(len(elements))
        for el in elements:
            yield el
            bar.tick()

    def __init__(s, steps, width=40):
        s.st, s.wi, s.fl, s.i = steps, width, 0, 0
        s.th = s.fl * s.st / s.wi
        s.p(f"[{' ' * s.wi}]")
        s.p('\b' * (s.wi + 1))

    def tick(s):
        s.i += 1
        while s.i > s.th:
            s.fl += 1
            s.th = s.fl * s.st / s.wi
            s.p('-')
        if s.i == s.st:
            s.p('\n')


def bit_not(n, numbits=8):
    return (1 << numbits) - 1 - n


###
##  PROBLEMS
#

def p_1_a(data):
    pass




FUN = p_1_a
print(run(FUN))