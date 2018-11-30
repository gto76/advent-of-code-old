from collections import Counter, namedtuple, defaultdict
from copy import copy, deepcopy
from enum import Enum
from functools import reduce
import hashlib
from inspect import signature
from itertools import *
import json
from math import *
import operator as op
import re
import sys
from time import sleep

from util import *


DATA_FILENAME = 'data_2015/{}.data'


###
##  MAIN
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
    comms = data[0].split(', ')
    p = P(0, 0)
    d = D.n
    for c in comms:
        r = R(c[0])
        d = ROTATE[(d, r)]
        fac = int(c[1:])
        p_n = MOVE[d]
        p_n = P(p_n.x * fac, p_n.y * fac)
        p = P(p.x + p_n.x, p.y + p_n.y)
    return abs(p.x) + abs(p.y)


def p_1_b(data):
    comms = data[0].split(', ')
    visited = []
    p = P(0, 0)
    visited.append(p)
    d = D.n
    for c in comms:
        r = R(c[0])
        d = ROTATE[(d, r)]
        fac = int(c[1:])
        p_n = MOVE[d]
        for _ in range(fac):
            p = P(p.x + p_n.x, p.y + p_n.y)
            if p in visited:
                return abs(p.x) + abs(p.y)
            visited.append(p)


def p_2_a(data):
    pass


FUN = p_1_b
print(run(FUN))