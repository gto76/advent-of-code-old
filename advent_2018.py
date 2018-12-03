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


DATA_FILENAME = 'data_2018/{}.data'


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


###
##  PROBLEMS
#

def p_1_a(data):
    numbers = [int(a) for a in data]
    return sum(numbers)


def p_1_b(data):
    frequencies = set()
    freq = 0
    for line in cycle(data):
        freq += int(line)
        if freq in frequencies:
            return freq
        frequencies.add(freq)


def p_2_a(data):
    twos, threes = 0, 0
    for line in data:
        two, three = False, False
        for a in range(97, 123):
            a = chr(a)
            b = len(re.findall(a, line))
            if b == 2:
                two = True
            if b == 3:
                three = True
        if two:
            twos += 1
        if three:
            threes += 1
    return twos * threes


def p_2_b(data):
    def exclude(i, lines):
        out = list(lines)
        out.pop(i)
        return out

    def exists(shrt, lines, i):
        for line in lines:
            shrt_2 = ''.join(exclude(i, line))
            if shrt == shrt_2:
                return True

    for i, line in enumerate(data):
        for j in range(0, len(line)):
            line_out = ''.join(exclude(j, line))
            if exists(line_out, data[i+1:], j):
                return line_out


def p_3_a(data):
    def parse_line(line):
        l, r = line.split(' @ ')
        ll, rr = r.split(': ')
        x_off, y_off = ll.split(',')
        wid, hei = rr.split('x')
        p_off = P(int(x_off), int(y_off))
        p_wid = P(int(wid), int(hei))
        fill_cnt(p_off, p_wid)

    def fill_cnt(off, wid):
        for i in range(off.x, off.x+wid.x):
            for j in range(off.y, off.y+wid.y):
                cnt[P(i, j)] += 1

    cnt = defaultdict(int)
    for line in data:
        parse_line(line)
    out = 0
    for v in cnt.values():
        if v > 1:
            out += 1
    return out


def p_3_b(data):
    class Square:
        def __init__(self):
            self.ids = []

    def parse_line(line):
        l, r = line.split(' @ ')
        id_ = l[1:]
        ids.add(id_)
        ll, rr = r.split(': ')
        x_off, y_off = ll.split(',')
        wid, hei = rr.split('x')
        p_off = P(int(x_off), int(y_off))
        p_wid = P(int(wid), int(hei))
        fill_cnt(p_off, p_wid, id_)

    def fill_cnt(off, wid, id_):
        for i in range(off.x, off.x+wid.x):
            for j in range(off.y, off.y+wid.y):
                cnt[P(i, j)].ids.append(id_)

    cnt = defaultdict(Square)
    ids = set()
    for line in data:
        parse_line(line)
    double_ids = set()
    for k, v in cnt.items():
        if len(v.ids) > 1:
            double_ids.update(v.ids)
    return ids.difference(double_ids)







FUN = p_3_a
print(run(FUN))





















