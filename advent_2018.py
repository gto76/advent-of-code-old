from collections import Counter, namedtuple, defaultdict
from copy import copy, deepcopy
from datetime import datetime
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


def p_4_a(data):
    class Event:
        def __init__(self, d, ev):
            self.d = d
            self.ev = ev
        def __str__(self):
            return f'{self.d}: {self.ev}'

    timeline = []
    for line in data:
        d = datetime.strptime(line[1:17], '%Y-%m-%d %H:%M')
        ev = line[19:]
        e = Event(d, ev)
        timeline.append(e)
    timeline = sorted(timeline, key=lambda a: a.d)
    for t in timeline:
        print(t)


def p_5_a(data):
    def process(line):
        out = []
        skip = False
        for i, a in enumerate(line[:-1], 1):
            if skip:
                skip = False
                continue
            b = line[i]
            if abs(ord(a) - ord(b)) == 32:
                skip = True
                continue
            out.append(a)
        if not skip:
            out.append(line[-1])
        return out

    line = data[0]
    out = process(line)
    old_out = out
    while True:
        out = process(old_out)
        if len(out) == len(old_out):
            break
        old_out = out
    print(''.join(out))
    return len(out)


def p_5_b(data):
    def process(line):
        out = []
        skip = False
        for i, a in enumerate(line[:-1], 1):
            if skip:
                skip = False
                continue
            b = line[i]
            if abs(ord(a) - ord(b)) == 32:
                skip = True
                continue
            out.append(a)
        if not skip:
            out.append(line[-1])
        return out

    def get_len(line):
        out = process(line)
        old_out = out
        while True:
            out = process(old_out)
            if len(out) == len(old_out):
                break
            old_out = out
        return len(out)

    line = data[0]
    min_ = inf
    for i in range(65, 65+32):
        print(chr(i))
        line_b = line.replace(chr(i), '')
        line_b = line_b.replace(chr(i+32), '')
        len_ = get_len(line_b)
        print(len_)
        if len_ < min_:
            min_ = len_
    return min_


def p_6_a(data):
    def parse_line(line):
        y, x = line.split(', ')
        coords.append(P(int(x), int(y)))

    def get_bounds():
        xxx = [a.x for a in coords]
        yyy = [a.y for a in coords]
        return P(min(xxx), min(yyy)), P(max(xxx), max(yyy))

    def get_man(p1, p2):
        return abs(p1.x - p2.x) + abs(p1.y - p2.y)

    def get_in_of_closest(p):
        out = []
        for a in coords:
            out.append(get_man(p, a))
        return out.index(min(out))

    coords = []
    for line in data:
        parse_line(line)

    b_1, b_2 = get_bounds()

    matrix = []

    print(b_1.y, b_2.y)
    for y in range(b_1.y, b_2.y+1):
        print(y)
        for x in range(b_1.x, b_2.x+1):
            i = get_in_of_closest(P(x, y))
            matrix.append(i)

    c = Counter(matrix)
    return str(c.most_common()[0][1])


def p_6_b(data):
    def parse_line(line):
        y, x = line.split(', ')
        coords.append(P(int(x), int(y)))

    def get_bounds():
        x = [a.x for a in coords]
        y = [a.y for a in coords]
        return P(min(x), min(y)), P(max(x), max(y))

    def get_man(p1, p2):
        return abs(p1.x - p2.x) + abs(p1.y - p2.y)

    def get_sum_of_distances(p):
        return sum(get_man(a, p) for a in coords)

    coords = []
    for line in data:
        parse_line(line)

    b_1, b_2 = get_bounds()

    out = 0
    print(b_1.y, b_2.y)
    for y in range(b_1.y, b_2.y+1):
        for x in range(b_1.x, b_2.x+1):
            sum_ = get_sum_of_distances(P(x, y))
            if sum_ < 10000:
                out += 1

    return out


def p_7_a(data):
    def parse_line(line):
        step = line[5]
        step_n = line[-12]
        parsed[step].add(step_n)


    parsed = defaultdict(set)
    for line in data:
        parse_line(line)

    all_values = set()
    for v in parsed.values():
        all_values.update(v)

    last = None
    for k in parsed.keys():
        if k not in all_values:
            last = k
            break 


    out = []
    while True:
        out.append(root)


def p_8_a(data):
    Node = namedtuple('Node', 'nodes metas')
    def get_node(nums):
        q_nodes = next(nums)
        q_met = next(nums)
        c_nodes = []
        for _ in range(q_nodes):
            c_nodes.append(get_node(nums))
        metas = []
        for _ in range(q_met):
            metas.append(next(nums))
        node = Node(c_nodes, metas)
        nodes.append(node)
        return node


    line = data[0]
    nums = (int(a) for a in line.split())
    nodes = []

    root = get_node(nums)

    #out = 0
    return sum(sum(a.metas) for a in nodes)



def p_8_b(data):
    Node = namedtuple('Node', 'nodes metas')
    def get_node(nums):
        q_nodes = next(nums)
        q_met = next(nums)
        c_nodes = []
        for _ in range(q_nodes):
            c_nodes.append(get_node(nums))
        metas = []
        for _ in range(q_met):
            metas.append(next(nums))
        node = Node(c_nodes, metas)
        nodes.append(node)
        return node


    line = data[0]
    nums = (int(a) for a in line.split())
    nodes = []

    root = get_node(nums)

    def get_val(node):
        if not node.nodes:
            return sum(node.metas)

        out = 0
        for i in node.metas:
            if len(node.nodes) < i or i == 0:
                continue
            out += get_val(node.nodes[i-1])
        return out

    return get_val(root)















FUN = p_8_b
print(run(FUN))
