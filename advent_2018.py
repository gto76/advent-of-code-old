from collections import Counter, namedtuple, defaultdict
from copy import copy, deepcopy
from datetime import datetime
from enum import Enum, auto
from functools import reduce
import hashlib
from itertools import *
import json
from math import *
import operator as op
import re
import sys
from time import sleep

from util import *


FILENAME_TEMPLATE = 'data_2018/{}.data'


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
            if exists(line_out, data[i + 1:], j):
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
        for i in range(off.x, off.x + wid.x):
            for j in range(off.y, off.y + wid.y):
                cnt[P(i, j)] += 1

    cnt = defaultdict(int)
    for line in data:
        parse_line(line)
    out = 0
    for v in Bar.foreach(cnt.values()):
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
        for i in range(off.x, off.x + wid.x):
            for j in range(off.y, off.y + wid.y):
                cnt[P(i, j)].ids.append(id_)

    cnt = defaultdict(Square)
    ids = set()
    for line in data:
        parse_line(line)
    double_ids = set()
    for k, v in Bar.foreach(cnt.items()):
        if len(v.ids) > 1:
            double_ids.update(v.ids)
    return ids.difference(double_ids)


def p_4_a(data):
    class Event:
        def __init__(self, date_, text):
            self.date = date_
            self.text = text

    class Guard:
        def __init__(self, id_):
            self.id = id_
            self.shifts = []

        def get_minutes(self):
            return sum(a.get_minutes() for a in self.shifts)

        def get_minute(self):
            out = []
            for i in range(60):
                sum_ = sum(a.get_minute(i) for a in self.shifts)
                out.append(sum_)
            return out.index(max(out))

    class Shift:
        def __init__(self):
            self.ranges = []

        def set_asleep(self, minute):
            self.ranges.append([minute])

        def set_awake(self, minute):
            self.ranges[-1].append(minute)

        def get_minutes(self):
            return sum(a[1] - a[0] for a in self.ranges)

        def get_minute(self, minute):
            for range_ in self.ranges:
                if range_[0] <= minute < range_[1]:
                    return 1
            return 0

    def parse_line(line):
        date_ = datetime.strptime(line[1:17], '%Y-%m-%d %H:%M')
        text = line[19:]
        return Event(date_, text)

    def get_guards(timeline):
        out = {}
        guard = None
        for event in timeline:
            if 'Guard' in event.text:
                g_id = int(re.search(r'#(\d+)', event.text).group(1))
                guard = out.setdefault(g_id, Guard(g_id))
                guard.shifts.append(Shift())
            elif 'falls asleep' in event.text:
                guard.shifts[-1].set_asleep(event.date.minute)
            elif 'wakes up' in event.text:
                guard.shifts[-1].set_awake(event.date.minute)
        return out

    timeline = [parse_line(a) for a in data]
    timeline = sorted(timeline, key=lambda a: a.date)
    guards = get_guards(timeline)
    max_minutes, max_guard_id = max((a.get_minutes(), a.id)
                                    for a in guards.values())
    minute = guards[max_guard_id].get_minute()
    return max_guard_id * minute


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
    for i in range(65, 65 + 32):
        print(chr(i))
        line_b = line.replace(chr(i), '')
        line_b = line_b.replace(chr(i + 32), '')
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
    for y in Bar.range(b_1.y, b_2.y + 1):
        for x in range(b_1.x, b_2.x + 1):
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
        return sum(get_man(p, a) for a in coords)

    coords = []
    for line in data:
        parse_line(line)
    b_1, b_2 = get_bounds()
    out = 0
    for y in Bar.range(b_1.y, b_2.y + 1):
        for x in range(b_1.x, b_2.x + 1):
            sum_ = get_sum_of_distances(P(x, y))
            if sum_ < 10000:
                out += 1
    return out


def p_7_a(data):
    class Node:
        def __init__(self, id_):
            self.id_ = id_
            self.done = False
            self.parents = set()
            self.children = set()

    def parse_line(line):
        step_l = line[5]
        step_r = line[-12]
        n_l = nodes.setdefault(step_l, Node(step_l))
        n_r = nodes.setdefault(step_r, Node(step_r))
        n_l.children.add(n_r)
        n_r.parents.add(n_l)

    def get_available():
        out = []
        for node in nodes.values():
            if all(a.done for a in node.parents) and not node.done:
                out.append(node.id_)
        return sorted(out)

    nodes = {}
    for line in data:
        parse_line(line)
    out = []
    while True:
        available = get_available()
        if not available:
            return ''.join(out)
        next_ = available[0]
        out.append(next_)
        nodes[next_].done = True


def p_7_b(data):
    class Node:
        def __init__(self, id_):
            self.id_ = id_
            self.done = False
            self.parents = set()
            self.children = set()

    class Worker:
        def __init__(self):
            self.task = None
            self.countdown = 0

    def parse_line(line):
        step_l = line[5]
        step_r = line[-12]
        n_l = nodes.setdefault(step_l, Node(step_l))
        n_r = nodes.setdefault(step_r, Node(step_r))
        n_l.children.add(n_r)
        n_r.parents.add(n_l)

    def get_available():
        out = []
        for node in nodes.values():
            if node.id_ in [a.task for a in workers]:
                continue
            if all(a.done for a in node.parents) and not node.done:
                out.append(node.id_)
        return sorted(out)

    def iterate_workers():
        for worker in workers:
            if not worker.task:
                continue
            worker.countdown -= 1
            if worker.countdown == 0:
                nodes[worker.task].done = True
                worker.task = None

    def get_duration(id_):
        return ord(id_) - 4

    nodes = {}
    for line in data:
        parse_line(line)
    workers = []
    for _ in range(5):
        workers.append(Worker())

    i = -1
    while not all(a.done for a in nodes.values()):
        i += 1
        iterate_workers()
        available_workers = [a for a in workers if not a.task]
        if not available_workers:
            continue
        available_nodes = get_available()
        if not available_nodes:
            continue
        available_nodes = [nodes[a] for a in available_nodes]
        for worker, node in zip(available_workers, available_nodes):
            worker.task = node.id_
            worker.countdown = get_duration(node.id_)
    return i


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
    get_node(nums)
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

    def get_val(node):
        if not node.nodes:
            return sum(node.metas)
        out = 0
        for i in node.metas:
            if len(node.nodes) < i or i == 0:
                continue
            out += get_val(node.nodes[i - 1])
        return out

    line = data[0]
    nums = (int(a) for a in line.split())
    nodes = []
    root = get_node(nums)
    return get_val(root)


def p_9_a():
    class Marble:
        def __init__(self, id_):
            self.id_ = id_
            self.left = None
            self.right = None

    def rearange(i_m, i_p, last):
        players[i_p] += i_m
        for _ in range(7):
            last = last.left
        players[i_p] += last.id_
        lefty = last.left
        righty = last.right
        lefty.right = righty
        righty.left = lefty
        return righty

    def add(i_m, last):
        marble = Marble(i_m)
        next_ = last.right
        nextnext = next_.right
        next_.right = marble
        marble.left = next_
        marble.right = nextnext
        nextnext.left = marble
        return marble

    NO_PL = 471
    NO_MARBLES = 72026
    players = [0] * NO_PL
    last = Marble(0)
    last.left = last
    last.right = last
    for i_m, i_p in zip(range(1, NO_MARBLES + 1), cycle(range(NO_PL))):
        if i_m % 23 == 0:
            last = rearange(i_m, i_p, last)
            continue
        last = add(i_m, last)
    return max(players)


def p_9_b():
    class Marble:
        def __init__(self, id_):
            self.id_ = id_
            self.left = None
            self.right = None

    def rearange(i_m, i_p, last):
        players[i_p] += i_m
        for _ in range(7):
            last = last.left
        players[i_p] += last.id_
        lefty = last.left
        righty = last.right
        lefty.right = righty
        righty.left = lefty
        return righty

    def add(i_m, last):
        marble = Marble(i_m)
        next_ = last.right
        nextnext = next_.right
        next_.right = marble
        marble.left = next_
        marble.right = nextnext
        nextnext.left = marble
        return marble

    NO_PL = 471
    NO_MARBLES = 72026 * 100
    players = [0] * NO_PL
    last = Marble(0)
    last.left = last
    last.right = last
    bar = Bar(NO_MARBLES)
    for i_m, i_p in zip(range(1, NO_MARBLES + 1), cycle(range(NO_PL))):
        bar.tick()
        if i_m % 23 == 0:
            last = rearange(i_m, i_p, last)
            continue
        last = add(i_m, last)
    return max(players)


def p_10(data):
    class Point:
        def __init__(self, p, v):
            self.p = p
            self.v = v

    def parse_line(a):
        x, y, v_x, v_y = int(a[10:16]), int(a[18:24]), int(a[36:38]), \
                         int(a[40:42])
        point = Point(P(x, y), P(v_x, v_y))
        points.append(point)

    def move_point():
        for point in points:
            x = point.p.x + point.v.x
            y = point.p.y + point.v.y
            point.p = P(x, y)

    def ocupied(p):
        for point in points:
            if point.p == p:
                return True

    def print_p(p_min, p_max):
        for i in range(p_min.y, p_max.y):
            for j in range(p_min.x, p_max.x):
                if ocupied(P(j, i)):
                    print('#', end='')
                else:
                    print('.', end='')
            print()

    def points_together(points):
        max_x = max(a.p.x for a in points)
        min_x = min(a.p.x for a in points)
        max_y = max(a.p.y for a in points)
        min_y = min(a.p.y for a in points)
        if max_x - min_x < 100 and max_y - min_y < 10:
            return P(min_x, min_y), P(max_x, max_y)

    points = []
    for line in data:
        parse_line(line)
    for i in count():
        move_point()
        if points_together(points):
            print(i)
            p_min, p_max = points_together(points)
            print_p(p_min, p_max)


def p_13_a(data):
    class Cart:
        def __init__(self, p, d):
            self.p = p
            self.d = d
            self.r = R.l

        def __repr__(self):
            return f'{self.__dict__}'

    def get_carts():
        out = []
        for y, line in enumerate(data):
            for x, ch in enumerate(line):
                if ch in dd:
                    p = P(x, y)
                    d = dd[ch]
                    cart = Cart(p, d)
                    out.append(cart)
        return out

    def get_track():
        out = []
        cc = {'^': '|', 'v': '|', '>': '-', '<': '-'}
        for line in data:
            line_list = []
            for ch in line:
                if ch in dd:
                    ch = cc[ch]
                line_list.append(ch)
            out.append(line_list)
        return out

    def move_cart(cart):
        cart.d = get_dir(cart)
        p = move(cart.p, cart.d)
        if p in [a.p for a in carts]:
            return p
        cart.p = p

    def get_dir(cart):
        ch = track[cart.p.y][cart.p.x]
        if ch in '|-':
            return cart.d
        if ch == '+':
            out = turn(cart.d, cart.r)
            cart.r = R.l if cart.r == R.r else R(cart.r.value + 1)
            return out
        if ch in '\/':
            return tt[(ch, cart.d)]

    dd = get_dict('^>v<', D)
    tt = {('/', D.n): D.e, ('/', D.e): D.n, ('/', D.s): D.w, ('/', D.w): D.s,
          ('\\', D.n): D.w, ('\\', D.e): D.s, ('\\', D.s): D.e,
          ('\\', D.w): D.n}
    carts = get_carts()
    track = get_track()

    while True:
        carts = sorted(carts, key=lambda a: (a.p.y, a.p.x))
        for cart in carts:
            collision_p = move_cart(cart)
            if collision_p:
                return f'{collision_p.x},{collision_p.y}'


def p_13_b():
    # class Cart:
    #     def __init__(self, p, d):
    #         self.p = p
    #         self.d = d
    #         self.r = R.l
    #     def __repr__(self):
    #         return f'{self.__dict__}'
    #
    # def get_carts():
    #     out = []
    #     for y, line in enumerate(data):
    #         for x, ch in enumerate(line):
    #             if ch in dd:
    #                 p = P(x, y)
    #                 d = dd[ch]
    #                 cart = Cart(p, d)
    #                 out.append(cart)
    #     return out
    #
    # def get_track():
    #     out = []
    #     cc = {'^': '|', 'v': '|', '>': '-', '<': '-'}
    #     for line in data:
    #         line_list = []
    #         for ch in line:
    #             if ch in dd:
    #                 ch = cc[ch]
    #             line_list.append(ch)
    #         out.append(line_list)
    #     return out
    #
    # def move_cart(cart):
    #     cart.d = get_dir(cart)
    #     p = move(cart.p, cart.d)
    #     if p in [a.p for a in carts]:
    #         return p
    #     cart.p = p
    #
    # def get_dir(cart):
    #     ch = track[cart.p.y][cart.p.x]
    #     if ch in '|-':
    #         return cart.d
    #     if ch == '+':
    #         out = turn(cart.d, cart.r)
    #         cart.r = R.l if cart.r == R.r else R(cart.r.value+1)
    #         return out
    #     if ch in '\/':
    #         return tt[(ch, cart.d)]
    #
    # dd = get_dict('^>v<', D)
    # tt = {('/', D.n): D.e, ('/', D.e): D.n, ('/', D.s): D.w, ('/', D.w): D.s,
    #       ('\\', D.n): D.w, ('\\', D.e): D.s, ('\\', D.s): D.e,
    #       ('\\', D.w): D.n}
    # carts = get_carts()
    # track = get_track()
    #
    # while True:
    #     if len(carts) == 1:
    #         return f'{carts[0].p.x},{carts[0].p.y}'
    #     carts = sorted(carts, key=lambda a: (a.p.y, a.p.x))
    #     carts = [move_cart(a) for a in carts]
    pass


def p_16_a(data):
    class Observation:
        def __init__(self):
            self.before = None
            self.comm = None
            self.after = None

    class Command(Enum):
        addr = auto(), lambda r, a, b: r[a] + r[b]
        addi = auto(), lambda r, a, b: r[a] + b
        mulr = auto(), lambda r, a, b: r[a] * r[b]
        muli = auto(), lambda r, a, b: r[a] * b
        banr = auto(), lambda r, a, b: r[a] & r[b]
        bani = auto(), lambda r, a, b: r[a] & b
        borr = auto(), lambda r, a, b: r[a] | r[b]
        bori = auto(), lambda r, a, b: r[a] | b
        setr = auto(), lambda r, a, b: r[a]
        seti = auto(), lambda r, a, b: a
        gtir = auto(), lambda r, a, b: 1 if a > r[b] else 0
        gtri = auto(), lambda r, a, b: 1 if r[a] > b else 0
        gtrr = auto(), lambda r, a, b: 1 if r[a] > r[b] else 0
        eqir = auto(), lambda r, a, b: 1 if a == r[b] else 0
        eqri = auto(), lambda r, a, b: 1 if r[a] == b else 0
        eqrr = auto(), lambda r, a, b: 1 if r[a] == r[b] else 0

    def parse():
        out = []
        iter_ = iter(data)
        while True:
            obs = get_observation(iter_)
            if not obs:
                break
            out.append(obs)
            next(iter_)
        return out

    def get_observation(d_iter):
        obs = Observation()
        before_line = next(d_iter, '')
        if 'Before' not in before_line:
            return
        obs.before = get_list(before_line)
        obs.comm = [int(a) for a in next(d_iter).split()]
        after_line = next(d_iter)
        obs.after = get_list(after_line)
        return obs

    def get_list(line):
        before_list = re.search('\[(.*)\]', line).group(1).split(',')
        return [int(a) for a in before_list]

    def get_no_matches(obs):
        out = 0
        for op in list(Command):
            if check(obs, op):
                out += 1
        return out

    def check(obs, op):
        op = op.value[1]
        res = op(obs.before, obs.comm[1], obs.comm[2])
        after = list(obs.before)
        after[obs.comm[3]] = res
        return after == obs.after

    observations = parse()
    out = 0
    for obs in observations:
        if get_no_matches(obs) >= 3:
            out += 1
    return out


def p_16_b(data):
    class Observation:
        def __init__(self):
            self.before = None
            self.inst = None
            self.after = None

    class Operation(Enum):
        addr = auto(), lambda r, a, b: r[a] + r[b]
        addi = auto(), lambda r, a, b: r[a] + b
        mulr = auto(), lambda r, a, b: r[a] * r[b]
        muli = auto(), lambda r, a, b: r[a] * b
        banr = auto(), lambda r, a, b: r[a] & r[b]
        bani = auto(), lambda r, a, b: r[a] & b
        borr = auto(), lambda r, a, b: r[a] | r[b]
        bori = auto(), lambda r, a, b: r[a] | b
        setr = auto(), lambda r, a, b: r[a]
        seti = auto(), lambda r, a, b: a
        gtir = auto(), lambda r, a, b: 1 if a > r[b] else 0
        gtri = auto(), lambda r, a, b: 1 if r[a] > b else 0
        gtrr = auto(), lambda r, a, b: 1 if r[a] > r[b] else 0
        eqir = auto(), lambda r, a, b: 1 if a == r[b] else 0
        eqri = auto(), lambda r, a, b: 1 if r[a] == b else 0
        eqrr = auto(), lambda r, a, b: 1 if r[a] == r[b] else 0

    def parse():
        out = []
        iter_ = iter(data)
        while True:
            obs = get_observation(iter_)
            if not obs:
                break
            out.append(obs)
            next(iter_)
        next(iter_)
        return out, list(iter_)

    def get_observation(d_iter):
        obs = Observation()
        before_line = next(d_iter, '')
        if 'Before' not in before_line:
            return
        obs.before = get_list(before_line)
        obs.inst = [int(a) for a in next(d_iter).split()]
        after_line = next(d_iter)
        obs.after = get_list(after_line)
        return obs

    def get_list(line):
        before_list = re.search('\[(.*)\]', line).group(1).split(',')
        return [int(a) for a in before_list]

    def get_operations(obs):
        out = set()
        for op in list(Operation):
            if check(obs, op):
                out.add(op)
        return out

    def check(obs, op):
        op = op.value[1]
        res = op(obs.before, obs.inst[1], obs.inst[2])
        after = list(obs.before)
        after[obs.inst[3]] = res
        return after == obs.after

    def first_pass(observations):
        out = {}
        for obs in observations:
            opcode = obs.inst[0]
            operations = get_operations(obs)
            if opcode not in out:
                out[opcode] = operations
            else:
                out[opcode] = out[opcode].intersection(operations)
        return out

    def deduce(opcodes):
        found = {list(a)[0] for a in opcodes.values() if len(a) == 1}
        for operations in opcodes.values():
            if len(operations) == 1:
                continue
            striped = operations.difference(found)
            operations.clear()
            operations.update(striped)

    def no_of_op(opcodes):
        return len([a for set_ in opcodes.values() for a in set_])

    observations, instructions = parse()
    opcodes = first_pass(observations)
    while no_of_op(opcodes) != len(opcodes):
        deduce(opcodes)
    regs = [0, 0, 0, 0]
    for inst in instructions:
        inst = [int(a) for a in inst.split()]
        op = list(opcodes[inst[0]])[0].value[1]
        regs[inst[3]] = op(regs, inst[1], inst[2])
    return regs[0]


def p_17_a(data):
    Matter = Enum('Matter', 'clay watter')

    def parse_line(line):
        x = parse_coordinate(line, 'x')
        y = parse_coordinate(line, 'y')
        return P(x, y)

    def parse_coordinate(line, coord_id):
        str_ = re.search(coord_id + r'=([0-9.]+)', line).group(1)
        if '..' in str_:
            start, stop = (int(a) for a in str_.split('..'))
            return range(start, stop + 1)
        return int(str_)

    def get_squares(veins):
        out = {}
        for vein in veins:
            if type(vein.x) == range:
                positions = [P(x, vein.y) for x in vein.x]
            else:
                positions = [P(vein.x, y) for y in vein.y]
            for pos in positions:
                out[pos] = Matter.clay
        return out

    def flow(pos):
        if pos.y > max_y:
            return True
        squares[pos] = Matter.watter
        if try_direction(pos, D.s):
            return True
        left = try_direction(pos, D.w)
        right = try_direction(pos, D.e)
        return left or right

    def try_direction(pos, dir_):
        new_pos = move(pos, dir_)
        if new_pos in squares:
            return False
        return flow(new_pos)

    old_rec_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(10000)
    veins = [parse_line(a) for a in data]
    squares = get_squares(veins)
    min_y, max_y = min(a.y for a in squares), max(a.y for a in squares)
    flow(P(500, 0))
    sys.setrecursionlimit(old_rec_limit)

    # for y in range(14):
    #     for x in range(494, 508):
    #         p = P(x, y)
    #         if p not in squares:
    #             print('.', end='')
    #         elif squares[p] == Matter.clay:
    #             print('#', end='')
    #         else:
    #             print('~', end='')
    #     print()

    def get_p(p):
        if p not in squares:
            return 0
        if squares[p] == Matter.clay:
            return 100
        return 202

    min_x, max_x = min(a.x for a in squares), max(a.x for a in squares)
    width, height = max_x - min_x + 1, max_y + 1
    out = [0] * width * height
    for p, m in squares.items():
        out[p.y*width + p.x-min_x] = get_p(p)

    from PIL import Image
    new_img = Image.new("L", (width, height), "white")
    new_img.putdata(out)
    new_img.save('out.png')

    return sum(1 for p, m in squares.items()
               if m == Matter.watter and p.y >= min_y)


FUN = p_17_a
print(run(FUN, FILENAME_TEMPLATE))
