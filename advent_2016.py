#!/usr/bin/env python3

from collections import Counter, namedtuple, defaultdict, deque
from copy import copy, deepcopy
from enum import Enum
from functools import reduce
import hashlib
from inspect import signature
from itertools import *
import json
from math import *
import operator as op
from operator import itemgetter
import re
import sys
from time import sleep, time
from tqdm import tqdm
from functools import partial
from time import perf_counter


from util import *


FILENAME_TEMPLATE = 'data_2016/{}.data'


###
##  PROBLEMS
#

def p_1_a(data):
    commands = data[0].split(', ')
    ROTS = get_dict('LR', R)
    p = P(0, 0)
    d = D.n
    for c in commands:
        r = ROTS[c[0]]
        d = TURN[(d, r)]
        fac = int(c[1:])
        p_n = MOVE[d]
        p_n = P(p_n.x * fac, p_n.y * fac)
        p = P(p.x + p_n.x, p.y + p_n.y)
    return abs(p.x) + abs(p.y)


def p_1_b(data):
    commands = data[0].split(', ')
    ROTS = get_dict('LR', R)
    visited = []
    p = P(0, 0)
    visited.append(p)
    d = D.n
    for c in commands:
        r = ROTS[c[0]]
        d = TURN[(d, r)]
        fac = int(c[1:])
        p_n = MOVE[d]
        for _ in range(fac):
            p = P(p.x + p_n.x, p.y + p_n.y)
            if p in visited:
                return abs(p.x) + abs(p.y)
            visited.append(p)


def p_2_a(data):
    MV = get_dict('URDL', D)
    PAD = {P(-1, 1): 5, P(0, 1): 2, P(1, 1): 3,
           P(-1, 0): 4, P(0, 0): 5, P(1, 0): 6,
           P(-1, -1): 7, P(0, -1): 8, P(1, -1): 9}

    pos = P(0, 0)
    out = []
    for line in data:
        for command in line:
            dir_ = MV[command]
            new_pos = move_mat(pos, dir_)
            if new_pos not in PAD:
                continue
            pos = new_pos
        out.append(PAD[pos])

    return ''.join(str(a) for a in out)


def p_2_b(data):
    # MV = {'U': D.n, 'D': D.s, 'L': D.w, 'R': D.e}
    MV = get_dict('URDL', D)
    PAD = {P(-2, 0): 5, P(0, 2): 1, P(2, 0): 9, P(0, -2): 'D',
           P(-1, 1): 2, P(0, 1): 3, P(1, 1): 4, 
           P(-1, 0): 6, P(0, 0): 7, P(1, 0): 8, 
           P(-1, -1): 'A', P(0, -1): 'B', P(1, -1): 'C'}

    pos = P(0, 0)
    out = []
    for line in data:
        for command in line:
            dir_ = MV[command]
            new_pos = move_mat(pos, dir_)
            if new_pos not in PAD:
                continue
            pos = new_pos
        out.append(PAD[pos])

    return ''.join(str(a) for a in out)


def p_3_a(data):
    def possible(s):
        return s[0] + s[1] > s[2] and s[1] + s[2] > s[0] and s[0] + s[2] > s[1]
    out = 0
    for line in data:
        sides = [int(a) for a in line.split()]
        if possible(sides):
            out += 1
    return out


def p_3_b(data):
    def possible(s):
        return s[0] + s[1] > s[2] and s[1] + s[2] > s[0] and s[0] + s[2] > s[1]

    aaa = []
    for i in range(3):
        for line in data:
            sides = [int(a) for a in line.split()]
            aaa.append(sides[i])

    triangles = []
    for i in range(0, len(aaa), 3):
        triangle = aaa[i:i+3]
        triangles.append(triangle)

    out = 0
    for triangle in triangles:
        sides = triangle
        if possible(sides):
            out += 1
    return out


def p_4_a(data):
    def checksum(name):
        name = [a for a in name if a != '-']
        counter = Counter(name)
        name = sorted(counter.items(), key=lambda a: (-a[1], a[0]))
        name = [a[0] for a in name]
        return ''.join(name[:5])

    out = 0
    for line in data:
        name = line[:-11]
        sum_ = line[-6:-1]
        sec = line[-10:-7]
        test = checksum(name)
        if test == sum_:
            out += int(sec)
    return out


# def p_4_b(data):
#     def checksum(name):
#         name = [a for a in name if a != '-']
#         counter = Counter(name)
#         name = sorted(counter.items(), key=lambda a: (-a[1], a[0]))
#         name = [a[0] for a in name]
#         return ''.join(name[:5])
#
#     def dec_(ch, sec):
#         if ch == '-':
#             return ' '
#         for _ in range(sec):
#             if ch == 'z':
#                 ch = 'a'
#             else:
#                 ch = chr(ord(ch) + 1)
#         return ch
#
#     def decrypt(name, sec):
#         return ''.join(dec_(a, sec) for a in name)
#
#     out = []
#     for line in data:
#         name = line[:-11]
#         sec = line[-10:-7]
#         out.append((decrypt(name, int(sec)), sec))
#     for a in out:
#         print(a)


def p_5_a():
    def five_zeros(i, id_):
        hex_ = hashlib.md5(f'{id_}{i}'.encode()).hexdigest()
        return hex_[:5] == '00000'

    in_ = 'ffykfhsq'
    out = []
    i = 0
    for _ in tqdm(range(8)):
        while not five_zeros(i, in_):
            i += 1
        c = hashlib.md5(f'{in_}{i}'.encode()).hexdigest()[5]
        out.append(c)
        i += 1
    return ''.join(out)


# def p_5_b():
#     def five_zeros(i, id_):
#         hex_ = hashlib.md5(f'{id_}{i}'.encode()).hexdigest()
#         return hex_[:5] == '00000'
#
#     def none_none(list_):
#         for a in list_:
#             if a is None:
#                 return False
#         return True
#
#     in_ = 'ffykfhsq'
#     out = [None] * 8
#     i = 0
#     while True:
#         while not five_zeros(i, in_):
#             i += 1
#         pos = hashlib.md5(f'{in_}{i}'.encode()).hexdigest()[5]
#         c = hashlib.md5(f'{in_}{i}'.encode()).hexdigest()[6]
#         if pos.isnumeric() and int(pos) <= 7 and out[int(pos)] is None:
#             out[int(pos)] = c
#         i += 1
#         if none_none(out):
#             return ''.join(out)


def p_6_a(data):
    out = []

    for _ in range(len(data[0])):
        out.append([])

    for line in data:
        for i, ch in enumerate(line):
            out[i].append(ch)

    outout = []
    for aaa in out:
        counter = Counter(aaa)
        outout.append(counter.most_common()[0][0])

    return ''.join(outout)


def p_6_b(data):
    out = []

    for _ in range(len(data[0])):
        out.append([])

    for line in data:
        for i, ch in enumerate(line):
            out[i].append(ch)

    outout = []
    for aaa in out:
        counter = Counter(aaa)
        outout.append(counter.most_common()[-1][0])

    return ''.join(outout)


def p_7_a(data):
    def has_abba(text):
        last_3 = []
        for c in text:
            if len(last_3) < 3:
                last_3.append(c)
                continue
            if c == last_3[0] and last_3[1] == last_3[2] \
                    and last_3[0] != last_3[1]:
                return True
            last_3.append(c)
            last_3.pop(0)
        return False

    ooo = 0

    for line in data:
        tok = [a for a in re.split('[\[\]]', line) if a]
        ln = len(tok)
        out = tok[0:ln:2]
        in_ = tok[1:ln:2]

        has_out = False
        for o in out:
            if has_abba(o):
                has_out = True
        if not has_out:
            continue

        has_in = False
        for o in in_:
            if has_abba(o):
                has_in = True
        if has_in:
            continue
        ooo += 1

    return ooo


def p_7_b(data):
    def has_abba(text):
        last_3 = []
        for c in text:
            if len(last_3) < 3:
                last_3.append(c)
                continue
            if c == last_3[0] and last_3[1] == last_3[2] \
                    and last_3[0] != last_3[1]:
                return True
            last_3.append(c)
            last_3.pop(0)
        return False
    def find_abas(sectors):
        out = []
        last_3 = []
        for sector in sectors:
            for c in sector:
                if len(last_3) < 2:
                    last_3.append(c)
                    continue
                if c == last_3[0] and last_3[1] != c:
                    out.append(''.join(last_3+[c]))
                last_3.append(c)
                last_3.pop(0)
        return out

    def baba_exists(in_, aba):
        for sector in in_:
            if aba in sector:
                return True
        return False

    ooo = 0

    for line in data:
        tok = [a for a in re.split('[\[\]]', line) if a]
        ln = len(tok)
        out = tok[0:ln:2]
        in_ = tok[1:ln:2]

        abas = find_abas(out)

        for aba in abas:
            if baba_exists(in_, aba[1]+aba[0]+aba[1]):
                ooo += 1
                continue

    return ooo


# def p_8_a(data):
#     def rect(p):
#         for x in range(p.x):
#             for y in range(p.y):
#                 grid[P(x, y)] = True
#
#     def rotate_row(y, by):
#         for x in range(X_MAX):
#             grid[P(x, y)] = None
#
#     X_MAX = 50
#     Y_MAX = 6
#     grid = defaultdict(bool)


def p_9_a(data):
    def process_marker(i, out, line):
        match = re.match('\((\d+)x(\d+)\)', line[i:])
        size, repeat = [int(a) for a in match.groups()]
        i += match.end() + size
        out += size * repeat
        return i, out
    out = 0
    i = 0
    line = data[0]
    while i < len(line):
        if line[i] == '(':
            i, out = process_marker(i, out, line)
        else:
            out += 1
            i += 1
    return out


def p_12_a(data, regs=None):
    def cpy(val, addr):
        val = int(val) if val.isnumeric() else regs[val]
        regs[addr] = val
        return 1
    def inc(addr):
        regs[addr] += 1
        return 1
    def dec(addr):
        regs[addr] -= 1
        return 1
    def jzn(val, delta):
        val = int(val) if val.isnumeric() else regs[val]
        return int(delta) if val != 0 else 1
    Inst = Enum('Inst', {'cpy': partial(cpy), 
                         'inc': partial(inc), 
                         'dec': partial(dec), 
                         'jnz': partial(jzn)})
    if regs is None:
        regs = defaultdict(lambda: 0)
    pc = 0
    while 0 <= pc < len(data):
        inst_code, *args = data[pc].split()
        inst = Inst[inst_code].value
        pc += inst(*args)
    return regs['a']


def p_12_b(data):
    regs = defaultdict(lambda: 0)
    regs['c'] = 1
    res = p_12_a(data, regs)
    return res




def p_13_a(data):
    from collections.abc import Mapping
    class Maze(Mapping):
        def __init__(self, seed):
            self.seed = seed
            self.a = {}
        def __len__(self):
            return len(self.a)
        def __getitem__(self, p):
            if p in self.a:
                return self.a[p]
            value_int = p.x**2 + 3*p.x + 2*p.x*p.y + p.y + p.y**2 + seed
            value_bin = bin(value_int)[2:]
            value = value_bin.count('1') % 2 == 0
            self.a[p] = value
            return value
        def __iter__(self):
            return iter(self.a)

    seed = int(data[0])
    maze = Maze(seed)
    start = P(1, 1)
    goal = P(31, 39)
    visited = set()
    current_points = set([start])
    next_points = set()

    for i in count(1):
        visited.update(current_points)
        while current_points:
            point = current_points.pop()
            neighbours = get_four_neighbours(point, a_min=P(0, 0))
            neighbours = {p for p in neighbours if maze[p]}
            if goal in neighbours:
                return i
            if not neighbours:
                continue
            next_points.update(neighbours)
        current_points = next_points - visited
        if not current_points:
            return


def p_13_b(data):
    def is_free(p):
        if p in maze:
            return maze[p]
        value_int = p.x**2 + 3*p.x + 2*p.x*p.y + p.y + p.y**2 + seed
        value_bin = bin(value_int)[2:]
        value = value_bin.count('1') % 2 == 0
        maze[p] = value
        return value

    seed = int(data[0])
    maze = {}
    start = P(1, 1)
    goal = P(31, 39)
    visited = set([start])
    current_points = set([start])
    next_points = set()

    for _ in range(50):
        while current_points:
            point = current_points.pop()
            neighbours = get_four_neighbours(point, a_min=P(0, 0))
            neighbours = {p for p in neighbours if is_free(p)}
            if not neighbours:
                continue
            next_points.update(neighbours)
        current_points = next_points - visited
        visited.update(current_points)
        if not current_points:
            return

    return len(visited)


def p_18_a(data):
    def get_cell(row, i):
        l = True if i == 0 else row[i-1]
        c = row[i]
        r = True if i == len(row) - 1 else row[i+1]
        trap_reqs = ((False, False, True),
                     (True, False, False),
                     (False, True, True),
                     (True, True, False))
        return (l, c, r) not in trap_reqs
    line = data[0]
    N_ROWS = 40
    out = [[a == '.' for a in line]]
    while len(out) < N_ROWS:
        last_row = out[-1]
        new_row = [get_cell(last_row, i) for i in range(len(last_row))]
        out.append(new_row)
    return sum(chain.from_iterable(out))



def p_18_b(data):
    def get_cell(row, i):
        l = True if i == 0 else row[i-1]
        c = row[i]
        r = True if i == len(row) - 1 else row[i+1]
        trap_reqs = ((False, False, True),
                     (True, False, False),
                     (False, True, True),
                     (True, True, False))
        return (l, c, r) not in trap_reqs
    N_ROWS = 400000
    row = [a == '.' for a in data[0]]
    out = sum(row)
    for _ in tqdm(range(N_ROWS-1)):
        row = [get_cell(row, i) for i in range(len(row))]
        out += sum(row)
    return out


def p_20_a(data):
    def is_blocked(ip):
        for r in ranges:
            if ip in r:
                return True
    lines = [a.split('-') for a in data]
    ranges = [range(int(lo), int(hi)+1) for lo, hi in lines]
    ranges.sort(key=itemgetter(1))
    for r in ranges:
        ip = r.stop
        if not is_blocked(ip):
            return ip


def p_20_b(data):
    def is_blocked(ip):
        for r in ranges:
            if ip in r:
                return True
    def find_next_start(ip):
        a_min = 4294967296
        for r in ranges:
            if ip < r.start < a_min:
                a_min = r.start
        return a_min
    lines = [a.split('-') for a in data]
    ranges = [range(int(lo), int(hi)+1) for lo, hi in lines]
    ranges.sort(key=itemgetter(1))
    out = 0
    for r in ranges:
        ip = r.stop
        if is_blocked(ip):
            continue
        stop = find_next_start(ip)
        out += stop - ip
    return out

    # Za vsak stop ki ni blocked:
    #     najdi naslednji start
    #     dodaj k sumu njuno razliko


def p_21_a(data):
    def swap_position(args, word):
        x, y = int(args[0]), int(args[3])
        word[x], word[y] = word[y], word[x]
    def swap_letter(args, word):
        x, y = word.index(args[0]), word.index(args[-1])
        word[x], word[y] = word[y], word[x]
    def rotate_left(args, word):
        a_deque = deque(word, maxlen=len(word))
        a_deque.rotate(-int(args[0]))
        word.clear()
        word.extend(a_deque)
    def rotate_right(args, word):
        a_deque = deque(word, maxlen=len(word))
        a_deque.rotate(int(args[0]))
        word.clear()
        word.extend(a_deque)
    def rotate_based(args, word):
        i = word.index(args[-1])
        a_deque = deque(word, maxlen=len(word))
        a_deque.rotate(1+i)
        if i >= 4:
            a_deque.rotate()
        word.clear()
        word.extend(a_deque)
    def reverse_positions(args, word):
        x, y = int(args[0]), int(args[-1])
        substring = word[y:x-1:-1] if x > 0 else word[y::-1]
        word[x:y+1] = substring
    def move_position(args, word):
        letter = word.pop(int(args[0]))
        word.insert(int(args[-1]), letter)
    commands = {'swap position': swap_position,
                'swap letter': swap_letter,
                'rotate left': rotate_left,
                'rotate right': rotate_right,
                'rotate based': rotate_based,
                'reverse positions': reverse_positions,
                'move position': move_position}
    word = list('abcdefgh')
    for command in data:
        tokens = command.split()
        first_words, args = ' '.join(tokens[0:2]), tokens[2:]
        commands[first_words](args, word)
    return ''.join(word)


def p_22_a(data):
    def get_node(line):
        tokens = line.split()
        p = P(*re.search('(\d+).*(\d+)', tokens[0]).groups())
        return Node(p, *(int(a[:-1]) for a in tokens[1:]))
    Node = namedtuple('Node', 'p size used avail use')
    nodes = [get_node(line) for line in data[2:]]
    out = 0
    for l, r in product(nodes, nodes):
        if (l.used == 0) or (l is r) or (l.used > r.avail):
            continue
        out += 1
    return out


def p_24_a(data):
    def parse_line(line, y):
        for x, ch in enumerate(line):
            p = P(x, y)
            maze[p] = ch != '#'
            if ch not in '#.':
                checkpoints[int(ch)] = p
    def get_distance(start, goal):
        visited = set()
        current_points = set([start])
        next_points = set()
        for i in count(1):
            visited.update(current_points)
            while current_points:
                point = current_points.pop()
                neighbours = get_four_neighbours(point)
                neighbours = {p for p in neighbours if maze[p]}
                if goal in neighbours:
                    return i
                if not neighbours:
                    continue
                next_points.update(neighbours)
            current_points = next_points - visited
            if not current_points:
                return
    def get_length(path):
        out = 0
        last = 0
        for a_next in path:
            out += distances[P(last, a_next)]
            last = a_next
        return out
    maze = defaultdict(bool)
    checkpoints = {}
    distances = {}
    for i, line in enumerate(data):
        parse_line(line, i)
    for a, b in combinations(checkpoints, 2):
        distance = get_distance(checkpoints[a], checkpoints[b])
        distances[P(a, b)] = distance
        distances[P(b, a)] = distance
    out = inf
    for path in permutations(range(1, len(checkpoints)), len(checkpoints)-1):
        length = get_length(path)
        if length < out:
            out = length
    return(out)


def p_24_b(data):
    def parse_line(line, y):
        for x, ch in enumerate(line):
            p = P(x, y)
            maze[p] = ch != '#'
            if ch not in '#.':
                checkpoints[int(ch)] = p
    def get_distance(start, goal):
        visited, next_points = set(), set()
        current_points = set([start])
        for i in count(1):
            visited.update(current_points)
            while current_points:
                point = current_points.pop()
                neighbours = get_four_neighbours(point)
                neighbours = {p for p in neighbours if maze[p]}
                if goal in neighbours:
                    return i
                next_points.update(neighbours)
            current_points = next_points - visited
    def get_length(path):
        out, last = 0, 0
        for a_next in path:
            out += distances[P(last, a_next)]
            last = a_next
        return out
    maze = defaultdict(bool)
    checkpoints = {}
    distances = {}
    for i, line in enumerate(data):
        parse_line(line, i)
    for a, b in combinations(checkpoints, 2):
        distance = get_distance(checkpoints[a], checkpoints[b])
        distances[P(a, b)] = distance
        distances[P(b, a)] = distance
    out = inf
    for path in permutations(range(1, len(checkpoints)), len(checkpoints)-1):
        path += (0, )
        length = get_length(path)
        if length < out:
            out = length
    return(out)








###
##  UTIL
#

def main():
    script_name = sys.argv[0]
    arguments   = sys.argv[1:]
    if len(sys.argv) == 1:
        run_all()
    else:
        fun_name = sys.argv[1]
        fun = globals()[fun_name]
        print(run(fun, FILENAME_TEMPLATE))


def run_all():
    functions = [(k, v) for k, v in globals().items()
                 if callable(v) and k.startswith('p_')]
    for name, fun in functions:
        print(f'{name}:')
        start_time = time()
        print(f'Result:   {run(fun, FILENAME_TEMPLATE)}')
        duration = time() - start_time
        print(f'Duration: {duration:.3f}s\n')


if __name__ == '__main__':
    main()