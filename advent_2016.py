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
            new_pos = move(pos, dir_)
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
            new_pos = move(pos, dir_)
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


def p_4_b(data):
    def checksum(name):
        name = [a for a in name if a != '-']
        counter = Counter(name)
        name = sorted(counter.items(), key=lambda a: (-a[1], a[0]))
        name = [a[0] for a in name]
        return ''.join(name[:5])

    def dec_(ch, sec):
        if ch == '-':
            return ' '
        for _ in range(sec):
            if ch == 'z':
                ch = 'a'
            else:
                ch = chr(ord(ch) + 1)
        return ch

    def decrypt(name, sec):
        return ''.join(dec_(a, sec) for a in name)

    out = []
    for line in data:
        name = line[:-11]
        sec = line[-10:-7]
        out.append((decrypt(name, int(sec)), sec))
    for a in out:
        print(a)


def p_5_a():
    def five_zeros(i, id_):
        hex_ = hashlib.md5(f'{id_}{i}'.encode()).hexdigest()
        return hex_[:5] == '00000'

    in_ = 'ffykfhsq'
    out = []
    i = 0
    for _ in range(8):
        while not five_zeros(i, in_):
            i += 1
        c = hashlib.md5(f'{in_}{i}'.encode()).hexdigest()[5]
        out.append(c)
        print(i, c)
        i += 1
    return ''.join(out)


def p_5_b():
    def five_zeros(i, id_):
        hex_ = hashlib.md5(f'{id_}{i}'.encode()).hexdigest()
        return hex_[:5] == '00000'

    def none_none(list_):
        for a in list_:
            if a is None:
                return False
        return True

    in_ = 'ffykfhsq'
    out = [None] * 8
    i = 0
    while True:
        while not five_zeros(i, in_):
            i += 1
        pos = hashlib.md5(f'{in_}{i}'.encode()).hexdigest()[5]
        c = hashlib.md5(f'{in_}{i}'.encode()).hexdigest()[6]
        print(pos, c)
        if pos.isnumeric() and int(pos) <= 7 and out[int(pos)] is None:
            out[int(pos)] = c
            print('out', int(pos), c)
        i += 1
        if none_none(out):
            return ''.join(out)


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


def p_8_a(data):
    def rect(p):
        for x in range(p.x):
            for y in range(p.y):
                grid[P(x, y)] = True

    def rotate_row(y, by):
        for x in range(X_MAX):
            grid[P(x, y)] = None

    X_MAX = 50
    Y_MAX = 6
    grid = defaultdict(bool)














FUN = p_7_b
print(run(FUN, FILENAME_TEMPLATE))
