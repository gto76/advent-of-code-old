from collections import namedtuple, defaultdict
from copy import deepcopy
from enum import Enum
from inspect import signature
import sys

# Position Tuple
P = namedtuple('P', 'x y')

# 3D Position Tuple
PP = namedtuple('PP', 'x y z')

# Direction Enum
D = Enum('D', 'n e s w')

# Double Direction Enum
DD = Enum('DD', 'n ne e se s sw w nw')

# Rotation Enum
R = Enum('R', 'l s r')

def get_manhattan(p_1, p_2):
    return sum(abs(a - b) for a, b in zip(p_1, p_2))

# Move dict
MOVE_MAT = {D.n: P(0, 1), D.e: P(1, 0), D.s: P(0, -1), D.w: P(-1, 0),
            DD.n: P(0, 1), DD.e: P(1, 0), DD.s: P(0, -1), DD.w: P(-1, 0),
            DD.ne: P(1, 1), DD.se: P(1, -1), DD.sw: P(-1, -1), DD.nw: P(-1, 1)}

def move_mat(position, direction):
    return P(*[sum(a) for a in zip(position, MOVE_MAT[direction])])

def move_mat_all(position, directions):
    return [move_mat(position, dir_) for dir_ in directions]

# Move dict
MOVE = {D.n: P(0, -1), D.e: P(1, 0), D.s: P(0, 1), D.w: P(-1, 0),
        DD.n: P(0, -1), DD.e: P(1, 0), DD.s: P(0, 1), DD.w: P(-1, 0),
        DD.ne: P(1, -1), DD.se: P(1, 1), DD.sw: P(-1, 1), DD.nw: P(-1, -1)}

def move(position, direction):
    return P(*[sum(a) for a in zip(position, MOVE[direction])])

def move_all(position, directions):
    return [move(position, dir_) for dir_ in directions]

def above(position):
    return move(position, D.n)

def on_right(position):
    return move(position, D.e)

def below(position):
    return move(position, D.s)

def on_left(position):
    return move(position, D.w)

# Rotate dict
TURN = {(D.n, R.l): D.w, (D.n, R.r): D.e, (D.n, R.s): D.n,
        (D.e, R.l): D.n, (D.e, R.r): D.s, (D.e, R.s): D.e,
        (D.s, R.l): D.e, (D.s, R.r): D.w, (D.s, R.s): D.s,
        (D.w, R.l): D.s, (D.w, R.r): D.n, (D.w, R.s): D.w}

def turn(direction, rotation):
    return TURN[(direction, rotation)]

def get_dict(list_a, list_b):
    return dict(zip(list_a, list_b))

def bit_not(n, numbits=8):
    return (1 << numbits) - 1 - n

def init_matrix(x, y, default):
    out = []
    for _ in range(y):
        line = []
        for _ in range(x):
            line.append(deepcopy(default))
        out.append(line)
    return out

class Bar:
    @staticmethod
    def range(*args):
        bar = Bar(len(list(range(*args))))
        for i in range(*args):
            yield i
            bar.tick()

    @staticmethod
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

    @staticmethod
    def p(t):
        sys.stdout.write(t)
        sys.stdout.flush()

def run(fun, filename_template):
    sig = signature(fun)
    no_parm = len(sig.parameters)
    if no_parm == 0:
        return fun()
    problem_name = fun.__name__
    if problem_name.endswith('_a') or problem_name.endswith('_b'):
        problem_name = problem_name[:-2]
    data = get_data(problem_name, filename_template)
    return fun(data)


def get_data(problem_name, filename_template):
    try:
        return get_file_contents(filename_template.format(problem_name))
    except FileNotFoundError:
        print("Missing data file {}".format(
            DATA_FILENAME.format(problem_name)), file=sys.stderr)
        return


def get_file_contents(file_name):
    with open(file_name, encoding='utf-8') as f:
        return [line.strip('\n') for line in f.readlines()]
