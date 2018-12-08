from collections import namedtuple, defaultdict
from enum import Enum
import sys

# Position Tuple
P = namedtuple('P', 'x y')

# Direction Enum
D = Enum('D', 'n e s w')

# Rotation Enum
R = Enum('R', 'l r')

# Move dict
MOVE = {D.n: P(0, 1), D.e: P(1, 0), D.s: P(0, -1), D.w: P(-1, 0)}

def move(position, direction):
    return P(*[sum(a) for a in zip(position, MOVE[direction])])

# Rotate dict
TURN = {(D.n, R.l): D.w, (D.n, R.r): D.e,
        (D.e, R.l): D.n, (D.e, R.r): D.s,
        (D.s, R.l): D.e, (D.s, R.r): D.w,
        (D.w, R.l): D.s, (D.w, R.r): D.n}

def get_dict(list_a, list_b):
    return dict(zip(list_a, list_b))

def bit_not(n, numbits=8):
    return (1 << numbits) - 1 - n

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
