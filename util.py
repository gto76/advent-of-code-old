from enum import Enum
from collections import namedtuple, defaultdict

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

def bit_not(n, numbits=8):
    return (1 << numbits) - 1 - n

