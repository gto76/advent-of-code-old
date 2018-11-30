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


DATA_FILENAME = 'data_2015/{}.data'

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

def p_1(data):
    out_a, out_b = 0, None
    for i, a in enumerate(data, 1):
        if a == '(':
            out_a += 1
        else:
            out_a -= 1
        if out_a == -1 and not out_b:
            out_b = i
    return out_a, out_b


def p_2(data):
    def paper_for_box(l, w, h):
        x, y, z = l * w, w * h, h * l
        return 2 * x + 2 * y + 2 * z + min(x, y, z)

    def ribbon_for_box(box):
        box.sort()
        return 2 * box[0] + 2 * box[1] + box[0] * box[1] * box[2]

    lines = data.split()
    int_data = [[int(side) for side in box.split(sep='x')] for box in lines]
    out_a = sum([paper_for_box(*box) for box in int_data])
    out_b = sum([ribbon_for_box(box) for box in int_data])
    return out_a, out_b


def p_3(data):
    D = Enum('D', {'n': '^', 'e': '>', 's': 'v', 'w': '<'})
    MOVE = {D.n: P(-1, 0), D.e: P(0, 1), D.s: P(1, 0), D.w: P(0, -1)}

    def move(position, direction):
        return P(*[sum(a) for a in zip(position, MOVE[direction])])

    def get_visited_houses(path):
        position = P(0, 0)
        visited_houses = {position}
        for direction in path:
            position = move(position, direction)
            visited_houses.add(position)
        return visited_houses

    path = [D(a) for a in data]
    out_a = len(get_visited_houses(path))
    houses_santa = get_visited_houses(path[::2])
    houses_robo_santa = get_visited_houses(path[1::2])
    out_b = len(houses_santa.union(houses_robo_santa))
    return out_a, out_b


def p_4(data):
    def check(key, no_zeros):
        hex_hash = hashlib.md5(key.encode()).hexdigest()
        return hex_hash.startswith('0' * no_zeros)

    out_a = 0
    while not check(f'{data}{out_a}', 5):
        out_a += 1
    out_b = 0
    while not check(f'{data}{out_b}', 6):
        out_b += 1
    return out_a, out_b


def p_5(data):
    def is_nice_a(str_):
        three_vowels = len(re.findall('[aeiou]', str_)) >= 3
        twice_in_row = re.search('(.)\\1', str_)
        forbidden = re.search('ab|cd|pq|xy', str_)
        return three_vowels and twice_in_row and not forbidden

    def is_nice_b(str_):
        pair = re.search('(..).*\\1', str_)
        repeats = re.search('(.).\\1', str_)
        return pair and repeats

    lines = data.split()
    out_a, out_b = 0, 0
    for a_str in lines:
        if is_nice_a(a_str):
            out_a += 1
        if is_nice_b(a_str):
            out_b += 1
    return out_a, out_b


def p_6(data):
    Op = Enum('Op', {'on': 1, 'off': -1, 'toggle': 2})

    def get_op(line):
        for op in Op:
            if op.name in line:
                return op

    def get_p(a_match):
        return P(int(a_match.group(1)), int(a_match.group(2)))

    def turn(grid, nw, se, on, is_a):
        for x in range(nw.x, se.x + 1):
            for y in range(nw.y, se.y + 1):
                if is_a:
                    grid[x][y] = on
                else:
                    grid[x][y] += on.value
                    if grid[x][y] < 0:
                        grid[x][y] = 0

    def toggle(grid, nw, se):
        for x in range(nw.x, se.x + 1):
            for y in range(nw.y, se.y + 1):
                grid[x][y] = not grid[x][y]

    lines = data.split('\n')
    grid_a = [[0 for _ in range(1000)] for _ in range(1000)]
    grid_b = [[0 for _ in range(1000)] for _ in range(1000)]
    for line in lines:
        op = get_op(line)
        nw_match = re.search('.*?(\d+),(\d+)', line)
        se_match = re.search('(\d+),(\d+)$', line.strip())
        nw, se = get_p(nw_match), get_p(se_match)
        if op == Op.toggle:
            toggle(grid_a, nw, se)
        else:
            turn(grid_a, nw, se, op == Op.on, is_a=True)
        turn(grid_b, nw, se, op, is_a=False)
    out_a = sum(sum(a) for a in grid_a)
    out_b = sum(sum(a) for a in grid_b)
    return out_a, out_b


def p_7_a(data):
    Op_ = Enum('Op_', {'AND': (lambda o: o[0] & o[1],),
                       'OR': (lambda o: o[0] | o[1],),
                       'LSHIFT': (lambda o: o[0] << o[1],),
                       'RSHIFT': (lambda o: o[0] >> o[1],),
                       'NOT': (lambda o: bit_not(o[0], 16),)})
    Value = type('Value', (), {'signal': None})
    Wire = type('Wire', (), {'input': None, 'signal': None})
    Gate = type('Gate', (), {'operation': None, 'operators': None,
                             'signal': None})

    def parse():
        for line in data:
            l, r = line.split(' -> ')
            wire = wires[r]
            op = get_op(l)
            if not op:
                l_obj = get_wire_or_val(l)
                wire.input = l_obj
                continue
            gate = Gate()
            gate.operation = op.value[0]
            tok = [a for a in l.split() if a not in list(a.name for a in Op_)]
            gate.operators = [get_wire_or_val(a) for a in tok]
            wire.input = gate
            gates.append(gate)

    def get_op(line):
        for op in Op_:
            if op.name in line:
                return op

    def get_wire_or_val(arg):
        if arg.isnumeric():
            value = Value()
            value.signal = int(arg)
            return value
        return wires[arg]

    def fire_gates():
        for gate in gates:
            if gate.signal:
                continue
            operators = [a.signal for a in gate.operators]
            if all(a for a in operators if a != 0):
                gate.signal = gate.operation(operators)

    def fire_wires():
        for wire in wires.values():
            if wire.input.signal is None:
                continue
            wire.signal = wire.input.signal

    gates, wires = [], defaultdict(Wire)
    parse()
    while wires['a'].signal is None:
        fire_wires()
        fire_gates()
    return wires['a'].signal


def p_7_b(data):
    Op_ = Enum('Op_', {'AND': (lambda i, p: i[0] & i[1],),
                       'OR': (lambda i, p: i[0] | i[1],),
                       'LSHIFT': (lambda i, p: i[0] << p[0],),
                       'RSHIFT': (lambda i, p: i[0] >> p[0],),
                       'NOT': (lambda i, p: bit_not(i[0], 16),)})

    Value = type('Value', (), {'signal': None})
    Wire = type('Wire', (), {'input': None, 'signal': None})

    class Gate:
        def __init__(self):
            self.wires_ = []
            self.operation_ = None
            self.params = []
            self.signal = None

    gates, wires = [], defaultdict(Wire)

    def main():
        parse()
        wires['b'].signal = 956
        while not wires['a'].signal:
            fire_wires(wires)
            fire_gates(gates)
        return wires['a'].signal

    def parse():
        for line in data:
            l, r = line.split(' -> ')
            wire = wires[r]
            op = get_op(l)
            if not op:
                l_obj = get_wire_or_val(l)
                wire.input = l_obj
                continue
            gate = Gate()
            gate.operation_ = op.value[0]
            tok = l.split()
            if op in (Op_.AND, Op_.OR):
                gate.wires_.append(get_wire_or_val(tok[0]))
                gate.wires_.append(get_wire_or_val(tok[2]))
            elif op in (Op_.LSHIFT, Op_.RSHIFT):
                gate.wires_.append(get_wire_or_val(tok[0]))
                gate.params.append(get_wire_or_val(tok[2]))
            else:
                gate.wires_.append(get_wire_or_val(tok[1]))
            wire.input = gate
            gates.append(gate)

    def get_op(line):
        for op in Op_:
            if op.name in line:
                return op

    def get_wire_or_val(arg):
        if arg.isnumeric():
            value = Value()
            value.signal = int(arg)
            return value
        return wires[arg]

    def fire_gates(gates):
        for gate in gates:
            if gate.signal:
                continue
            signals = [a.signal for a in gate.wires_]
            params = [a.signal for a in gate.params]
            if all(a for a in signals + params if a != 0):
                gate.signal = gate.operation_(signals, params)

    def fire_wires(wires):
        for wire in wires.values():
            if wire.signal is not None or wire.input.signal is None:
                continue
            wire.signal = wire.input.signal

    return main()


def p_8_a(data):
    code_ch = 0
    memory_ch = 0
    for line in data:
        memory_ch += len(line)
        code = line[1:-1]
        code = re.sub(r'\\\\', 'B', code)
        code = re.sub(r'\\"', 'A', code)
        code = re.sub(r'\\x..', 'X', code)
        # print(line, code, len(line), len(code))
        code_ch += len(code)
    return memory_ch - code_ch


def p_8_b(data):
    code_ch = 0
    memory_ch = 0
    for line in data:
        memory_ch += len(line)
        code = line[1:-1]
        print(line, ':')
        code = re.sub(r'\\\\', r'\\', code)
        code = re.sub(r'\\"', r'"', code)
        # code = re.sub(r'\\x(..)', lambda a: bytearray.fromhex(a.group(1)).decode(), code)
        code = re.sub(r'\\x(..)',
                      lambda a: bytes.fromhex(a.group(1)).decode('latin-1'),
                      code)
        code = r'"\"' + code + r'\""'
        print(code, len(line), len(code))
        code_ch += len(code)
    return memory_ch - code_ch


def p_9_a(data):
    def parse():
        for line in data:
            l, r = line.split(' = ')
            a, b = l.split(' to ')
            di_a = dist.setdefault(a, defaultdict(int))
            di_b = dist.setdefault(b, defaultdict(int))
            di_a[b] = int(r)
            di_b[a] = int(r)

    def get_dist(route):
        out = 0
        last_city = route[0]
        for city in route[1:]:
            out += dist[last_city][city]
            last_city = city
        return out

    dist = {}
    parse()
    min_ = inf
    cities = dist.keys()
    for route in permutations(cities):
        route_dist = get_dist(route)
        if route_dist < min_:
            min_ = route_dist
    return min_


def p_9_b(data):
    def parse():
        for line in data:
            l, r = line.split(' = ')
            a, b = l.split(' to ')
            di_a = dist.setdefault(a, defaultdict(int))
            di_b = dist.setdefault(b, defaultdict(int))
            di_a[b] = int(r)
            di_b[a] = int(r)

    def get_dist(route):
        out = 0
        last_city = route[0]
        for city in route[1:]:
            out += dist[last_city][city]
            last_city = city
        return out

    dist = {}
    parse()
    max_ = 0
    cities = dist.keys()
    for route in permutations(cities):
        route_dist = get_dist(route)
        if route_dist > max_:
            max_ = route_dist
    return max_


def p_10(data):
    def one_pass(seq):
        out = []
        last, repeats = None, 1
        for digit in seq:
            if digit == last:
                repeats += 1
            else:
                if last:
                    out.append(f'{repeats}{last}')
                last, repeats = digit, 1
        out.append(f'{repeats}{last}')
        return ''.join(out)

    out_a = data
    for _ in range(50):
        out_a = one_pass(out_a)
    return len(out_a)


def p_12(data):
    def run(a_str):
        numbers = re.findall('[-]*\d+', a_str)
        return sum(int(a) for a in numbers)

    def process(dt):
        if type(dt) != dict and type(dt) != list:
            return dt
        dt = dt.values() if type(dt) == dict else dt
        return [process(a) for a in dt if
                not (type(a) == dict and 'red' in a.values())]

    dt = json.loads(data)
    dt = process(dt)
    return run(data), run(json.dumps(dt))


def p_14(data):
    Deer = type('Deer', (), {'speed': 0, 'fly': 0, 'rest': 0, 'resting': False,
                             'sec': 0, 'distance': 0, 'points': 0})
    deers = []
    lines = data.split('\n')
    for line in lines:
        tok = line.split()
        speed = int(tok[3])
        fly = int(tok[6])
        rest = int(tok[13])
        deer = Deer()
        deer.speed, deer.fly, deer.rest = speed, fly, rest
        deers.append(deer)
    for _ in range(2503):
        for deer in deers:
            if deer.resting and deer.sec == deer.rest:
                deer.resting = False
                deer.sec = 0
            elif not deer.resting and deer.sec == deer.fly:
                deer.resting = True
                deer.sec = 0
            if not deer.resting:
                deer.distance += deer.speed
            deer.sec += 1
        mx = max(a.distance for a in deers)
        for deer in deers:
            if deer.distance == mx:
                deer.points += 1
    return max(a.distance for a in deers), max(a.points for a in deers)


def p_15(data):
    Prop = namedtuple('Prop', 'capacity durability flavor texture calories')

    def score(props, cents):
        out = [0, 0, 0, 0]
        for p, cent in zip(props, cents):
            out[0] += p.capacity * cent
            out[1] += p.durability * cent
            out[2] += p.flavor * cent
            out[3] += p.texture * cent
        out = [max(0, a) for a in out]
        return out[0] * out[1] * out[2] * out[3]

    def get_calories(props, cents):
        out = 0
        for p, cent in zip(props, cents):
            out += cent * p.calories
        return out

    def subdivide(n, mx):
        if n == 1:
            return [mx]
        out = []
        for a in zip(range(1, mx - n + 2), range(mx - 1, -1, -1)):
            curr = a[0]
            nexts = subdivide(n - 1, a[1])
            for nx in nexts:
                if type(nx) == int:
                    nx = [nx]
                out.append([curr] + nx)
        return out

    lines = data.split('\n')
    props = []
    for line in lines:
        t = line.split()
        t = [re.sub(',$', '', a) for a in t]
        prop = Prop(int(t[2]), int(t[4]), int(t[6]), int(t[8]), int(t[10]))
        props.append(prop)
    mx_a, mx_b = 0, 0
    for cents in subdivide(4, 100):
        sc = score(props, cents)
        if sc > mx_a:
            mx_a = sc
        if get_calories(props, cents) != 500:
            continue
        if sc > mx_b:
            mx_b = sc
    return mx_a, mx_b


def p_16(data):
    knowns = {
        'children': 3,
        'cats': 7,
        'samoyeds': 2,
        'pomeranians': 3,
        'akitas': 0,
        'vizslas': 0,
        'goldfish': 5,
        'trees': 3,
        'cars': 2,
        'perfumes': 1}

    def get_sue(line):
        line = re.sub('^.*?\d: ', '', line)
        to = line.split(', ')
        out = {}
        for t in to:
            k, v = t.split(': ')
            out[k] = int(v)
        return out

    def is_match(sue):
        for k in sue.keys():
            if knowns[k] != sue[k]:
                return False
        return True

    sues = []
    for line in data:
        sues.append(get_sue(line))
    for i, sue in enumerate(sues):
        if is_match(sue):
            return i + 1


def p_17(data):
    cist = [int(a) for a in data]
    out_a, out_b = 0, 0
    for i in range(len(cist)):
        for comb in combinations(cist, i):
            if sum(comb) == 150:
                out_a += 1
        if out_a != 0 and out_b == 0:
            out_b = out_a
    return out_a, out_b


def p_19_a(data):
    dict_data = data[:-2]
    mol = data[-1]
    a_dict = defaultdict(list)
    for line in dict_data:
        k, v = line.split(' => ')
        a_dict[k].append(v)
    out = set()
    for k, vvv in a_dict.items():
        for v in vvv:
            mtchs = re.finditer(k, mol)
            for mtch in mtchs:
                start = mtch.start()
                end = mtch.end()
                new_line = mol[:start] + v + mol[end:]
                out.add(new_line)
    return len(out)


def p_19_b(data):
    def parse_data(data):
        dict_data = data[:-2]
        mol = data[-1]
        a_dict = defaultdict(list)
        for line in dict_data:
            k, v = line.split(' => ')
            a_dict[k].append(v)
        return mol, a_dict

    def get_new_mols(k, v, mol):
        mtch = re.search(v, mol)
        start = mtch.start()
        end = mtch.end()
        new_line = mol[:start] + k + mol[end:]
        return [Sol(new_line)]

    def deconstruct(sol, depth):
        out = []
        if sol.mol == 'e':
            return depth
        max_len = 0
        for k, vvv in a_dict.items():
            for v in vvv:
                if v not in sol.mol:
                    continue
                if len(v) <= max_len:
                    continue
                max_len = len(v)
                out = get_new_mols(k, v, sol.mol)
        return out

    mol, a_dict = parse_data(data)
    Sol = namedtuple('Sol', 'mol')
    solutions = [Sol(mol)]
    new_sol = []
    for depth in range(1000):
        for sol in solutions:
            n_sol = deconstruct(sol, depth)
            if type(n_sol) == int:
                return n_sol
            new_sol.extend(n_sol)
        solutions = new_sol
        new_sol = []


def p_20():
    def walk_elf(n, a_max):
        for i in range(0, a_max, n):
            houses[i] += n * 10

    inp = 36000000
    a_max = 2000000
    houses = defaultdict(int)
    for i in range(1, a_max):
        walk_elf(i, a_max)
        if houses[i] > inp:
            return i
    print(houses[a_max - 1])


def p_20_b():
    def walk_elf(n, a_max):
        for i in list(range(0, a_max, n))[:50]:
            houses[i] += n * 11

    inp = 36000000
    a_max = 2000000
    houses = defaultdict(int)
    for i in range(1, a_max):
        walk_elf(i, a_max)
        if houses[i] > inp:
            return i
    print(houses[a_max - 1])


def p_21_a():
    S = namedtuple('S', 'weapon armor ring_1 ring_2')
    T = namedtuple('T', 'cost damage armor')

    def play_out(s):
        b_hit_points = 103
        b_damage = 9
        b_armor = 2
        y_hit_points = 100
        y_damage = sum(a[1] for a in s if a)
        y_armor = sum(a[2] for a in s if a)
        player_turn = True
        while b_hit_points > 0 and y_hit_points > 0:
            if player_turn:
                damage = max(1, y_damage - b_armor)
                b_hit_points -= damage
                player_turn = False
            else:
                damage = max(1, b_damage - y_armor)
                y_hit_points -= damage
                player_turn = True
        return b_hit_points <= 0

    weapons = [T(8, 4, 0), T(10, 5, 0), T(25, 6, 0), T(40, 7, 0), T(74, 8, 0)]
    armor = [T(13, 0, 1), T(31, 0, 2), T(53, 0, 3), T(75, 0, 4), T(102, 0, 5),
             None]
    rings = [T(25, 1, 0), T(50, 2, 0), T(100, 3, 0), T(20, 0, 1), T(40, 0, 2),
             T(80, 0, 3), None, None]
    min_ = 100000
    for sol in product(weapons, armor, combinations(rings, 2)):
        sol = S(sol[0], sol[1], sol[2][0], sol[2][1])
        price = sum(a[0] for a in sol if a)
        if price > min_:
            continue
        if play_out(sol):
            min_ = price

    return min_


def p_21_b():
    S = namedtuple('S', 'weapon armor ring_1 ring_2')
    T = namedtuple('T', 'cost damage armor')

    def play_out(s):
        b_hit_points = 103
        b_damage = 9
        b_armor = 2
        y_hit_points = 100
        y_damage = sum(a[1] for a in s if a)
        y_armor = sum(a[2] for a in s if a)
        player_turn = True
        while b_hit_points > 0 and y_hit_points > 0:
            if player_turn:
                damage = max(1, y_damage - b_armor)
                b_hit_points -= damage
                player_turn = False
            else:
                damage = max(1, b_damage - y_armor)
                y_hit_points -= damage
                player_turn = True
        return y_hit_points <= 0

    weapons = [T(8, 4, 0), T(10, 5, 0), T(25, 6, 0), T(40, 7, 0), T(74, 8, 0)]
    armor = [T(13, 0, 1), T(31, 0, 2), T(53, 0, 3), T(75, 0, 4), T(102, 0, 5),
             None]
    rings = [T(25, 1, 0), T(50, 2, 0), T(100, 3, 0), T(20, 0, 1), T(40, 0, 2),
             T(80, 0, 3), None, None]
    a_max = 0
    max_sol = None
    for sol in product(weapons, armor, combinations(rings, 2)):
        sol = S(sol[0], sol[1], sol[2][0], sol[2][1])
        price = sum(a[0] for a in sol if a)
        if price < a_max:
            continue
        if play_out(sol):
            a_max = price
            max_sol = sol
    print(max_sol)
    return a_max


def p_23_a(data):
    r = {'a': 0, 'b': 0}
    i = 0
    while 0 <= i < len(data):
        comm = data[i]
        tokens = [a for a in re.split('[ ,]', comm) if a]
        inst = tokens[0]
        p1 = tokens[1]
        p2 = None if len(tokens) < 3 else tokens[2]
        if inst == 'hlf':
            r[p1] //= 2
            i += 1
        elif inst == 'tpl':
            r[p1] *= 3
            i += 1
        elif inst == 'inc':
            r[p1] += 1
            i += 1
        elif inst == 'jmp':
            i += int(p1)
        elif inst == 'jie':
            if r[p1] % 2 == 0:
                i += int(p2)
            else:
                i += 1
        elif inst == 'jio':
            if r[p1] == 1:
                i += int(p2)
            else:
                i += 1
    return r['b']


def p_23_b(data):
    r = {'a': 1, 'b': 0}
    i = 0
    while 0 <= i < len(data):
        comm = data[i]
        tokens = [a for a in re.split('[ ,]', comm) if a]
        inst = tokens[0]
        p1 = tokens[1]
        p2 = None if len(tokens) < 3 else tokens[2]
        if inst == 'hlf':
            r[p1] //= 2
            i += 1
        elif inst == 'tpl':
            r[p1] *= 3
            i += 1
        elif inst == 'inc':
            r[p1] += 1
            i += 1
        elif inst == 'jmp':
            i += int(p1)
        elif inst == 'jie':
            if r[p1] % 2 == 0:
                i += int(p2)
            else:
                i += 1
        elif inst == 'jio':
            if r[p1] == 1:
                i += int(p2)
            else:
                i += 1
    return r['b']


def p_24_a(data):
    def combinate(packets, n, weight):
        l = chain(*(combinations(packets, i) for i in range(1, n)))
        l = (a for a in l if sum(a) == weight)
        r = (tuple(set(packets).difference(a)) for a in l)
        return zip(l, r)

    def get_all_combs(packets, weight):
        l = combinate(packets, len(packets) - 1, weight)
        min_ = 10000
        for a in l:
            if len(a[0]) > min_:
                continue
            min_ = len(a[0])
            yield a[0], a[1]

    data = [int(a) for a in data]
    weight = sum(data) / 4
    out = get_all_combs(data, weight)
    i = 0
    min_ = float('inf')
    from functools import reduce
    for a in out:
        prod = reduce(lambda out, x: out * x, a[0])
        if prod >= min_:
            continue
        min_ = prod
        i += 1
        return min_


def p_25_a():
    y, x = 3010 - 1, 3019 - 1
    value = 20151125
    for i in count(1):
        for j, k in zip(range(i, -1, -1), range(i + 1)):
            value = (value * 252533) % 33554393
            if j == y and k == x:
                return value


FUN = p_9_b
print(run(FUN))
