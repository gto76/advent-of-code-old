#!/usr/bin/env python3
#
# Usage: ./advent_2020.py
# Descriptions of problems can be found here: https://adventofcode.com/2020
# Script runs a test for every function with test data that is stored in 'IN_<problem_num>'
# variable. The expected result is stored in function's docstring.


###
##  DAY 1
#

IN_1 = \
'''1721
979
366
299
675
1456'''


def problem_1_a(lines):
    '''514579'''
    import itertools
    numbers = [int(line) for line in lines]
    for l, r in itertools.combinations(numbers, 2):
        if l + r == 2020:
            return l * r


def problem_1_b(lines):
    '''241861950'''
    import itertools
    numbers = [int(line) for line in lines]
    for a, b, c in itertools.combinations(numbers, 3):
        if a + b + c == 2020:
            return a * b * c


###
##  DAY 2
#

IN_2 = \
'''1-3 a: abcde
1-3 b: cdefg
2-9 c: ccccccccc'''


def problem_2_a(lines):
    '''2'''
    import re
    def is_valid(line):
        min_, max_, letter, password = re.match('^(\d+)-(\d+) (\w): (\w+)$', line).groups()
        return int(min_) <= password.count(letter) <= int(max_)
    return sum(is_valid(line) for line in lines)


def problem_2_b(lines):
    '''1'''
    import re
    def is_valid(line):
        i_1, i_2, letter, password = re.match('^(\d+)-(\d+) (\w): (\w+)$', line).groups()
        return (password[int(i_1)-1] == letter) + (password[int(i_2)-1] == letter) == 1
    return sum(is_valid(line) for line in lines)


###
##  DAY 3
#

IN_3 = \
'''..##.......
#...#...#..
.#....#..#.
..#.#...#.#
.#...##..#.
..#.##.....
.#.#.#....#
.#........#
#.##...#...
#...##....#
.#..#...#.#'''


def problem_3_a(lines):
    '''7'''
    import collections
    P = collections.namedtuple('P', 'x y')
    positions = (P(x=y*3, y=y) for y in range(len(lines)))
    is_tree = lambda p: lines[p.y][p.x % len(lines[0])] == '#'
    return sum(is_tree(p) for p in positions)


def problem_3_b(lines):
    '''336'''
    import collections, functools, itertools, operator
    P = collections.namedtuple('P', 'x y')
    def get_positions(slope):
        x_generator = itertools.count(start=0, step=slope.x)
        return (P(next(x_generator), y) for y in range(0, len(lines), slope.y))
    is_tree = lambda p: lines[p.y][p.x % len(lines[0])] == '#'
    count_trees = lambda slope: sum(is_tree(p) for p in get_positions(slope))
    slopes = [P(x=1, y=1), P(x=3, y=1), P(x=5, y=1), P(x=7, y=1), P(x=1, y=2)]
    return functools.reduce(operator.mul, (count_trees(slope) for slope in slopes))


###
##  TESTS
#

for function in [a for a in vars().values() if callable(a)]:
    input_name = 'IN_' + function.__name__.split('_')[1]
    lines = vars()[input_name].splitlines()
    assert str(function(lines)) == function.__doc__

print('All tests passed.')
