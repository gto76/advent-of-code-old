#!/usr/bin/env python3
#
# Usage: ./advent_2020.py
# Descriptions of problems can be found here: https://adventofcode.com/2020
# Script runs a test for every function with test data that is stored in 'IN_<problem_num>'
# variable. The expected result should be stored in function's docstring.


def main():
    for function in [a for a in globals().values() if callable(a) and a.__name__ != 'main']:
        input_name = 'IN_' + function.__name__.split('_')[1]
        lines = globals()[input_name].splitlines()
        result = function(lines)
        expected_result = function.__doc__.split('?')[1].strip()
        if str(result) != expected_result:
            print(f'Function "{function.__name__}" returned {result} instead of',
                  f'{expected_result}.')
            break
    else:
        print('All tests passed.')


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
    '''Find the two entries that sum to 2020; what do you get if you multiply them together?
    514579'''
    import itertools
    numbers = [int(line) for line in lines]
    for l, r in itertools.combinations(numbers, 2):
        if l + r == 2020:
            return l * r


def problem_1_b(lines):
    '''In your expense report, what is the product of the three entries that sum to 2020?
    241861950'''
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
    '''How many passwords are valid according to their policies? 2'''
    import re
    def is_valid(line):
        min_, max_, letter, password = re.match('^(\d+)-(\d+) (\w): (\w+)$', line).groups()
        return int(min_) <= password.count(letter) <= int(max_)
    return sum(is_valid(line) for line in lines)


def problem_2_b(lines):
    '''How many passwords are valid according to the new interpretation of the policies? 1'''
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
    '''Starting at the top-left corner of your map and following a slope of right 3 and down 1,
    how many trees would you encounter? 7'''
    import collections
    P = collections.namedtuple('P', 'x y')
    positions = (P(x=y*3, y=y) for y in range(len(lines)))
    is_tree = lambda p: lines[p.y][p.x % len(lines[0])] == '#'
    return sum(is_tree(p) for p in positions)


def problem_3_b(lines):
    '''What do you get if you multiply together the number of trees encountered on each of the
    listed slopes? 336'''
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
##  DAY 4
#

IN_4 = \
'''ecl:gry pid:860033327 eyr:2020 hcl:#fffffd
byr:1937 iyr:2017 cid:147 hgt:183cm

iyr:2013 ecl:amb cid:350 eyr:2023 pid:028048884
hcl:#cfa07d byr:1929

hcl:#ae17e1 iyr:2013
eyr:2024
ecl:brn pid:760753108 byr:1931
hgt:179cm

hcl:#cfa07d eyr:2025 pid:166559648
iyr:2011 ecl:brn hgt:59in'''


def problem_4_a(lines):
    '''In your batch file, how many passports are valid? 2'''
    passports = ' '.join(lines).split('  ')
    get_keys = lambda passport: {item.split(':')[0] for item in passport.split()}
    is_valid = lambda passport: len(get_keys(passport) - {'cid'}) == 7
    return sum(is_valid(p) for p in passports)


def problem_4_b(lines):
    '''In your batch file, how many passports are valid? 2'''
    import re
    RULES = dict(
        byr=lambda v: 1920 <= int(v) <= 2002,
        iyr=lambda v: 2010 <= int(v) <=  2020,
        eyr=lambda v: 2020 <= int(v) <=  2030,
        hgt=lambda v: 150 <= int(v[:-2]) <= 193 if 'cm' in v else 59 <= int(v[:-2]) <= 76,
        hcl=lambda v: re.match('#[0-9a-f]{6}$', v) != None,
        ecl=lambda v: v in 'amb blu brn gry grn hzl oth'.split(),
        pid=lambda v: re.match('\d{9}$', v) != None
    )
    def is_field_valid(key, value):
        try:
            return RULES[key](value)
        except Exception:
            return False
    def is_passport_valid(passport):
        return sum(is_field_valid(*item.split(':')) for item in passport.split()) == 7
    passports = ' '.join(lines).split('  ')
    return sum(is_passport_valid(p) for p in passports)


###
##  DAY 5
#

IN_5 = \
'''BBFFBBFLLR
BBFFBBFLRL
BBFFBBFRLL'''


def problem_5_a(lines):
    '''What is the highest seat ID on a boarding pass? 820'''
    get_row    = lambda code: int(code[:-3].replace('F', '0').replace('B', '1'), 2)
    get_column = lambda code: int(code[-3:].replace('L', '0').replace('R', '1'), 2)
    get_id     = lambda code: get_row(code) * 8 + get_column(code)
    return max(get_id(code) for code in lines)


def problem_5_b(lines):
    '''What is the ID of your seat? 819'''
    get_row    = lambda code: int(code[:-3].replace('F', '0').replace('B', '1'), 2)
    get_column = lambda code: int(code[-3:].replace('L', '0').replace('R', '1'), 2)
    get_id     = lambda code: get_row(code) * 8 + get_column(code)
    taken_ids  = {get_id(code) for code in lines}
    all_ids    = range(min(taken_ids), max(taken_ids)+1)
    return (set(all_ids) - taken_ids).pop()


# ###
# ##  DAY X
# #
#
# IN_X = \
# ''''''
#
#
# def problem_X_a(lines):
#     ''''''
#     pass
#
#
# def problem_X_b(lines):
#     ''''''
#     pass


###
##  MAIN
#  

if __name__ == '__main__':
    main()
