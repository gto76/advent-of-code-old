

###
##  DAY 1
#

def problem_1_a(lines):
    from itertools import combinations
    entries = [int(line) for line in lines]
    for l, r in combinations(entries, 2):
        if l + r == 2020:
            return l * r


def problem_1_b(lines):
    from itertools import combinations
    entries = [int(line) for line in lines]
    for a, b, c in combinations(entries, 3):
        if a + b + c == 2020:
            return a * b * c


def problem_1_a(lines):
    from itertools import combinations
    for l, r in combinations(map(int, lines), 2):
        if l + r == 2020:
            return l * r


def problem_1_b(lines):
    from itertools import combinations
    for a, b, c in combinations(map(int, lines), 3):
        if a + b + c == 2020:
            return a * b * c


def problem_1_a(lines):
    from itertools import combinations
    entries = (int(line) for line in lines)
    for l, r in combinations(entries, 2):
        if l + r == 2020:
            return l * r


def problem_1_b(lines):
    from itertools import combinations
    entries = (int(line) for line in lines)
    for a, b, c in combinations(entries, 3):
        if a + b + c == 2020:
            return a * b * c


def problem_1_a(lines):
    from itertools import combinations
    entries = map(int, lines)
    for l, r in combinations(entries, 2):
        if l + r == 2020:
            return l * r


def problem_1_b(lines):
    from itertools import combinations
    entries = map(int, lines)
    for a, b, c in combinations(entries, 3):
        if a + b + c == 2020:
            return a * b * c


def problem_1_a(lines):
    from itertools import combinations
    entries = map(int, lines)
    return [l*r for l, r in combinations(entries, 2) if l+r == 2020][0]


def problem_1_b(lines):
    from itertools import combinations
    entries = map(int, lines)
    return [a*b*c for a, b, c in combinations(entries, 3) if a+b+c == 2020][0]


###
##  DAY 2
#

def problem_2_a(lines):
    out = 0
    for line in lines:
        min_, max_, letter, password = re.match('^(\d+)-(\d+) (\w): (\w+)$', line).groups()
        out += int(min_) <= password.count(letter) <= int(max_)
    return out


def problem_2_a(lines):
    def process_():
        for line in lines:
            min_, max_, letter, password = re.match('^(\d+)-(\d+) (\w): (\w+)$', line).groups()
            yield int(min_) <= password.count(letter) <= int(max_)
    return sum(process_())


def problem_2_a(lines):
    def parser():
        for line in lines:
            yield re.match('^(\d+)-(\d+) (\w): (\w+)$', line).groups()
    return sum(int(min_) <= password.count(letter) <= int(max_) for min_, max_, letter, password in parser())


def problem_2_a(lines):
    def parser():
        for line in lines:
            yield re.match('^(\d+)-(\d+) (\w): (\w+)$', line).groups()
    def is_valid(min_, max_, letter, password):
        return int(min_) <= password.count(letter) <= int(max_)
    return sum(is_valid(*tokens) for tokens in parser())


def problem_2_a(lines):
    def parse_line(line):
        return re.match('^(\d+)-(\d+) (\w): (\w+)$', line).groups()
    def is_valid(min_, max_, letter, password):
        return int(min_) <= password.count(letter) <= int(max_)
    return sum(is_valid(*parse_line(line)) for line in lines)


def problem_2_a(lines):
    def parse_line(line): return re.match('^(\d+)-(\d+) (\w): (\w+)$', line).groups()
    def is_valid(min_, max_, letter, password): return int(min_) <= password.count(letter) <= int(max_)
    return sum(is_valid(*parse_line(line)) for line in lines)


def problem_2_a(lines):
    parse_line = lambda line: re.match('^(\d+)-(\d+) (\w): (\w+)$', line).groups()
    is_valid   = lambda min_, max_, letter, password: int(min_) <= password.count(letter) <= int(max_)
    return sum(is_valid(*parse_line(line)) for line in lines)


def problem_2_b(lines):
    out = 0
    for line in lines:
        i_1, i_2, letter, password = re.match('^(\d+)-(\d+) (\w): (\w+)$', line).groups()
        out += (password[int(i_1)-1] == letter) + (password[int(i_2)-1] == letter) == 1
    return out


def problem_2_a(lines):
    def is_valid(line):
        min_, max_, letter, password = re.match('^(\d+)-(\d+) (\w): (\w+)$', line).groups()
        return int(min_) <= password.count(letter) <= int(max_)
    return sum(is_valid(line) for line in lines)


def problem_2_b(lines):
    def is_valid(line):
        i_1, i_2, letter, password = re.match('^(\d+)-(\d+) (\w): (\w+)$', line).groups()
        return (password[int(i_1)-1] == letter) + (password[int(i_2)-1] == letter) == 1
    return sum(is_valid(line) for line in lines)


def problem_2_a(lines):
    is_valid = lambda min_, max_, letter, password: int(min_) <= password.count(letter) <= int(max_)
    return sum(is_valid(*re.match('^(\d+)-(\d+) (\w): (\w+)$', line).groups()) for line in lines)


def problem_2_a(lines):
    def is_valid(min_, max_, letter, password): return int(min_) <= password.count(letter) <= int(max_)
    return sum(is_valid(*re.match('^(\d+)-(\d+) (\w): (\w+)$', line).groups()) for line in lines)


def problem_2_a(lines):
    def is_valid(min_, max_, letter, password):
        return int(min_) <= password.count(letter) <= int(max_)
    return sum(is_valid(*re.match('^(\d+)-(\d+) (\w): (\w+)$', line).groups()) for line in lines)


def problem_2_b(lines):
    def is_valid(i_1, i_2, letter, password):
        return (password[int(i_1)-1] == letter) + (password[int(i_2)-1] == letter) == 1
    return sum(is_valid(*re.match('^(\d+)-(\d+) (\w): (\w+)$', line).groups()) for line in lines)


###
##  DAY 3
#

def problem_3_a(lines):
    import collections
    P = collections.namedtuple('P', 'x y')
    positions = (P(x=y*3, y=y) for y in range(len(lines)))
    is_tree = lambda p: lines[p.y][p.x % len(lines[0])] == '#'
    return sum(is_tree(p) for p in positions)


def problem_3_b(lines):
    import collections, functools, itertools, operator
    P = collections.namedtuple('P', 'x y')
    def get_positions(slope):
        x_generator = itertools.count(start=0, step=slope.x)
        return (P(next(x_generator), y) for y in range(0, len(lines), slope.y))
    is_tree = lambda p: lines[p.y][p.x % len(lines[0])] == '#'
    count_trees = lambda slope: sum(is_tree(p) for p in get_positions(slope))
    slopes = [P(x=1, y=1), P(x=3, y=1), P(x=5, y=1), P(x=7, y=1), P(x=1, y=2)]
    return functools.reduce(operator.mul, (count_trees(slope) for slope in slopes))





















