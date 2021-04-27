from typing import List

from Grid import Grid
from random import randint, choices
from math import sqrt, log2

directionVectors = (UP_VEC, DOWN_VEC, LEFT_VEC, RIGHT_VEC) = ((-1, 0), (1, 0), (0, -1), (0, 1))


def getANewGrid(grid: Grid, dir) -> Grid:
    '''
    :param grid: A grid
    :param dir: A move
    :return: a random result of move
    '''
    if grid.isTerminal():
        # return the original grid if the game is terminal
        return grid.clone()
    ret = grid.clone()
    ret.move(dir)
    cells = ret.getAvailableCells()
    cell = cells[randint(0, len(cells) - 1)]
    ret.setCellValue(cell, 2)
    return ret


def getAllPossibleGrid(grid: Grid, dir) -> List[Grid]:
    '''
    :param grid: A grid
    :param dir: A move
    :return: a list of all possible result of move
    '''
    if grid.isTerminal():
        # return the original grid if the game is terminal
        return [grid.clone()]
    ret = grid.clone()
    ret.move(dir)
    if not grid.canMove([dir]):
        raise ValueError('Invalid Move')
    cells = ret.getAvailableCells()
    grids = []
    for cell in cells:
        ret2 = ret.clone()
        ret2.setCellValue(cell, 2)
        grids.append(ret2)
    return grids


def getKNewGrid(grid: Grid, dir, k=4):
    '''
    random sample k resulting grids with replacement
    '''

    return [getANewGrid(grid, dir) for _ in range(k)]


def freeCellsHeuristic(grid: Grid):
    '''
    :param grid: estimated Grid
    :return: free cells(empty cell) Count
    :rtype: int
    '''
    return len(grid.getAvailableCells())


def smoothnessHeuristic(grid: Grid):
    size = grid.size
    value = 0
    for i in range(size):
        for j in range(size):
            cellValue = float('inf')
            if i > 0: cellValue = min(cellValue, abs(grid.mat[i][j] - grid.mat[i - 1][j]))
            if i < size - 1: cellValue = min(cellValue, abs(grid.mat[i][j] - grid.mat[i + 1][j]))
            if j > 0: cellValue = min(cellValue, abs(grid.mat[i][j] - grid.mat[i][j - 1]))
            if j < size - 1: cellValue = min(cellValue, abs(grid.mat[i][j] - grid.mat[i][j + 1]))
            if 4096 > cellValue > 0:
                value += log2(cellValue)
    return -(value)


def monotonicityHeuristic(grid:Grid):
    '''
    calculate the monotonicity of the grid, we prefer to the rows and columns increasing / decreasing monotonically
    '''

    size = grid.size
    monotonicity_right, monotonicity_left = 0, 0
    monotonicity_up, monotonicity_down = 0, 0

    # row check
    for j in range(size):
        previous, current = None, None
        for i in range(size):
            if grid.mat[i][j] > 0:
                previous = current
                current = grid.mat[i][j]
                if previous and current:
                    if current > previous:
                        monotonicity_down += log2(previous) - log2(current)
                    elif current < previous:
                        monotonicity_up += log2(current) - log2(previous)

    for i in range(size):
        previous, current = None, None
        for j in range(size):
            if grid.mat[i][j] > 0:
                previous = current
                current = grid.mat[i][j]
                if previous and current:
                    if current > previous:
                        monotonicity_right += log2(previous) - log2(current)
                    elif current < previous:
                        monotonicity_left += log2(current) - log2(previous)

    return max(monotonicity_left, monotonicity_right) + max(monotonicity_down, monotonicity_up)



def maxValueHeuristic(grid: Grid):
    return log2(grid.getMaxTile())


def edgeHeuristic(grid: Grid):
    '''
    We like big tiles on the edge
    '''
    size = grid.size
    value = 0
    for i in range(size):
        for j in range(size):
            if grid.mat[i][j] == 0:
                continue
            elif (i == 0 or i == size) and (j == 0 or j == size - 1):
                cellValue = 2 * log2(grid.mat[i][j])
            elif i == 0 or i == size - 1 or j == 0 or j == size - 1:
                cellValue = log2(grid.mat[i][j])
            else:
                cellValue = -log2(grid.mat[i][j])
            value += cellValue
    return value

def getGridKey(grid: Grid):
    gridKey = ' '.join([' '.join(str(c) for c in lst) for lst in grid.mat])
    return gridKey


class Counter(dict):
    """
    A counter keeps track of counts for a set of keys.

    The counter class is an extension of the standard python
    dictionary type.  It is specialized to have number values
    (integers or floats), and includes a handful of additional
    functions to ease the task of counting data.  In particular,
    all keys are defaulted to have value 0.  Using a dictionary:

    a = {}
    print a['test']

    would give an error, while the Counter class analogue:

    >>> a = Counter()
    >>> print a['test']
    0

    returns the default 0 value. Note that to reference a key
    that you know is contained in the counter,
    you can still use the dictionary syntax:

    >>> a = Counter()
    >>> a['test'] = 2
    >>> print a['test']
    2

    This is very useful for counting things without initializing their counts,
    see for example:

    >>> a['blah'] += 1
    >>> print a['blah']
    1

    The counter also includes additional functionality useful in implementing
    the classifiers for this assignment.  Two counters can be added,
    subtracted or multiplied together.  See below for details.  They can
    also be normalized and their total count and arg max can be extracted.
    """

    def __getitem__(self, idx):
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)

    def incrementAll(self, keys, count):
        """
        Increments all elements of keys by the same count.

        >>> a = Counter()
        >>> a.incrementAll(['one','two', 'three'], 1)
        >>> a['one']
        1
        >>> a['two']
        1
        """
        for key in keys:
            self[key] += count

    def argMax(self):
        """
        Returns the key with the highest value.
        """
        if len(list(self.keys())) == 0:
            return None
        all = list(self.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def sortedKeys(self):
        """
        Returns a list of keys sorted by their values.  Keys
        with the highest values will appear first.

        >>> a = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> a['third'] = 1
        >>> a.sortedKeys()
        ['second', 'third', 'first']
        """
        sortedItems = list(self.items())

        def compare(x, y): return sign(y[1] - x[1])
        sortedItems.sort(key=functools.cmp_to_key(compare))
        return [x[0] for x in sortedItems]

    def totalCount(self):
        """
        Returns the sum of counts for all keys.
        """
        return sum(self.values())

    def normalize(self):
        """
        Edits the counter such that the total count of all
        keys sums to 1.  The ratio of counts for all keys
        will remain the same. Note that normalizing an empty
        Counter will result in an error.
        """
        total = float(self.totalCount())
        if total == 0:
            return
        for key in list(self.keys()):
            self[key] = self[key] / total

    def divideAll(self, divisor):
        """
        Divides all counts by divisor
        """
        divisor = float(divisor)
        for key in self:
            self[key] /= divisor

    def copy(self):
        """
        Returns a copy of the counter
        """
        return Counter(dict.copy(self))

    def __mul__(self, y):
        """
        Multiplying two counters gives the dot product of their vectors where
        each unique label is a vector element.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['second'] = 5
        >>> a['third'] = 1.5
        >>> a['fourth'] = 2.5
        >>> a * b
        14
        """
        sum = 0
        x = self
        if len(x) > len(y):
            x, y = y, x
        for key in x:
            if key not in y:
                continue
            sum += x[key] * y[key]
        return sum

    def __radd__(self, y):
        """
        Adding another counter to a counter increments the current counter
        by the values stored in the second counter.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> a += b
        >>> a['first']
        1
        """
        for key, value in list(y.items()):
            self[key] += value

    def __add__(self, y):
        """
        Adding two counters gives a counter with the union of all keys and
        counts of the second added to counts of the first.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> (a + b)['first']
        1
        """
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] + y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = y[key]
        return addend

    def __sub__(self, y):
        """
        Subtracting a counter from another gives a counter with the union of all keys and
        counts of the second subtracted from counts of the first.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> (a - b)['first']
        -5
        """
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] - y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = -1 * y[key]
        return addend

def sign(x):
    """
    Returns 1 or -1 depending on the sign of x
    """
    if(x >= 0):
        return 1
    else:
        return -1

if __name__ == '__main__':
    grid = Grid()
    grid.mat = [[1024,64,4,2],[128,512,16,8],[32,256,8,4],[4,128,4,2]]
    print(smoothnessHeuristic(grid))

