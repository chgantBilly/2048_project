from copy import deepcopy
from random import randint

directionVectors = (UP_VEC, DOWN_VEC, LEFT_VEC, RIGHT_VEC) = ((-1, 0), (1, 0), (0, -1), (0, 1))
vecIndex = [UP, DOWN, LEFT, RIGHT] = range(4)


class Grid:
    '''
    Modified from https://github.com/MeGaCrazy/2048-Puzzle-Solver
    All game rules for 2048
    '''

    def __init__(self, size=4):
        self.size = size
        self.mat = [[0] * self.size for i in range(self.size)]
        self.score = 0

    # Make a Deep Copy of This Object
    def clone(self):
        gridCopy = Grid()
        gridCopy.mat = deepcopy(self.mat)
        gridCopy.size = self.size
        return gridCopy

    # Insert a Tile in an Empty Cell
    def insertTile(self, pos, value):
        self.setCellValue(pos, value)

    def setCellValue(self, pos, value):
        self.mat[pos[0]][pos[1]] = value

    # Return All the Empty c\Cells
    def getAvailableCells(self):
        cells = []

        for x in range(self.size):
            for y in range(self.size):
                if self.mat[x][y] == 0:
                    cells.append((x, y))

        return cells

    # Return the Tile with Maximum Value
    def getMaxTile(self):
        maxTile = 0
        for x in range(self.size):
            for y in range(self.size):
                maxTile = max(maxTile, self.mat[x][y])
        return maxTile

    # Check If Able to Insert a Tile in Position
    def canInsert(self, pos):
        return self.getCellValue(pos) == 0

    # Move the Grid
    def move(self, dir):
        dir = int(dir)
        if dir == UP:
            return self.moveUD(False)
        if dir == DOWN:
            return self.moveUD(True)
        if dir == LEFT:
            return self.moveLR(False)
        if dir == RIGHT:
            return self.moveLR(True)
        else:
            raise ValueError('Invalid Move!')

    # Move Up or Down
    def moveUD(self, down):
        r = range(self.size - 1, -1, -1) if down else range(self.size)
        moved = False
        for j in range(self.size):
            cells = []
            for i in r:
                cell = self.mat[i][j]
                if cell != 0:
                    cells.append(cell)
            self.merge(cells)

            for i in r:
                value = cells.pop(0) if cells else 0
                if self.mat[i][j] != value:
                    moved = True
                self.mat[i][j] = value

        return moved

    # move left or right
    def moveLR(self, right):
        r = range(self.size - 1, -1, -1) if right else range(self.size)
        moved = False
        for i in range(self.size):
            cells = []
            for j in r:
                cell = self.mat[i][j]
                if cell != 0:
                    cells.append(cell)

            self.merge(cells)

            for j in r:
                value = cells.pop(0) if cells else 0
                if self.mat[i][j] != value:
                    moved = True
                self.mat[i][j] = value

        return moved

    # Merge Tiles
    def merge(self, cells):
        if len(cells) <= 1:
            return cells
        i = 0
        moveScore = 0
        while i < len(cells) - 1:
            if cells[i] == cells[i + 1]:
                cells[i] *= 2
                moveScore += cells[i]

                del cells[i + 1]
            i += 1

        self.score = moveScore
        # print(self.score)

    def canMove(self, dirs=vecIndex):
        # Init Moves to be Checked
        checkingMoves = set(dirs)
        for x in range(self.size):
            for y in range(self.size):
                # If Current Cell is Filled
                if self.mat[x][y]:
                    # Look Ajacent Cell Value
                    for i in checkingMoves:
                        move = directionVectors[i]
                        adjCellValue = self.getCellValue((x + move[0], y + move[1]))
                        # If Value is the Same or Adjacent Cell is Empty
                        if adjCellValue == self.mat[x][y] or adjCellValue == 0:
                            return True
                # Else if Current Cell is Empty
                elif self.mat[x][y] == 0:
                    return True
        return False

    # Return All Available Moves
    def getAvailableMoves(self, dirs=vecIndex):
        availableMoves = []
        for x in dirs:
            gridCopy = self.clone()
            if gridCopy.move(x):
                availableMoves.append(x)
        return availableMoves

    def crossBound(self, pos):
        return pos[0] < 0 or pos[0] >= self.size or pos[1] < 0 or pos[1] >= self.size

    def getCellValue(self, pos):
        if not self.crossBound(pos):
            return self.mat[pos[0]][pos[1]]
        else:
            return None

    def isWin(self):
        return self.getMaxTile() == 2048

    def isLose(self):
        return not self.canMove()

    def isTerminal(self):
        return self.isWin() or self.isLose()


if __name__ == '__main__':
    g = Grid()
    g.mat[0][0] = 2
    g.mat[1][0] = 2
    g.mat[3][0] = 4

    while True:
        for i in g.mat:
            print(i)

        print(g.getAvailableMoves())

        v = input()

        g.move(v)
