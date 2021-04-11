from Grid import Grid
from util import *
from random import randint, shuffle
from abc import ABC, abstractmethod


class Agent(ABC):

    @abstractmethod
    def getMove(self, grid: Grid):
        pass


class RandomAgent(Agent):
    """
    Just random
    """
    def __str__(self):
        return 'Random Agent'

    def getMove(self, grid: Grid):
        #time.sleep(0.1)
        moves = grid.getAvailableMoves()
        return moves[randint(0,len(moves)-1)]


class GreedyAgent(Agent):
    """
    A greedy agent:
    always return the move whose resulting grids has the least (average) number of numbered tiles.
    """

    def __init__(self, policy = freeCellsHeuristic):
        self.policy = policy

    def __str__(self):
        return 'Greedy Agent'

    def getScore(self, grid, move):
        score = 0
        rets = getKNewGrid(grid,move,4)
        for ret in rets:
            score += self.policy(ret)
        return score / len(rets)



    def getMove(self, grid: Grid):
        #time.sleep(0.1)
        moves = grid.getAvailableMoves()
        shuffle(moves)
        maxScore = float('-inf')
        maxMove = None
        for move in moves:
            gridCopy = grid.clone()
            score = self.getScore(gridCopy, move)
            if score > maxScore:
                maxScore, maxMove = score, move
        return maxMove