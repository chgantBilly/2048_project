from Agent import *
from Grid import Grid
from util import *
from random import choice,shuffle
from math import sqrt, log
from Display import BeatifulDisplay

class MCTSNode:
    def __init__(self, parent, state:Grid):
        self.parent = parent
        self.state = state
        self.children = {} # key: actions, values: A list of result Nodes(after adding random tite)
        self.visitCount = 0
        self.rewardSum = 0
        self.availableMoves = self.state.getAvailableMoves()

    def __hash__(self):
        return hash(self.state.mat)

    def __eq__(self, other):
        if isinstance(other,MCTSNode):
            return other.state.mat == self.state.mat
        return False


    def getVisitCount(self):
        return self.visitCount

    def getRewardSum(self):
        return self.rewardSum

    def getState(self):
        return self.state

    def getAvailableMoves(self):
        return self.availableMoves

    def hasExploredMove(self, move):
        return move in self.children.keys()

    def getExploreMoves(self):
        return list(self.children.keys())

    def isWin(self):
        return self.state.isWin()

    def isLose(self):
        return self.state.isLose()

    def isLeaf(self):
        return (self.isWin() or self.isLose()) or self.visitCount == 0

    ##methods for MCTS
    def selectMove(self, C = 1.41):
        '''
        :param selector: a function for selection
        :return: the result move of selection part for searching
        '''
        if self.isLose() or self.isWin():
            return None
        availableMoves = self.getAvailableMoves()
        exploredMoves = self.getExploreMoves()
        if len(availableMoves) != len(exploredMoves):
            # if the node haven't explored all the moves available, choose a un-explored move randomly
            return choice([m for m in availableMoves if m not in exploredMoves])
        bestMove, bestScore = None, float('-inf')
        parentVisitCount = self.getVisitCount()
        shuffle(exploredMoves)  # break tie with random order
        for m in exploredMoves:
            assert self.hasExploredMove(m)
            moveVisitCount = sum((child.getVisitCount() for child in self.children[m]))
            moveRewardSum = sum((child.getRewardSum() for child in self.children[m]))
            moveScore = moveRewardSum / moveVisitCount + C * sqrt(log(parentVisitCount) / moveVisitCount) #UCB
            if moveScore > bestScore:
                bestMove, bestScore = m, moveScore
        return bestMove

    def exploreMove(self, move):
        assert move in self.availableMoves
        grid = getANewGrid(self.state, move)

        childNode = MCTSNode(parent=self, state=grid)

        if self.hasExploredMove(move):
            if childNode not in self.children[move]:
                self.children[move].append(childNode)
            else:
                for c in self.children[move]:
                    if c == childNode:
                        childNode = c
        else:
            self.children[move] = [childNode]
        assert childNode in self.children[move]

        return childNode

    def backpropagate(self, reward):
        self.rewardSum += reward
        self.visitCount += 1
        if self.parent:
            self.parent.backpropagate(reward)


    def bestMove(self):
        '''
        :return: the best move (with highest reward) when making decision
        '''

        if self.isLose() or self.isWin():
            return None
        exploredMoves = self.getExploreMoves()
        shuffle(exploredMoves) # break the tie with random
        maxReward, maxMove = float('-inf'), None
        for m in exploredMoves:
            moveVisitCount = sum((child.getVisitCount() for child in self.children[m]))
            moveRewardSum = sum((child.getRewardSum() for child in self.children[m]))
            moveReward = moveRewardSum / moveVisitCount
            if moveReward > maxReward:
                maxReward, maxMove = moveReward, m

        return maxMove


class MCTSAgent(Agent):
    def __init__(self, simulateIter = 100, rollingOutDepth = 3, explorationParameter = 1.41, verbose = 0):
        self.simulateIter = simulateIter
        self.rollingOutDepth = rollingOutDepth
        assert explorationParameter > 0
        self.explorationParameter = explorationParameter
        assert verbose in [0,1]
        self.verbose = int(verbose)

    def __str__(self):
        return 'MCTS 2048 Agent (done by Guangyi)'


    def getMove(self, grid: Grid):
        self.root = MCTSNode(parent=None, state=grid)
        if grid.isWin() or grid.isLose():
            _ = input()
        for i in range(self.simulateIter):
            if self.verbose:
                print('===== Monte Carlo Tree Search Iter {} ====='.format(i+1))
            self.tree_policy()
        return self.root.bestMove()

    def tree_policy(self):

        if self.verbose:
            print('Start Selection from the root')

        current_node = self.root
        while not current_node.isLeaf():

            if self.verbose:
                print('Selection from state')
                print(current_node.state.mat)

            move = current_node.selectMove(C=self.explorationParameter)
            current_node = current_node.exploreMove(move)

            if self.verbose:
                print('Selected move', str(move))
                print('Next state:')
                print(current_node.state.mat)

        if self.verbose:
            print('Expanded state:')
            print(current_node.state.mat)
            print('Start RollingOut')

        reward = self.rollingOut(current_node)

        if self.verbose:
            print('Get Reward', reward)

        current_node.backpropagate(reward)


    def rollingOut(self, node: MCTSNode):
        grid = node.state
        depth = 0
        rollingOut = GreedyAgent(estimate)
        while not grid.isTerminal() and depth < self.rollingOutDepth:
            moves = grid.getAvailableMoves()
            move = rollingOut.getMove(grid)
            grid = getANewGrid(grid, move)
            depth += 1
        if self.verbose:
            print('rolling Out Depth {}'.format(depth))
            print(grid.mat)
        reward = estimate(grid)
        return reward


def estimate(grid: Grid):
    '''
    :param grid:
    :type grid:
    :return:
    :rtype:
    '''
    # return freeCellsHeuristic(grid)
    if grid.isWin():
        return 1000
    if grid.isLose():
        return -1000
    return (maxValueHeuristic(grid) + freeCellsHeuristic(grid) + 0.25 * edgeHeuristic(grid) + monotonicityHeuristic(grid))
    #return freeCellsHeuristic(grid) + maxValueHeuristic(grid)



if __name__ == '__main__':
    g = Grid()
    g.mat = [[2,0,4,4],[0,2,4,4],[4,8,32,256],[16,128,512,1024]]
    #dis = BeatifulDisplay()
    #dis.display(g)

    agent = MCTSAgent(simulateIter=50, rollingOutDepth=2,verbose=1)
    move = agent.getMove(g)
    print(move)









