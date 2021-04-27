from Agent import *
from Grid import Grid
import util
import numpy as np
import random
from Display import BeatifulDisplay

class ApproximateQAgent(Agent):
    def __init__(self, alpha = 0.001, epsilon = 0.5, discount = 0.5):
        self.weights = util.Counter()
        # self.QValues = util.Counter()
        
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(discount)
    
    def __str__(self):
        return 'Approximate Q Agent'
    
    def getMove(self, grid: Grid):
        if grid.isWin() or grid.isLose():
            return None
            
        actions = grid.getAvailableMoves()        
        maxQValue = self.getQValue(grid, actions[0])
        maxAction = actions[0]
                
        for action in actions:
            QValue = self.getQValue(grid, action)
            if  QValue >= maxQValue:
                maxQValue = QValue
                maxAction = action
            # print(maxQValue)
                
        nextState = util.getANewGrid(grid, action)
        
        reward = nextState.score
        
        if not nextState.isTerminal():
            self.update(grid, action, nextState, reward)
        
        #print(grid.mat)
        #print(reward)
        #print(maxQValue)
        #print(selectedAction)
        
        return maxAction
    
    def getFeatures(self, grid: Grid, action):
        """
        features:
        max tile (log2)
        second biggest tile(log2)
        is the biggest tile in the corner
        number of merges
        adjacent pairs differ by a factor of 2
        adjacent pairs that are equal
        number of free tiles
        monotonicity of the grid
        """
        newGrid = util.getANewGrid(grid, action)
        
        featureVector = util.Counter()
        
        #
        maxTile = newGrid.getMaxTile()
        maxTileLog = np.log2(maxTile)
        
        #
        secondMaxTile = 0
        for i in range(4):
            for j in range(4):
                if newGrid.getCellValue([i,j]) > secondMaxTile & newGrid.getCellValue([i,j]) != maxTile:
                    secondMaxTile = newGrid.getCellValue([i,j])
        secondMaxTileLog = np.log2(secondMaxTile)
        
        #
        isBiggestTileInCorner = 0
        if newGrid.getCellValue([0,0]) == maxTile or\
        newGrid.getCellValue([0,3]) == maxTile or\
        newGrid.getCellValue([3,0]) == maxTile or\
        newGrid.getCellValue([3,3]) == maxTile:
            isBiggestTileInCorner = 1
        
        #
        numMerges = max(0, len(grid.getAvailableCells()) - len(newGrid.getAvailableCells()))
        
        #
        adjDoubleCount = 0
        for i in range(4):
            for j in range(3):
                if newGrid.getCellValue([i,j]) == 2 * newGrid.getCellValue([i,j+1]) or\
                newGrid.getCellValue([i,j+1]) == 2 * newGrid.getCellValue([i,j]):
                    adjDoubleCount += 1
        for i in range(3):
            for j in range(4):
                if newGrid.getCellValue([i,j]) == 2 * newGrid.getCellValue([i+1,j]) or\
                newGrid.getCellValue([i+1,j]) == 2 * newGrid.getCellValue([i,j]):
                    adjDoubleCount += 1
        
        #
        adjEqualCount = 0
        for i in range(4):
            for j in range(3):
                if newGrid.getCellValue([i,j]) == newGrid.getCellValue([i,j+1]):
                    adjEqualCount += 1
        for i in range(3):
            for j in range(4):
                if newGrid.getCellValue([i,j]) == newGrid.getCellValue([i+1,j]):
                    adjEqualCount += 1
        #
        freeTiles = len(newGrid.getAvailableCells())
        
        #
        monotonicity = util.monotonicityHeuristic(newGrid)
        
        #
        # edge = util.edgeHeuristic(newGrid)
        
        #
        featureVector['maxTileLog'] = maxTileLog
        featureVector['secondMaxTileLog'] = secondMaxTileLog
        featureVector['isBiggestTileInCorner'] = isBiggestTileInCorner
        featureVector['numMerges'] = numMerges
        featureVector['adjDoubleCount'] = adjDoubleCount
        featureVector['adjEqualCount'] = adjEqualCount
        featureVector['freeTiles'] = freeTiles
        featureVector['monotonicity'] = monotonicity
        # featureVector['edge'] = edge

        # print(featureVector)
        
        return featureVector
    
    def getWeights(self):
        return self.weights
    
    def getQValue(self, state, action):
        QValue = 0
        featureVector = self.getFeatures(state, action)
        
        for key in featureVector.keys():
            QValue = QValue + self.weights[key] * featureVector[key]
        
        # print(QValue)
        
        return QValue
    
    def update(self, state, action, nextState, reward):
        nextActions = nextState.getAvailableMoves()
        nextStateValue = self.getQValue(nextState, nextActions[0])
        
        for nextAction in nextActions:
            nextStateValue = max(nextStateValue, self.getQValue(nextState, nextAction))
        
        difference = reward + self.discount * nextStateValue - self.getQValue(state, action)

        featureVector = self.getFeatures(state, action)
        for key in featureVector.keys():
            self.weights[key] += self.alpha * difference * featureVector[key]  
        
        # print(nextStateValue)
        # print(self.weights)

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
    return 0
    #return freeCellsHeuristic(grid) + maxValueHeuristic(grid)