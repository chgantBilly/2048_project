from Grid import Grid
from random import randint
from Agent import *
from Display import Display, BeatifulDisplay
from time import sleep

from MCTSAgent import MCTSAgent

defaultInitialTiles = 2
defaultProbability = 0.9
defaultDisplayStep = -1

actionDic = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT"
}

class Game:
    def __init__(self, size=4):
        self.grid = Grid(size)
        self.possibleNewTiles = [2]
        self.probability = defaultProbability
        self.initTiles = defaultInitialTiles
        self.agent = GreedyAgent()
        self.displayer = BeatifulDisplay()
        self.displayStep = defaultDisplayStep
        self._debugFlag = False
        self._failMess = None

    def setAgent(self, agent:Agent):
        self.agent = agent

    def setDisplay(self, displayer:Display, displayStep:int):
        self.displayer = displayer
        self.displayStep = displayStep

    def isLose(self):
        return not self.grid.canMove()

    def isWin(self):
        return self.grid.getMaxTile() == 2048

    def getNewTileValue(self):
        return self.possibleNewTiles[randint(0,len(self.possibleNewTiles)-1)]

    def insertRandonTile(self):
        tileValue = self.getNewTileValue()
        cells = self.grid.getAvailableCells()
        cell = cells[randint(0, len(cells) - 1)]
        if not cell:
            self._debugFlag = True
            self._failMess = "No available empty cell"
            raise ValueError(self._failMess)
        self.grid.setCellValue(cell, tileValue)

    def display(self):
        self.displayer.display(self.grid)

    def getScore(self):
        return self.grid.getMaxTile()

    def main(self):
        for i in range(self.initTiles):
            self.insertRandonTile()
        if self.displayStep > 0:
            self.display()

        step = 1

        while not self.isWin() and not self.isLose() and not self._debugFlag:
            gridCopy = self.grid.clone()
            move = self.agent.getMove(gridCopy)
            #print(actionDic[move])
            if move is not None and 0 <= move < 4:
                if self.grid.canMove([move]):
                    self.grid.move(move)
                else:
                    self._debugFlag = True
                    self._failMess = "Invalid Agent Move: the grid would not change"
                    break
            else:
                self._debugFlag = True
                self._failMess = "Invalid Agent Move: Value Error: " + str(move)
                break

            self.insertRandonTile()
            if self.displayStep > 0 and step % self.displayStep == 0:
                #sleep(0.1)
                print('step: ',step)
                self.display()
            #print(self.getScore())
            step += 1

        if self._debugFlag:
            raise ValueError(self._failMess)

        if self.displayStep > 0:
            self.display()
            print('Ends in step',step)
            if self.isWin():
                print("Agent wins!!")
            if self.isLose():
                print("Agent Loses!!")

        return self.isWin(), self.getScore()


if __name__ == '__main__':
    game = Game()
    game.setAgent(MCTSAgent(simulateIter=50, rollingOutDepth=10))
    game.setDisplay(BeatifulDisplay(), 100)
    game.main()









