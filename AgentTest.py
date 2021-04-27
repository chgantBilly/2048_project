from Agent import *
from MCTSAgent import MCTSAgent
from ApproximateQAgent import  ApproximateQAgent
from ExpectimaxAgent import expectimaxAgent
from Game import Game
from Display import BeatifulDisplay

from collections import defaultdict

class AgentTest:
    def __init__(self, testAgent:Agent=GreedyAgent(), testTurns:int=100, verbose:int=0):
        self.testAgent = testAgent
        self.testTurns = testTurns
        self.verbose = verbose

        self.scores = []
        self.winCount = 0
        self.scoreDict = defaultdict(int)

        for i in range(3,15):
            self.scoreDict[int(2**i)] = 0

    def printStat(self):
        for i in range(3,14)[::-1]:
            curr_score = int(2 ** i)
            self.scoreDict[curr_score] += self.scoreDict[int(curr_score * 2)]

        print('max score = {:.0f}, min score = {:.0f}, avg score = {:.0f}, winning rate = {:.2%}'.format(
            max(self.scores), min(self.scores), sum(self.scores) / self.testTurns, self.winCount / self.testTurns))

        print('Highest score counts:')
        for i in range(5,12):
            curr_score = int(2 ** i)
            print('===\t{}:{}/{}\t==='.format(curr_score, self.scoreDict[curr_score], self.testTurns))


    def main(self):
        print(self.testAgent)
        print('===== Test ' + str(self.testAgent) + ' for ' + str(self.testTurns) + ' turn(s)=======')
        for i in range(self.testTurns):
            game = Game()
            game.setAgent(self.testAgent)
            if self.verbose == 0:
                game.setDisplay(BeatifulDisplay(), -1)
            else:
                game.setDisplay(BeatifulDisplay(), self.verbose)
                print('Start test {}'.format((i + 1)))
            isWin, score = game.main()
            self.scores.append(score)
            self.scoreDict[score] += 1
            if isWin:
                self.winCount += 1
            if i > 0 and (i + 1) % 5 == 0:
                print('===== {}/{} tests done ====='.format((i + 1), self.testTurns))
        self.printStat()


if __name__ == '__main__':
    test = AgentTest(testAgent=MCTSAgent(simulateIter=50, rollingOutDepth=10),testTurns=50)
    test.main()
