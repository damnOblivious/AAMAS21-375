# layout.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Grid
import os
import random
import numpy as np
from variables import *

VISIBILITY_MATRIX_CACHE = {}

class Layout:
    """
    A Layout manages the static information about the game board.
    """

    def __init__(self, layoutText):
        self.width = len(layoutText[0])
        self.height= len(layoutText)
        self.racetrack = Grid(self.width, self.height, False)
        self.startStates = []
        self.finishStates = []
        self.agentPositions = []
        self.possibleAgentPositions = []
        self.processLayoutText(layoutText)
        self.layoutText = layoutText

    def isWall(self, pos):
        x, col = pos
        return self.racetrack[x][col] == WALL_CELL

    def getRandomLegalPosition(self):
        x = random.choice(range(self.width))
        y = random.choice(range(self.height))
        while self.isWall( (x, y) ):
            x = random.choice(range(self.width))
            y = random.choice(range(self.height))
        return (x,y)

    def getRandomCorner(self):
        poses = [(1,1), (1, self.height - 2), (self.width - 2, 1), (self.width - 2, self.height - 2)]
        return random.choice(poses)

    def getFurthestCorner(self, pacPos):
        poses = [(1,1), (1, self.height - 2), (self.width - 2, 1), (self.width - 2, self.height - 2)]
        dist, pos = max([(manhattanDistance(p, pacPos), p) for p in poses])
        return pos

    def __str__(self):
        return "\n".join(self.layoutText)

    def deepCopy(self):
        return Layout(self.layoutText[:])

    def processLayoutText(self, layoutText):
        """
        Coordinates are flipped from the input format to the (x,y) convention here

        The shape of the maze.  Each character
        represents a different type of object.
         # - start  - 1
         % - Wall   - 2
         . - Finish - 3
         P - Player - 4
        Other characters are ignored.
        """
        maxY = self.height - 1
        for y in range(self.height):
            for x in range(self.width):
                layoutChar = layoutText[maxY - y][x]
                self.processLayoutChar(x, y, layoutChar)
        # import random
        # random.seed(9)
        # random.shuffle(self.possibleAgentPositions)
        # for i, pos in enumerate(self.possibleAgentPositions[:3]):
        #     self.food[x][y] = False
        #     self.agentPositions.append( (i == 0, pos) )
        # x = 1 + random.randrange(self.width - 1)
        # y = 1 + random.randrange(self.height - 1)
        # print y, x
        # print self.height, self.width
        # self.agentPositions = [ ( True, (9, 3))]
        # randPos = self.getRandomLegalPosition()
        # randPos = self.getRandomCorner()
        # print randPos
        # self.agentPositions = [ ( True, randPos)]
        self.agentPositions.sort()
        self.agentPositions = [ ( i == 0, pos) for i, pos in self.agentPositions]

    def processLayoutChar(self, x, y, layoutChar):
        if layoutChar == '#':
            self.racetrack[x][y] = START_CELL
            self.startStates.append( (x, y) )
        elif layoutChar == '%':
            self.racetrack[x][y] = WALL_CELL
        elif layoutChar == '.':
            self.finishStates.append( (x, y) )
            self.racetrack[x][y] = FINISH_CELL
        elif layoutChar == 'P':
            self.racetrack[x][y] = PLAYER_CELL
            # self.racetrack.append( (0, (x, y) ) )

def getLayout(name, back = 2):
    if name.endswith('.lay'):
        layout = tryToLoad('layouts/' + name)
        if layout == None: layout = tryToLoad(name)
    else:
        layout = tryToLoad('layouts/' + name + '.lay')
        if layout == None: layout = tryToLoad(name + '.lay')
    if layout == None and back >= 0:
        curdir = os.path.abspath('.')
        os.chdir('..')
        layout = getLayout(name, back -1)
        os.chdir(curdir)
    return layout

def tryToLoad(fullname):
    if(not os.path.exists(fullname)): return None
    f = open(fullname)
    try:
        # lines = [line.strip() for line in f]
        # lines.reverse()
        # return Layout(lines)
        return Layout([line.strip() for line in f])
    finally: f.close()
