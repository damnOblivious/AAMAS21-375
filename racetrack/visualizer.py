# visualizer.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Vaibhav Gupta

import math
import pygame
import numpy as np

from variables import *

################################################################################
################################################################################
WHITE = (255,255,255)
RED = (255, 0, 0)
BLUE = (0, 255 ,0)
GREEN = (0, 0, 255)
BLACK = (0, 0, 0)
################################################################################
################################################################################


class Visualizer:

    def __init__(self, layout):
        self.layout = layout
        self.window = False

    def setup(self):
        '''
        Does things which occur only at the beginning
        '''
        self.cell_edge = 10
        self.rows = self.layout.racetrack.width
        self.cols = self.layout.racetrack.height
        self.width = self.rows * self.cell_edge
        self.height = self.cols * self.cell_edge
        self.blockSize = (self.cell_edge, self.cell_edge)
        self.create_window()
        self.window = True

    def create_window(self):
        '''
        Creates window and assigns self.display variable
        '''
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Racetrack")

    def close_window(self):
        self.window = False
        pygame.quit()

    def draw(self, state = np.array([])):
        self.display.fill(0)
        for i in range(self.rows):
            for j in range(self.cols):
                if self.layout.racetrack[i][j] == WALL_CELL:
                    color = RED
                elif self.layout.racetrack[i][j] == FINISH_CELL:
                    color = BLUE
                # elif self.layout.racetrack[i][j] == START_CELL:
                #     color = GREEN
                else:
                    color = BLACK
                pygame.draw.rect(self.display, color, ((i*self.cell_edge,(self.cols - j - 1)*self.cell_edge), self.blockSize), 1)

        if len(state)>0:
            pygame.draw.rect(self.display, WHITE ,((state[0] * self.cell_edge, (self.cols - state[1] - 1) * self.cell_edge), self.blockSize), 0)

        pygame.display.update()

        global count

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.loop = False
                self.close_window()
                return 'stop'
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                pygame.image.save(vis.display, str(count)+'.png')
                count += 1
                self.loop = False

        return None

    def visualize_racetrack(self, state = np.array([])):
        '''
        Draws Racetrack in a pygame window
        '''
        if self.window == False:
            self.setup()
        self.loop = True
        while(self.loop):
            ret = self.draw(state)
            break
            if ret!=None:
                return ret
