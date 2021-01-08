import numpy as np
import util
from variables import *

class FeatureExtractor():

    def __init__(self, layout):
        self.layout = layout
        self.rows = self.layout.racetrack.width
        self.cols = self.layout.racetrack.height
        # self.divide_factor = (self.rows * self.cols)
        self.divide_all = 100.

    def getSimplestFeatures(self, state):
        features = util.Counter()
        features = self.__addBasicFeatures(state, features)

        features.divideAll(self.divide_all)
        return np.array(features.values())

    def getCollisionFeatures(self, state):
        features = util.Counter()
        features = self.__addBasicFeatures(state, features)
        features = self.__addWallDistFeatures(state, features)

        features.divideAll(self.divide_all)
        return np.array(features.values())

    def __addBasicFeatures(self, state, features = util.Counter()):
        features["x"] = state[0]
        features["y"] = state[1]
        features["vx"] = state[2]
        features["vy"] = state[3]
        return features

    def __addWallDistFeatures(self, state, features = util.Counter()):
        float_x, float_y = state[:2]
        x, y = int(float_x), int(float_y)

        for i in range(self.rows):
            if self.layout.racetrack[x-i][y] == WALL_CELL:
                features["closest_left_wall"] = (float_x - x + i)  # / self.divide_factor
                break
        for i in range(self.rows):
            if self.layout.racetrack[x+i][y] == WALL_CELL:
                features["closest_right_wall"] = (x + i - float_x) # / self.divide_factor
                break
        for i in range(self.cols):
            if self.layout.racetrack[x][y+i] == WALL_CELL:
                features["closest_up_wall"] = (y + i - float_y) # / self.divide_factor
                break
        for i in range(self.cols):
            if self.layout.racetrack[x][y-i] == WALL_CELL:
                features["closest_down_wall"] = (float_y - y + i) # / self.divide_factor
                break
        # print features

        return features
