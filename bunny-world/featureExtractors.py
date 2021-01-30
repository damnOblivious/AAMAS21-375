# featureExtractors.py
# --------------------
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


"Feature extractors for Pacman game states"

from game import Directions, Actions
import util

import numpy as np

class ActionMapping:
    NORTH = 'North'
    SOUTH = 'South'
    EAST = 'East'
    WEST = 'West'
    STOP = 'Stop'

    ActionToNumber = {
        NORTH :  0,
        SOUTH :  1,
        EAST  :  2,
        WEST  :  3,
        STOP  :  4
    }

    NumbertoAction = {
        0 : NORTH,
        1 : SOUTH,
        2 : EAST,
        3 : WEST,
        4 : STOP
    }


class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

def closestFoodCoordinates(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist, pos_x, pos_y
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

# def isCapturable(pos, ghost):
#     return abs(pos[0] - ghost[0]) + abs(pos[1] - ghost[1]) < 0.5

def closestGhost(pos, ghosts, walls):
    """
    closestGhost -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        # if dist > 10:
        #     return 5000
        ghosts = [(int(x), int(y)) for x,y in ghosts]
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a ghost at this location then exit
        if (pos_x, pos_y) in ghosts:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    return None


class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features

class CustomizedExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    """

    def getFeatures(self, state):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        puddles = state.getPuddles()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        x, y = state.getPacmanPosition()

        ghostDist = closestGhost((x, y), ghosts, walls)
        if ghostDist is not None:
            features["closest-ghost"] = float(ghostDist) / (walls.width * walls.height)

        features["x"] = x / (walls.width * walls.height)
        features["y"] = y / (walls.width * walls.height)

        # for i in range(walls.width):
        #     if walls[x-i][y]:
        #         features["closest_left_wall"] = i / (walls.width * walls.height)
        #         break
        # for i in range(walls.width):
        #     if walls[x+i][y]:
        #         features["closest_right_wall"] = i / (walls.width * walls.height)
        #         break
        # for i in range(walls.height):
        #     if walls[x][y+i]:
        #         features["closest_up_wall"] = i / (walls.width * walls.height)
        #         break
        # for i in range(walls.height):
        #     if walls[x][y-i]:
        #         features["closest_down_wall"] = i / (walls.width * walls.height)
        #         break

        features["left_wall"] = 1 if walls[x - 1][y] else 0
        features["right_wall"] = 1 if walls[x + 1][y] else 0
        features["up_wall"] = 1 if walls[x][y + 1] else 0
        features["down_wall"] = 1 if walls[x][y - 1] else 0

        # features["current_puddle"] = 1 if puddles[x][y] else 0
        # features["left_puddle"] = 1 if puddles[x - 1][y] else 0
        # features["right_puddle"] = 1 if puddles[x + 1][y] else 0
        # features["up_puddle"] = 1 if puddles[x][y + 1] else 0
        # features["down_puddle"] = 1 if puddles[x][y - 1] else 0
        #
        # features["closest_left_puddle"] = 100 / (walls.width * walls.height)
        # features["closest_right_puddle"] = 100 / (walls.width * walls.height)
        # features["closest_up_puddle"] = 100 / (walls.width * walls.height)
        # features["closest_down_puddle"] = 100 / (walls.width * walls.height)
        #
        # for i in range(walls.width):
        #     if (x - i >= 0) and puddles[x-i][y]:
        #         features["closest_left_puddle"] = i / (walls.width * walls.height)
        #         break
        # for i in range(walls.width):
        #     if (x + i < walls.width) and puddles[x+i][y]:
        #         features["closest_right_puddle"] = i / (walls.width * walls.height)
        #         break
        # for i in range(walls.height):
        #     if (y + i < walls.height) and puddles[x][y+i]:
        #         features["closest_up_puddle"] = i / (walls.width * walls.height)
        #         break
        # for i in range(walls.height):
        #     if (y - i >= 0) and puddles[x][y-i]:
        #         features["closest_down_puddle"] = i / (walls.width * walls.height)
        #         break
        #
        # features["closest_left_non_puddle"] = 100 / (walls.width * walls.height)
        # features["closest_right_non_puddle"] = 100 / (walls.width * walls.height)
        # features["closest_up_non_puddle"] = 100 / (walls.width * walls.height)
        # features["closest_down_non_puddle"] = 100 / (walls.width * walls.height)
        #
        # for i in range(walls.width):
        #     if (x - i >= 0) and (not puddles[x-i][y]):
        #         features["closest_left_non_puddle"] = i / (walls.width * walls.height)
        #         break
        # for i in range(walls.width):
        #     if (x + i < walls.width) and (not puddles[x+i][y]):
        #         features["closest_right_non_puddle"] = i / (walls.width * walls.height)
        #         break
        # for i in range(walls.height):
        #     if (y + i < walls.height) and (not puddles[x][y+i]):
        #         features["closest_up_non_puddle"] = i / (walls.width * walls.height)
        #         break
        # for i in range(walls.height):
        #     if (y - i >= 0) and (not puddles[x][y-i]):
        #         features["closest_down_non_puddle"] = i / (walls.width * walls.height)
        #         break

        for i, ghost in enumerate(ghosts):
            features["i_ghost_" + str(i)] = x - ghost[0]
            features["j_ghost_" + str(i)] = y - ghost[1]


        closestFoodCoordinate = closestFoodCoordinates((x, y), food, walls)

        features["number-of-food-left"] = state.getNumFood() / 100.
        # print features["number-of-food-left"]

        if closestFoodCoordinate is not None:
            dist, food_x, food_y = closestFoodCoordinate
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
            features["closest-food_x"] = float(x - food_x) / (walls.width * walls.height)
            features["closest-food_y"] = float(y - food_y) / (walls.width * walls.height)
        else:
            features["closest-food"] = 1.0
            features["closest-food_x"] = 1.0
            features["closest-food_y"] = 1.0
            # print '-------------------------------------------------------------------------------------'
        features.divideAll(10.0)

        # if np.array(features.values()).shape[0] != 11:
        #     print '-------------------------------------------------------------------------------------'
        #     print state, ghosts, x, y, features
        #     print '-------------------------------------------------------------------------------------'
        return np.array(features.values())

    def getFeatures2ghosts(self, state):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        puddles = state.getPuddles()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        x, y = state.getPacmanPosition()

        for i, ghost in enumerate(ghosts):
            ghostDist = closestGhost((x, y), [ghost], walls)
            if ghostDist is not None:
                features["dist_ghost_" + str(i)] = float(ghostDist) / (walls.width * walls.height)

        features["x"] = x / (walls.width * walls.height)
        features["y"] = y / (walls.width * walls.height)

        features["left_wall"] = 1 if walls[x - 1][y] else 0
        features["right_wall"] = 1 if walls[x + 1][y] else 0
        features["up_wall"] = 1 if walls[x][y + 1] else 0
        features["down_wall"] = 1 if walls[x][y - 1] else 0

        for i, ghost in enumerate(ghosts):
            features["i_ghost_" + str(i)] = x - ghost[0]
            features["j_ghost_" + str(i)] = y - ghost[1]


        closestFoodCoordinate = closestFoodCoordinates((x, y), food, walls)

        # features["number-of-food-left"] = state.getNumFood() / 100.
        # # print features["number-of-food-left"]
        #
        if closestFoodCoordinate is not None:
            dist, food_x, food_y = closestFoodCoordinate
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
            # features["closest-food_x"] = float(x - food_x) / (walls.width * walls.height)
            # features["closest-food_y"] = float(y - food_y) / (walls.width * walls.height)
        else:
            features["closest-food"] = 1.0
            # features["closest-food_x"] = 1.0
            # features["closest-food_y"] = 1.0
            # print '-------------------------------------------------------------------------------------'
        features.divideAll(10.0)

        # if np.array(features.values()).shape[0] != 11:
        #     print '-------------------------------------------------------------------------------------'
        #     print state, ghosts, x, y, features
        #     print '-------------------------------------------------------------------------------------'
        return np.array(features.values())

    def getReducedFeatures(self, state):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        puddles = state.getPuddles()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        x, y = state.getPacmanPosition()

        ghostDist = closestGhost((x, y), ghosts, walls)
        if ghostDist is not None:
            features["closest-ghost"] = float(ghostDist) / (walls.width * walls.height)

        # for i, ghost in enumerate(ghosts):
        #     features["i_ghost_" + str(i)] = x - ghost[0]
        #     features["j_ghost_" + str(i)] = y - ghost[1]

        closestFoodCoordinate = closestFoodCoordinates((x, y), food, walls)

        if closestFoodCoordinate is not None:
            dist, food_x, food_y = closestFoodCoordinate
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
            # features["closest-food_x"] = float(x - food_x) / (walls.width * walls.height)
            # features["closest-food_y"] = float(y - food_y) / (walls.width * walls.height)
        else:
            features["closest-food"] = 1.0
            # features["closest-food_x"] = 1.0
            # features["closest-food_y"] = 1.0
            # print '-------------------------------------------------------------------------------------'
        features.divideAll(10.0)

        # if np.array(features.values()).shape[0] != 11:
        #     print '-------------------------------------------------------------------------------------'
        #     print state, ghosts, x, y, features
        #     print '-------------------------------------------------------------------------------------'
        return np.array(features.values())

    def getLinearFeatures(self, state):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        x, y = state.getPacmanPosition()
        # features["bias"] = 1.

        ghostDist = closestGhost((x, y), ghosts, walls)
        # print state
        # print ghostDist
        if ghostDist is not None:
            features["closest-ghost"] = 1 if ghostDist < 4 else 0
            return 1 if ghostDist < 3 else 0
            # features["closest-ghost"] = float(ghostDist) / (walls.width * walls.height)

        # features["x"] = x / (walls.width * walls.height)
        # features["y"] = y / (walls.width * walls.height)

        # for i in range(walls.width):
        #     if walls[x-i][y]:
        #         features["closest_left_wall"] = i / (walls.width * walls.height)
        #         break
        # for i in range(walls.width):
        #     if walls[x+i][y]:
        #         features["closest_right_wall"] = i / (walls.width * walls.height)
        #         break
        # for i in range(walls.height):
        #     if walls[x][y+i]:
        #         features["closest_up_wall"] = i / (walls.width * walls.height)
        #         break
        # for i in range(walls.height):
        #     if walls[x][y-i]:
        #         features["closest_down_wall"] = i / (walls.width * walls.height)
        #         break

        # features["left_wall"] = 1 if walls[x - 1][y] else 0
        # features["right_wall"] = 1 if walls[x + 1][y] else 0
        # features["up_wall"] = 1 if walls[x][y + 1] else 0
        # features["down_wall"] = 1 if walls[x][y - 1] else 0

        # for i, ghost in enumerate(ghosts):
        #     features["i_ghost_" + str(i)] = x - ghost[0]
        #     features["j_ghost_" + str(i)] = y - ghost[1]


        # closestFoodCoordinate = closestFoodCoordinates((x, y), food, walls)
        #
        # features["number-of-food-left"] = state.getNumFood() / 100.
        # # print features["number-of-food-left"]
        #
        # if closestFoodCoordinate is not None:
        #     dist, food_x, food_y = closestFoodCoordinate
        #     # make the distance a number less than one otherwise the update
        #     # will diverge wildly
        #     features["closest-food"] = float(dist) / (walls.width * walls.height)
        #     features["closest-food_x"] = float(x - food_x) / (walls.width * walls.height)
        #     features["closest-food_y"] = float(y - food_y) / (walls.width * walls.height)
        # else:
        #     features["closest-food"] = 1.0
        #     features["closest-food_x"] = 1.0
        #     features["closest-food_y"] = 1.0
            # print '-------------------------------------------------------------------------------------'
        features.divideAll(10.0)

        # if np.array(features.values()).shape[0] != 11:
        #     print '-------------------------------------------------------------------------------------'
        #     print state, ghosts, x, y, features
        #     print '-------------------------------------------------------------------------------------'
        return features.values()

    def getPuddleFeatures(self, state):
        # extract the grid of food and wall locations and get the ghost locations
        walls = state.getWalls()
        puddles = state.getPuddles()

        features = util.Counter()

        x, y = state.getPacmanPosition()

        # features["x"] = x / (walls.width * walls.height)
        # features["y"] = y / (walls.width * walls.height)

        # features["dist"] = closestFood((x,y), puddles, walls)

        features["left_wall"] = 1 if walls[x - 1][y] else 0
        features["right_wall"] = 1 if walls[x + 1][y] else 0
        features["up_wall"] = 1 if walls[x][y + 1] else 0
        features["down_wall"] = 1 if walls[x][y - 1] else 0

        features["current_puddle"] = 1 if puddles[x][y] else 0
        features["left_puddle"] = 1 if puddles[x - 1][y] else 0
        features["right_puddle"] = 1 if puddles[x + 1][y] else 0
        features["up_puddle"] = 1 if puddles[x][y + 1] else 0
        features["down_puddle"] = 1 if puddles[x][y - 1] else 0


        features["closest_left_puddle"] = 100 / (walls.width * walls.height)
        features["closest_right_puddle"] = 100 / (walls.width * walls.height)
        features["closest_up_puddle"] = 100 / (walls.width * walls.height)
        features["closest_down_puddle"] = 100 / (walls.width * walls.height)

        for i in range(walls.width):
            if (x - i >= 0) and puddles[x-i][y]:
                features["closest_left_puddle"] = i / (walls.width * walls.height)
                break
        for i in range(walls.width):
            if (x + i < walls.width) and puddles[x+i][y]:
                features["closest_right_puddle"] = i / (walls.width * walls.height)
                break
        for i in range(walls.height):
            if (y + i < walls.height) and puddles[x][y+i]:
                features["closest_up_puddle"] = i / (walls.width * walls.height)
                break
        for i in range(walls.height):
            if (y - i >= 0) and puddles[x][y-i]:
                features["closest_down_puddle"] = i / (walls.width * walls.height)
                break

        features["closest_left_non_puddle"] = 100 / (walls.width * walls.height)
        features["closest_right_non_puddle"] = 100 / (walls.width * walls.height)
        features["closest_up_non_puddle"] = 100 / (walls.width * walls.height)
        features["closest_down_non_puddle"] = 100 / (walls.width * walls.height)

        for i in range(walls.width):
            if (x - i >= 0) and (not puddles[x-i][y]):
                features["closest_left_non_puddle"] = i / (walls.width * walls.height)
                break
        for i in range(walls.width):
            if (x + i < walls.width) and (not puddles[x+i][y]):
                features["closest_right_non_puddle"] = i / (walls.width * walls.height)
                break
        for i in range(walls.height):
            if (y + i < walls.height) and (not puddles[x][y+i]):
                features["closest_up_non_puddle"] = i / (walls.width * walls.height)
                break
        for i in range(walls.height):
            if (y - i >= 0) and (not puddles[x][y-i]):
                features["closest_down_non_puddle"] = i / (walls.width * walls.height)
                break
        #
        #     # print '-------------------------------------------------------------------------------------'

        features.divideAll(10.0)

        # if np.array(features.values()).shape[0] != 11:
        #     print '-------------------------------------------------------------------------------------'
        #     print state, ghosts, x, y, features
        #     print '-------------------------------------------------------------------------------------'
        return np.array(features.values())

    def getFoodFeatures(self, state):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        puddles = state.getPuddles()
        features = util.Counter()

        x, y = state.getPacmanPosition()

        features["x"] = x / (walls.width * walls.height)
        features["y"] = y / (walls.width * walls.height)

        features["left_wall"] = 1 if walls[x - 1][y] else 0
        features["right_wall"] = 1 if walls[x + 1][y] else 0
        features["up_wall"] = 1 if walls[x][y + 1] else 0
        features["down_wall"] = 1 if walls[x][y - 1] else 0

        # for i in range(walls.width):
        #     if walls[x-i][y]:
        #         features["closest_left_wall"] = i / (walls.width * walls.height)
        #         break
        # for i in range(walls.width):
        #     if walls[x+i][y]:
        #         features["closest_right_wall"] = i / (walls.width * walls.height)
        #         break
        # for i in range(walls.height):
        #     if walls[x][y+i]:
        #         features["closest_up_wall"] = i / (walls.width * walls.height)
        #         break
        # for i in range(walls.height):
        #     if walls[x][y-i]:
        #         features["closest_down_wall"] = i / (walls.width * walls.height)
        #         break

        # features["current_puddle"] = 1 if puddles[x][y] else 0
        # features["left_puddle"] = 1 if puddles[x - 1][y] else 0
        # features["right_puddle"] = 1 if puddles[x + 1][y] else 0
        # features["up_puddle"] = 1 if puddles[x][y + 1] else 0
        # features["down_puddle"] = 1 if puddles[x][y - 1] else 0

        features["number-of-food-left"] = state.getNumFood() / 100.
        closestFoodCoordinate = closestFoodCoordinates((x, y), food, walls)
        if closestFoodCoordinate is not None:
            dist, food_x, food_y = closestFoodCoordinate
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
            features["closest-food_x"] = float(x - food_x) / (walls.width * walls.height)
            features["closest-food_y"] = float(y - food_y) / (walls.width * walls.height)
        else:
            features["closest-food"] = 1.0
            features["closest-food_x"] = 1.0
            features["closest-food_y"] = 1.0
            # print '-------------------------------------------------------------------------------------'
        features.divideAll(10.0)

        # if np.array(features.values()).shape[0] != 9:
        #     print '-------------------------------------------------------------------------------------'
        #     print state, ghosts, x, y, features
        #     print '-------------------------------------------------------------------------------------'
        return np.array(features.values())

    def getGhostFeatures(self, state):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        x, y = state.getPacmanPosition()

        ghostDist = closestGhost((x, y), ghosts, walls)
        if ghostDist is not None:
            features["closest-ghost"] = float(ghostDist) / (walls.width * walls.height)

        features["x"] = x / (walls.width * walls.height)
        features["y"] = y / (walls.width * walls.height)

        features["left_wall"] = 1 if walls[x - 1][y] else 0
        features["right_wall"] = 1 if walls[x + 1][y] else 0
        features["up_wall"] = 1 if walls[x][y + 1] else 0
        features["down_wall"] = 1 if walls[x][y - 1] else 0

        # for i in range(walls.width):
        #     if walls[x-i][y]:
        #         features["closest_left_wall"] = i / (walls.width * walls.height)
        #         break
        # for i in range(walls.width):
        #     if walls[x+i][y]:
        #         features["closest_right_wall"] = i / (walls.width * walls.height)
        #         break
        # for i in range(walls.height):
        #     if walls[x][y+i]:
        #         features["closest_up_wall"] = i / (walls.width * walls.height)
        #         break
        # for i in range(walls.height):
        #     if walls[x][y-i]:
        #         features["closest_down_wall"] = i / (walls.width * walls.height)
        #         break

        # features["left_wall"] = 1 if walls[x - 1][y] else 0
        # features["right_wall"] = 1 if walls[x + 1][y] else 0
        # features["up_wall"] = 1 if walls[x][y + 1] else 0
        # features["down_wall"] = 1 if walls[x][y - 1] else 0

        for i, ghost in enumerate(ghosts):
            features["i_ghost_" + str(i)] = x - ghost[0]
            features["j_ghost_" + str(i)] = y - ghost[1]

        features.divideAll(10.0)

        # if np.array(features.values()).shape[0] != 8:
        #     print '-------------------------------------------------------------------------------------'
        #     print state, ghosts, x, y, features
        #     print '-------------------------------------------------------------------------------------'
        return np.array(features.values())

    def getGhostFeatures1(self, state):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = [state.getGhostPositions()[0]]

        features = util.Counter()

        x, y = state.getPacmanPosition()

        ghostDist = closestGhost((x, y), ghosts, walls)
        if ghostDist is not None:
            features["closest-ghost"] = float(ghostDist) / (walls.width * walls.height)

        features["x"] = x / (walls.width * walls.height)
        features["y"] = y / (walls.width * walls.height)

        features["left_wall"] = 1 if walls[x - 1][y] else 0
        features["right_wall"] = 1 if walls[x + 1][y] else 0
        features["up_wall"] = 1 if walls[x][y + 1] else 0
        features["down_wall"] = 1 if walls[x][y - 1] else 0

        for i, ghost in enumerate(ghosts):
            features["i_ghost_" + str(i)] = x - ghost[0]
            features["j_ghost_" + str(i)] = y - ghost[1]

        features.divideAll(10.0)

        # if np.array(features.values()).shape[0] != 8:
        #     print '-------------------------------------------------------------------------------------'
        #     print state, ghosts, x, y, features
        #     print '-------------------------------------------------------------------------------------'
        return np.array(features.values())

    def getGhostFeatures2(self, state):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = [state.getGhostPositions()[1]]

        features = util.Counter()

        x, y = state.getPacmanPosition()

        ghostDist = closestGhost((x, y), ghosts, walls)
        if ghostDist is not None:
            features["closest-ghost"] = float(ghostDist) / (walls.width * walls.height)

        features["x"] = x / (walls.width * walls.height)
        features["y"] = y / (walls.width * walls.height)

        features["left_wall"] = 1 if walls[x - 1][y] else 0
        features["right_wall"] = 1 if walls[x + 1][y] else 0
        features["up_wall"] = 1 if walls[x][y + 1] else 0
        features["down_wall"] = 1 if walls[x][y - 1] else 0

        for i, ghost in enumerate(ghosts):
            features["i_ghost_" + str(i)] = x - ghost[0]
            features["j_ghost_" + str(i)] = y - ghost[1]

        features.divideAll(10.0)

        # if np.array(features.values()).shape[0] != 8:
        #     print '-------------------------------------------------------------------------------------'
        #     print state, ghosts, x, y, features
        #     print '-------------------------------------------------------------------------------------'
        return np.array(features.values())
