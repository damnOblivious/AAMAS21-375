# environment.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Vaibhav Gupta

import math
import numpy as np
import pygame
import random

import layout_parser
from rewardScales import *
from variables import *
from keras.models import model_from_json

class Agent:

    def map_to_1D(self,action):
        alpha = [(-1,-1),(-1,0),(0,-1),(-1,1),(0,0),(1,-1),(0,1),(1,0),(1,1)]
        for i,x in zip(range(9),alpha):
            if action[0]==x[0] and action[1]==x[1]:
                return i

    def map_to_2D(self,action):
        alpha = [(-1,-1),(-1,0),(0,-1),(-1,1),(0,0),(1,-1),(0,1),(1,0),(1,1)]
        return alpha[action]

    def saveModel(self, model, file_name):
        model_json = model.to_json()
        with open('weights/' + file_name + '.json', "w") as json_file:
            json_file.write(model_json)
        model.save_weights('weights/' + file_name + '.h5')

    def loadModel(self, file_name):
        json_file = open('weights/' + file_name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights('weights/' + file_name + '.h5')
        return loaded_model

    def setRewardScales(self):
        if self.trialNum == -1:
            return
        self.modifiedFinishReward = rewardScales[self.trialNum][0]
        self.modifiedCollisionPenalty = rewardScales[self.trialNum][1]
        self.modifiedTimeFinish = rewardScales[self.trialNum][2]
        self.modifiedTimeCollision = rewardScales[self.trialNum][3]

    def getModifiedFinishReward(self, reward, shapedReward):
        if reward == TIME_STEP_PENALTY + FINISH_REWARD + COLLISION_PENALTY:
            reward = self.modifiedTimeFinish + self.modifiedFinishReward
        elif reward == TIME_STEP_PENALTY + COLLISION_PENALTY:
            reward = self.modifiedTimeFinish
        elif reward == TIME_STEP_PENALTY + FINISH_REWARD:
            reward = self.modifiedTimeFinish + self.modifiedFinishReward
        elif reward == TIME_STEP_PENALTY:
            reward = self.modifiedTimeFinish

        reward += shapedReward

        return reward / 50.0

    def getModifiedCollisionReward(self, reward, shapedReward):
        if reward == TIME_STEP_PENALTY + FINISH_REWARD + COLLISION_PENALTY:
            reward = self.modifiedCollisionPenalty
        elif reward == TIME_STEP_PENALTY + COLLISION_PENALTY:
            reward = self.modifiedCollisionPenalty
        elif reward == TIME_STEP_PENALTY + FINISH_REWARD:
            reward = -self.modifiedTimeCollision
        elif reward == TIME_STEP_PENALTY:
            reward = -self.modifiedTimeCollision

        return reward / 50.0

    def __init__(self, layout, trialNum = -1):
        self.layout = layout
        self.trialNum = trialNum
        if self.trialNum != -1:
            self.setRewardScales()
            self.finishReward = self.getModifiedFinishReward
            self.collisionReward = self.getModifiedCollisionReward
        else:
            self.finishReward = self.getFinishReward
            self.collisionReward = self.getCollisionReward


    def update(self, state, action, nextState, reward):
        pass

    def getCollisionReward(self, reward, shapedReward):
        pass

    def getFinishReward(self, reward, shapedReward):
        pass

    def get_action(self, state):
        '''
        Returns action given state using policy
        '''
        # return self.map_to_2D(policy(state, self.possible_actions(state[2:4])))
        # action = np.random.choice(possible_actions)
        # return self.map_to_2D(policy(state, self.possible_actions(state[2:4])))
        pass
