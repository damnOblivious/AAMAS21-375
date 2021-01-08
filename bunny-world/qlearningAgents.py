# qlearningAgents.py
# ------------------
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

identifier = '1'

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from memory import SequentialMemory
from featureExtractors import closestFood, closestGhost
from game import Actions

from keras.models import model_from_json

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input
from keras.optimizers import RMSprop, Adam
from keras.layers.merge import Add, Concatenate
import keras.backend as K

# from keras.layers import Dense, Dropout, Input



from collections import deque
import numpy as np
import random,util,math


TIME_PENALTY = -1 # Number of points lost each round
FOOD_REWARD = 10
DIE_PENALTY = -20
EAT_ALL_FOOD_REWARD = 0
PUDDLE_PENALTY = .8

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


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.qValues = util.Counter()
        self.totalTrainingSteps = 0

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.qValues[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        possibleActions = self.getLegalActions(state)
        if possibleActions:
            maxv = float("-inf")
            for action in possibleActions:
                q = self.getQValue(state, action)
                if q >= maxv:
                    maxv = q
            return maxv
        return 0.0

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        possibleActions = self.getLegalActions(state)
        if possibleActions:
            maxv = float("-inf")
            bestAction = None
            for action in possibleActions:
                q = self.getQValue(state, action)
                if q >= maxv:
                    maxv = q
                    bestAction = action
            return bestAction
        return None

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        possibleActions = self.getLegalActions(state)
        action = None
        # print self.foodAgent.getQValues(state)
        "*** YOUR CODE HERE ***"
        if possibleActions:
            return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        possibleActions = self.getLegalActions(nextState)
        R = reward
        if possibleActions:
            Q = []
            for a in possibleActions:
                Q.append(self.getQValue(nextState, a))
            R = reward + self.discount * max(Q)
        self.qValues[(state, action)] = self.getQValue(state, action) + self.alpha * (R - self.getQValue(state, action))

    def getPolicy(self, state):
        if not self.selfTesting:
            self.totalTrainingSteps += 1

        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()
        self.epsilon = 1
        self.min_epsilon = 0.01
        self.decay = .9999


    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        f = self.featExtractor.getFeatures(state, action)
        qv = 0
        for feature in f:
            qv = qv + self.weights[feature] * f[feature]
        return qv

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"

        if self.epsilon > self.min_epsilon:
            self.epsilon = self.epsilon * self.decay

        R = reward
        f = self.featExtractor.getFeatures(state, action)
        alphadiff = self.alpha * ((R + self.discount * self.getValue(nextState)) - self.getQValue(state, action))
        for feature in f.keys():
            self.weights[feature] = self.weights[feature] + alphadiff * f[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass


class DQNBaselineAgent(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', **args):
        """
        TODO:
            save weights
        """
        PacmanQAgent.__init__(self, **args)
        self.nb_features = 13
        self.neuralNet = NeuralNetwork(self.nb_features, 5)
        self.model = self.neuralNet.model
        self.replay_memory_buffer = deque(maxlen=50000)
        self.batch_size = 32
        self.nb_actions = 5
        self.epsilon = 1
        self.min_epsilon = 0.01
        self.decay = .9995
        self.discount = .8
        self.extractor = CustomizedExtractor().getFeatures
        self.lastSavedWeights = -1

        print '----------'
        print '############ DQNBaselineAgent ############'
        print 'Epsilon Decay = %s, Discount Factor = %.2f' % (self.decay, self.discount)
        print 'Rewards for Agent: (Eat ghost) = %.2f, (Eat Food) = %.2f, (Death Penalty) = %.2f, (Time Penalty) = %.2f' % \
        (self.getModifiedReward(50), self.getModifiedReward(10), self.getModifiedReward(-500), self.getModifiedReward(-1))
        print '----------'


    def getQValue(self, state, action):
        qValues = self.model.predict(self.extractor(state))
        return qValues[ActionMapping.ActionToNumber[action]]

    # def getModifiedReward(self, reward):
    #     if reward > 20:
    #         reward = 50.    # Eat ghost   (Yum! Yum!)
    #     elif reward > 0:
    #         reward = 10.    # Eat food    (Yum!)
    #     elif reward < -10:
    #         reward = -20.   # Get eaten   (Ouch!) -500
    #     elif reward < 0:
    #         reward = -1    # Punish time (Pff..)
    #     return reward / 10.0

    def getModifiedReward(self, reward):

        MODIFIED_PUDDLE_PENALTY = -5
        MODIFIED_TIME_PENALTY = -1. # Number of points lost each round
        MODIFIED_FOOD_REWARD = 10
        MODIFIED_EAT_ALL_FOOD_REWARD = 50
        MODIFIED_DIE_PENALTY = -20

        if reward == TIME_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        if reward == TIME_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD
        if reward == TIME_PENALTY + DIE_PENALTY:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_DIE_PENALTY
        if reward == TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_EAT_ALL_FOOD_REWARD + MODIFIED_FOOD_REWARD
        if reward == TIME_PENALTY + DIE_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD + MODIFIED_DIE_PENALTY

        if reward == TIME_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_PUDDLE_PENALTY
        if reward == TIME_PENALTY + DIE_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_DIE_PENALTY + MODIFIED_PUDDLE_PENALTY

        return reward / 10.0

    def add_to_replay_memory(self, state, action, reward, next_state, done):
        if abs(reward) > 100:
            done = 1

        self.replay_memory_buffer.append((self.extractor(state),
            ActionMapping.ActionToNumber[action], self.getModifiedReward(reward), \
            self.extractor(next_state), done))

    def computeActionFromQValues(self, state):
        # legalActions = self.getLegalActions(state)
        legalActions = ['North', 'South', 'East', 'West', 'Stop']
        if np.random.rand() < self.epsilon:
            return random.choice(legalActions)
        else:
            qValues = self.model.predict(np.array([self.extractor(state)]), batch_size=1)[0]
            bestAction = ActionMapping.NumbertoAction[np.argmax(qValues)]
            # maxQ, bestAction = float('-inf'), None
            # for action in legalActions:
            #     if qValues[ActionMapping.ActionToNumber[action]] > maxQ:
            #         maxQ, bestAction = qValues[ActionMapping.ActionToNumber[action]], action
            return bestAction

    def replayExperience(self):
        # replay_memory_buffer size check
        if len(self.replay_memory_buffer) < self.batch_size:
            return
        random_sample = self.get_random_sample_from_replay_mem()
        states, actions, rewards, next_states, done_list = self.get_attribues_from_sample(random_sample)
        targets = rewards + self.discount * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - done_list)
        target_vec = self.model.predict_on_batch(states)
        indexes = np.array([i for i in range(self.batch_size)])
        target_vec[[indexes], [actions]] = targets

        self.model.fit(states, target_vec, epochs=1, verbose=0)

    def saveModel(self, model, file_name):
        model_json = model.to_json()
        with open('weights/' + file_name + '.json', "w") as json_file:
            json_file.write(model_json)
        model.save_weights('weights/' + file_name + '.h5')


    def update(self, state, action, nextState, reward):
        #if self.selfTesting and self.currentTrainingEpisode > self.lastSavedWeights:
           # self.saveModel(self.model, 'DQNBaselineAgent_' + identifier + '_' + str(self.currentTrainingEpisode))
            #self.lastSavedWeights = self.currentTrainingEpisode

        if self.alpha < 0.0001:
            return
        done = 0
        if self.epsilon > self.min_epsilon:
            self.epsilon = self.epsilon * self.decay
        self.add_to_replay_memory(state, action, reward/10.0, nextState, done)
        self.replayExperience()

    def get_attribues_from_sample(self, random_sample):
        states = np.array([i[0] for i in random_sample])
        actions = np.array([i[1] for i in random_sample])
        rewards = np.array([i[2] for i in random_sample])
        next_states = np.array([i[3] for i in random_sample])
        done_list = np.array([i[4] for i in random_sample])
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        return np.squeeze(states), actions, rewards, next_states, done_list

    def get_random_sample_from_replay_mem(self):
        return random.sample(self.replay_memory_buffer, self.batch_size)

class GmQAgent(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', **args):
        PacmanQAgent.__init__(self, **args)
        self.nb_foodFeatures = 10
        self.nb_ghostFeatures = 9
        self.nb_puddleFeatures = 13
        self.nb_actions = 5
        self.epsilon = 1
        self.min_epsilon = 0.01
        self.decay = .999
        self.ghostAgent = DqnModule(nb_features = self.nb_ghostFeatures, featureExtractor = CustomizedExtractor().getGhostFeatures)
        self.foodAgent = DqnModule(nb_features = self.nb_foodFeatures, featureExtractor = CustomizedExtractor().getFoodFeatures)
        # self.puddleAgent = DqnModule(nb_features = self.nb_puddleFeatures, featureExtractor = CustomizedExtractor().getPuddleFeatures)
        # self.foodAgent.model = self.loadModel(name)
        # self.ghostAgent.model = self.loadModel(name)
        print '----------'
        print '############ GmQAgent ############'
        print 'Epsilon Decay = %s, Discount Factor = %.2f' % (self.decay, self.discount)
        print 'Feature Count: Ghost = %d, Food = %d' % (self.nb_ghostFeatures, self.nb_foodFeatures)
        print 'Rewards for foodAgent: Time Penalty = %.2f, (Food Reward + Time Penalty) = %.2f, \
        (Food Reward + Time Penalty + LastReward) = %.2f' % (self.getFoodReward(TIME_PENALTY), \
        self.getFoodReward(TIME_PENALTY + FOOD_REWARD), self.getFoodReward(TIME_PENALTY + FOOD_REWARD + EAT_ALL_FOOD_REWARD))
        print 'Rewards for ghostAgent: Time Penalty = %.2f, (Death Penalty) = %.2f' % (self.getGhostReward(TIME_PENALTY), \
        self.getGhostReward(TIME_PENALTY + DIE_PENALTY))
        print '----------'
        self.isSaved = 0
        self.lastSavedWeights = -1

    def computeActionFromQValues(self, state):
        # legalActions = self.getLegalActions(state)
        legalActions = ['North', 'South', 'East', 'West', 'Stop']
        if np.random.rand() < self.epsilon:
            return random.choice(legalActions)
        else:
            ghostQValues = self.ghostAgent.getQValues(state)
            foodQValues = self.foodAgent.getQValues(state)
            # puddleQValues = self.puddleAgent.getQValues(state)
            # if self.alpha < 0.0001:
            #     print 'g', ghostQValues
            #     print 'f', foodQValues
            qValues = foodQValues + ghostQValues
            # qValues = puddleQValues
            bestAction = ActionMapping.NumbertoAction[np.argmax(qValues)]
            # qValues = foodQValues
            maxQ, bestAction = float('-inf'), None
            for action in legalActions:
                if qValues[ActionMapping.ActionToNumber[action]] > maxQ:
                    maxQ, bestAction = qValues[ActionMapping.ActionToNumber[action]], action
            return bestAction

    def getPuddleReward(self, reward, state, nextState):
        # print reward

        # MODIFIED_PUDDLE_PENALTY = 0.8
        MODIFIED_PUDDLE_PENALTY = -1.
        MODIFIED_TIME_PENALTY = -1. # Number of points lost each round

        # dist1 = closestFood(state.getPacmanPosition(), state.getPuddles(), state.getWalls())
        # dist2 = closestFood(nextState.getPacmanPosition(), nextState.getPuddles(), nextState.getWalls())
        # del_dist = dist1 - dist2
        # new_PUDDLE_PENALTY = PUDDLE_PENALTY * dist
        # new_MODIFIED_PUDDLE_PENALTY = MODIFIED_PUDDLE_PENALTY * dist

        if reward == TIME_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY

        elif reward == TIME_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_PUDDLE_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_PUDDLE_PENALTY

        # print reward, MODIFIED_PUDDLE_PENALTY * del_dist
        # reward = MODIFIED_PUDDLE_PENALTY * del_dist

        return reward / 10.0

    def getFoodReward(self, reward):
        MODIFIED_TIME_PENALTY = -10. # Number of points lost each round
        MODIFIED_FOOD_REWARD = 10
        MODIFIED_EAT_ALL_FOOD_REWARD = 50
        # MODIFIED_PUDDLE_PENALTY = 10

        if reward == TIME_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD
        elif reward == TIME_PENALTY + DIE_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_EAT_ALL_FOOD_REWARD + MODIFIED_FOOD_REWARD
        elif reward == TIME_PENALTY + DIE_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD

        elif reward == TIME_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_TIME_PENALTY

        return reward / 10.0

    def getGhostReward(self, reward):
        MODIFIED_TIME_PENALTY = -1 # Number of points lost each round
        MODIFIED_DIE_PENALTY = -20

        if reward == TIME_PENALTY:
            reward = -MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + FOOD_REWARD:
            reward = -MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY:
            reward = MODIFIED_DIE_PENALTY
        elif reward == TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD:
            reward = -MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY + FOOD_REWARD:
            reward = MODIFIED_DIE_PENALTY

        elif reward == TIME_PENALTY + PUDDLE_PENALTY:
            reward = -MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_DIE_PENALTY

        return reward / 30.0

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

    def update(self, state, action, nextState, reward):
        if self.selfTesting and self.currentTrainingEpisode > self.lastSavedWeights:
            # self.saveModel(self.ghostAgent.model, 'ghostAgent_' + identifier + '_' + str(self.currentTrainingEpisode))
            # self.saveModel(self.foodAgent.model, 'foodAgent_' + identifier + '_' + str(self.currentTrainingEpisode))
            # self.saveModel(self.puddleAgent.model, 'puddleAgent_' + identifier + '_' + str(self.currentTrainingEpisode))
            self.lastSavedWeights = self.currentTrainingEpisode

        if self.alpha < 0.0001:
            return

        if self.epsilon > self.min_epsilon:
            self.epsilon = self.epsilon * self.decay

        done = 1 if abs(reward) > 100 else 0
        # if done:
        #     print '------------------------------------------------------------'

        self.ghostAgent.update(state, ActionMapping.ActionToNumber[action], nextState, self.getGhostReward(reward), done)
        self.foodAgent.update(state, ActionMapping.ActionToNumber[action], nextState, self.getFoodReward(reward), done)
        # self.puddleAgent.update(state, ActionMapping.ActionToNumber[action], nextState, self.getPuddleReward(reward, state, nextState), done)

class GmQ_Pre(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', **args):
        PacmanQAgent.__init__(self, **args)
        self.nb_foodFeatures = 10
        self.nb_ghostFeatures = 9
        self.nb_puddleFeatures = 13
        self.nb_actions = 5
        self.epsilon = 1
        self.min_epsilon = 0.01
        self.decay = .999
        self.ghostAgent = DqnModule(nb_features = self.nb_ghostFeatures, featureExtractor = CustomizedExtractor().getGhostFeatures)
        self.foodAgent = DqnModule(nb_features = self.nb_foodFeatures, featureExtractor = CustomizedExtractor().getFoodFeatures)
        # self.puddleAgent = DqnModule(nb_features = self.nb_puddleFeatures, featureExtractor = CustomizedExtractor().getPuddleFeatures)
        # self.foodAgent.model = self.loadModel(name)
        self.ghostAgent.model = self.loadModel('ghostAgent_ghostTimePenalty1_1_1100')

        print '----------'
        print '############ GmQAgent ############'
        print 'Epsilon Decay = %s, Discount Factor = %.2f' % (self.decay, self.discount)
        print 'Feature Count: Ghost = %d, Food = %d' % (self.nb_ghostFeatures, self.nb_foodFeatures)
        print 'Rewards for foodAgent: Time Penalty = %.2f, (Food Reward + Time Penalty) = %.2f, \
        (Food Reward + Time Penalty + LastReward) = %.2f' % (self.getFoodReward(TIME_PENALTY), \
        self.getFoodReward(TIME_PENALTY + FOOD_REWARD), self.getFoodReward(TIME_PENALTY + FOOD_REWARD + EAT_ALL_FOOD_REWARD))
        print 'Rewards for ghostAgent: Time Penalty = %.2f, (Death Penalty) = %.2f' % (self.getGhostReward(TIME_PENALTY), \
        self.getGhostReward(TIME_PENALTY + DIE_PENALTY))
        print '----------'
        self.isSaved = 0
        self.lastSavedWeights = -1

    def computeActionFromQValues(self, state):
        # legalActions = self.getLegalActions(state)
        legalActions = ['North', 'South', 'East', 'West', 'Stop']
        if np.random.rand() < self.epsilon:
            return random.choice(legalActions)
        else:
            ghostQValues = self.ghostAgent.getQValues(state)/3.0
            foodQValues = self.foodAgent.getQValues(state)
            # puddleQValues = self.puddleAgent.getQValues(state)
            # if self.alpha < 0.0001:
            #     print 'g', ghostQValues
            #     print 'f', foodQValues
            qValues = foodQValues + ghostQValues
            # qValues = puddleQValues
            bestAction = ActionMapping.NumbertoAction[np.argmax(qValues)]
            # qValues = foodQValues
            maxQ, bestAction = float('-inf'), None
            for action in legalActions:
                if qValues[ActionMapping.ActionToNumber[action]] > maxQ:
                    maxQ, bestAction = qValues[ActionMapping.ActionToNumber[action]], action
            return bestAction

    def getPuddleReward(self, reward, state, nextState):
        # print reward

        # MODIFIED_PUDDLE_PENALTY = 0.8
        MODIFIED_PUDDLE_PENALTY = -1.
        MODIFIED_TIME_PENALTY = -1. # Number of points lost each round

        # dist1 = closestFood(state.getPacmanPosition(), state.getPuddles(), state.getWalls())
        # dist2 = closestFood(nextState.getPacmanPosition(), nextState.getPuddles(), nextState.getWalls())
        # del_dist = dist1 - dist2
        # new_PUDDLE_PENALTY = PUDDLE_PENALTY * dist
        # new_MODIFIED_PUDDLE_PENALTY = MODIFIED_PUDDLE_PENALTY * dist

        if reward == TIME_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY

        elif reward == TIME_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_PUDDLE_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_PUDDLE_PENALTY

        # print reward, MODIFIED_PUDDLE_PENALTY * del_dist
        # reward = MODIFIED_PUDDLE_PENALTY * del_dist

        return reward / 10.0

    def getFoodReward(self, reward):
        MODIFIED_TIME_PENALTY = -10. # Number of points lost each round
        MODIFIED_FOOD_REWARD = 10
        MODIFIED_EAT_ALL_FOOD_REWARD = 50
        # MODIFIED_PUDDLE_PENALTY = 10

        if reward == TIME_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD
        elif reward == TIME_PENALTY + DIE_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_EAT_ALL_FOOD_REWARD + MODIFIED_FOOD_REWARD
        elif reward == TIME_PENALTY + DIE_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD

        elif reward == TIME_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_TIME_PENALTY

        return reward / 10.0

    def getGhostReward(self, reward):
        MODIFIED_TIME_PENALTY = -1 # Number of points lost each round
        MODIFIED_DIE_PENALTY = -20

        if reward == TIME_PENALTY:
            reward = -MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + FOOD_REWARD:
            reward = -MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY:
            reward = MODIFIED_DIE_PENALTY
        elif reward == TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD:
            reward = -MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY + FOOD_REWARD:
            reward = MODIFIED_DIE_PENALTY

        elif reward == TIME_PENALTY + PUDDLE_PENALTY:
            reward = -MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_DIE_PENALTY

        return reward / 30.0

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

    def update(self, state, action, nextState, reward):
        if self.selfTesting and self.currentTrainingEpisode > self.lastSavedWeights:
            # self.saveModel(self.ghostAgent.model, 'ghostAgent_' + identifier + '_' + str(self.currentTrainingEpisode))
            # self.saveModel(self.foodAgent.model, 'foodAgent_' + identifier + '_' + str(self.currentTrainingEpisode))
            # self.saveModel(self.puddleAgent.model, 'puddleAgent_' + identifier + '_' + str(self.currentTrainingEpisode))
            self.lastSavedWeights = self.currentTrainingEpisode

        if self.alpha < 0.0001:
            return

        if self.epsilon > self.min_epsilon:
            self.epsilon = self.epsilon * self.decay

        done = 1 if abs(reward) > 100 else 0
        # if done:
        #     print '------------------------------------------------------------'

        # self.ghostAgent.update(state, ActionMapping.ActionToNumber[action], nextState, self.getGhostReward(reward), done)
        self.foodAgent.update(state, ActionMapping.ActionToNumber[action], nextState, self.getFoodReward(reward), done)
        # self.puddleAgent.update(state, ActionMapping.ActionToNumber[action], nextState, self.getPuddleReward(reward, state, nextState), done)


class DqnModule():
    '''
        This class only deals with numerical actions
    '''
    def __init__(self,  nb_features, featureExtractor, batch_size = 32, start_epsilon = 1, min_epsilon = 0.01, decay = 0.9995, discount = 0.8, nb_actions = 5):
        self.batch_size = batch_size
        self.epsilon = start_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.discount = discount
        self.model = NeuralNetwork(input_size = nb_features, nb_actions = nb_actions).model
        self.replay_memory_buffer = deque(maxlen=50000)
        self.extractor = featureExtractor
        print '----------'
        print '### DqnModule ###'
        print 'Epsilon Decay = %s, Discount Factor = %.2f' % (self.decay, self.discount)
        print 'Input Features = %d' % (nb_features)
        print '----------'


    def getQValue(self, state, action):
        qValues = self.model.predict(self.extractor(state))
        return qValues[action]

    def getAction(self, state, legalActions):
        qValues = self.model.predict(np.array([self.extractor(state)]), batch_size=1)[0]
        maxQ, bestAction = float('-inf'), None
        for action in legalActions:
            if qValues[action] > maxQ:
                maxQ, bestAction = qValues[action], action
        return bestAction

    def getQValues(self, state):
        return self.model.predict(np.array([self.extractor(state)]), batch_size=1)[0]

    def update(self, state, action, nextState, reward, done):
        self.add_to_replay_memory(state, action, reward, nextState, done)
        self.replayExperience()

    def add_to_replay_memory(self, state, action, reward, next_state, done):
        self.replay_memory_buffer.append((self.extractor(state),
            action, reward, self.extractor(next_state), done))

    def replayExperience(self):
        # replay_memory_buffer size check
        if len(self.replay_memory_buffer) < self.batch_size:
            return
        random_sample = self.get_random_sample_from_replay_mem()
        indexes = np.array([i for i in range(self.batch_size)])
        states, actions, rewards, next_states, done_list = self.get_attribues_from_sample(random_sample)
        targets = rewards + self.discount * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - done_list)
        target_vec = self.model.predict_on_batch(states)
        target_vec[[indexes], [actions]] = targets

        self.model.fit(states, target_vec, epochs=1, verbose=0)

    def get_random_sample_from_replay_mem(self):
        return random.sample(self.replay_memory_buffer, self.batch_size)

    def get_attribues_from_sample(self, random_sample):
        states = np.array([i[0] for i in random_sample])
        actions = np.array([i[1] for i in random_sample])
        rewards = np.array([i[2] for i in random_sample])
        next_states = np.array([i[3] for i in random_sample])
        done_list = np.array([i[4] for i in random_sample])
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        return np.squeeze(states), actions, rewards, next_states, done_list

class GmSarsaAgent(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', **args):
        PacmanQAgent.__init__(self, **args)
        self.nb_foodFeatures = 10
        self.nb_ghostFeatures = 8
        self.nb_actions = 5
        self.epsilon = 1
        self.min_epsilon = 0.01
        self.decay = .999
        self.ghostAgent = SarsaModule(nb_features = self.nb_ghostFeatures, featureExtractor = CustomizedExtractor().getGhostFeatures)
        self.foodAgent = SarsaModule(nb_features = self.nb_foodFeatures, featureExtractor = CustomizedExtractor().getFoodFeatures)
        self.resetLastState()
        # self.foodAgent.model = self.loadModel(name)
        # self.ghostAgent.model = self.loadModel(name)
        self.isSaved = 0
        self.lastSavedWeights = -1
        print '----------'
        print '############ GmSarsaAgent ############'
        print 'Epsilon Decay = %.2f, Discount Factor = %.2f' % (self.decay, self.discount)
        print 'Feature Count: Ghost = %d, Food = %d' % (self.nb_ghostFeatures, self.nb_foodFeatures)
        print 'Rewards for foodAgent: Time Penalty = %.2f, (Food Reward + Time Penalty) = %.2f, \
        (Food Reward + Time Penalty + LastReward) = %.2f' % (self.getFoodReward(TIME_PENALTY), \
        self.getFoodReward(TIME_PENALTY + FOOD_REWARD), self.getFoodReward(TIME_PENALTY + FOOD_REWARD + EAT_ALL_FOOD_REWARD))
        print 'Rewards for ghostAgent: Time Penalty = %.2f, (Death Penalty) = %.2f' % (self.getGhostReward(TIME_PENALTY), \
        self.getGhostReward(TIME_PENALTY + DIE_PENALTY))
        print '----------'


    def resetLastState(self):
        self.prevState = None
        self.prevAction = None
        self.prevReward = None

    def computeActionFromQValues(self, state):
        legalActions = self.getLegalActions(state)
        if np.random.rand() < self.epsilon:
            return random.choice(legalActions)
        else:
            ghostQValues = self.ghostAgent.getQValues(state)
            foodQValues = self.foodAgent.getQValues(state)
            qValues = foodQValues + ghostQValues
            maxQ, bestAction = float('-inf'), None
            for action in legalActions:
                if qValues[ActionMapping.ActionToNumber[action]] > maxQ:
                    maxQ, bestAction = qValues[ActionMapping.ActionToNumber[action]], action
            return bestAction

    def getFoodReward(self, reward):
        MODIFIED_TIME_PENALTY = -.1 # Number of points lost each round
        MODIFIED_FOOD_REWARD = 10
        MODIFIED_EAT_ALL_FOOD_REWARD = 50

        if reward == TIME_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        if reward == TIME_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD
        if reward == TIME_PENALTY + DIE_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        if reward == TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_EAT_ALL_FOOD_REWARD + MODIFIED_FOOD_REWARD
        if reward == TIME_PENALTY + DIE_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD
        # print reward
        return reward / 10.0

    def getGhostReward(self, reward):
        MODIFIED_TIME_PENALTY = -.1 # Number of points lost each round
        MODIFIED_DIE_PENALTY = -20

        if reward == TIME_PENALTY:
            reward = -MODIFIED_TIME_PENALTY
        if reward == TIME_PENALTY + FOOD_REWARD:
            reward = -MODIFIED_TIME_PENALTY
        if reward == TIME_PENALTY + DIE_PENALTY:
            reward = MODIFIED_DIE_PENALTY
        if reward == TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD:
            reward = -MODIFIED_TIME_PENALTY
        if reward == TIME_PENALTY + DIE_PENALTY + FOOD_REWARD:
            reward = MODIFIED_DIE_PENALTY

        return reward / 10.0

    def update(self, state, action, nextState, reward):
        if self.selfTesting and self.currentTrainingEpisode > self.lastSavedWeights:
            self.saveModel(self.ghostAgent.model, 'ghostAgent_' + identifier + '_' + str(self.currentTrainingEpisode))
            self.saveModel(self.foodAgent.model, 'foodAgent_' + identifier + '_' + str(self.currentTrainingEpisode))
            self.lastSavedWeights = self.currentTrainingEpisode

        if self.alpha < 0.0001:
            return
        if self.epsilon > self.min_epsilon:
            self.epsilon = self.epsilon * self.decay

        if (self.prevReward != None):
            self.ghostAgent.update(self.prevState, self.prevAction, self.getGhostReward(self.prevReward), state, action, 0)
            self.foodAgent.update(self.prevState, self.prevAction, self.getFoodReward(self.prevReward), state, action, 0)

        done = 1 if abs(reward) > 100 else 0

        self.prevState = state
        self.prevAction = action
        self.prevReward = reward

        if done:
            # state, action -- can be anything (here it is same as prevState and prevAction respectively)
            self.ghostAgent.update(self.prevState, self.prevAction, self.getGhostReward(self.prevReward), state, action, 1)
            self.foodAgent.update(self.prevState, self.prevAction, self.getFoodReward(self.prevReward), state, action, 1)
            self.resetLastState()

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

class SarsaModule():
    def __init__(self,  nb_features, featureExtractor, batch_size = 32, start_epsilon = 1, min_epsilon = 0.01, decay = 0.9995, discount = 0.8, nb_actions = 5):
        self.batch_size = batch_size
        self.epsilon = start_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.discount = discount
        self.model = NeuralNetwork(input_size = nb_features, nb_actions = nb_actions).model
        self.replay_memory_buffer = deque(maxlen=50000)
        # NOTE TODO CHANGE IT
        self.extractor = featureExtractor
        print '----------'
        print '### SarsaModule ###'
        print 'Epsilon Decay = %s, Discount Factor = %.2f' % (self.decay, self.discount)
        print 'Input Features = %d' % (nb_features)
        print '----------'


    def getQValue(self, state, action):
        qValues = self.model.predict(self.extractor(state))
        return qValues[ActionMapping.ActionToNumber[action]]

    def getQValues(self, state):
        return self.model.predict(np.array([self.extractor(state)]), batch_size=1)[0]

    def update(self, state, action, reward, nextState, nextAction, done):
        self.add_to_replay_memory(state, action, reward, nextState, nextAction, done)
        self.replayExperience()

    def add_to_replay_memory(self, state, action, reward, nextState, nextAction, done):
        self.replay_memory_buffer.append((self.extractor(state),
            ActionMapping.ActionToNumber[action], reward, self.extractor(nextState), ActionMapping.ActionToNumber[nextAction], done))

    def replayExperience(self):
        # replay_memory_buffer size check
        if len(self.replay_memory_buffer) < self.batch_size:
            return
        random_sample = self.get_random_sample_from_replay_mem()
        indexes = np.array([i for i in range(self.batch_size)])
        states, actions, rewards, next_states, next_actions, done_list = self.get_attribues_from_sample(random_sample)
        targets = rewards + self.discount * (self.model.predict_on_batch(next_states)[0, next_actions]) * (1 - done_list)
        target_vec = self.model.predict_on_batch(states)
        target_vec[[indexes], [actions]] = targets

        self.model.fit(states, target_vec, epochs=1, verbose=0)

    def get_random_sample_from_replay_mem(self):
        return random.sample(self.replay_memory_buffer, self.batch_size)

    def get_attribues_from_sample(self, random_sample):
        states = np.array([i[0] for i in random_sample])
        actions = np.array([i[1] for i in random_sample])
        rewards = np.array([i[2] for i in random_sample])
        next_states = np.array([i[3] for i in random_sample])
        next_actions = np.array([i[4] for i in random_sample])
        done_list = np.array([i[5] for i in random_sample])
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        return np.squeeze(states), actions, rewards, next_states, next_actions, done_list

class HierarchicalQAgent(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', **args):
        PacmanQAgent.__init__(self, **args)
        self.nb_foodFeatures = 10
        self.nb_ghostFeatures = 9
        self.nb_actions = 5
        self.nb_features = 13
        self.epsilon = 1
        self.min_epsilon = 0.01
        self.decay = .999
        self.arbitratorDecay = .9995
        self.arbitratorEpsilon = 1
        self.ghostAgent = DqnModule(nb_features = self.nb_ghostFeatures, featureExtractor = CustomizedExtractor().getGhostFeatures)
        self.foodAgent = DqnModule(nb_features = self.nb_foodFeatures, featureExtractor = CustomizedExtractor().getFoodFeatures)
        self.arbitrator = DqnModule(nb_features = self.nb_features, featureExtractor = CustomizedExtractor().getFeatures, nb_actions = 2)
        self.subModules = [self.ghostAgent, self.foodAgent]
        self.lastSavedWeights = -1
        # self.foodAgent.model = self.loadModel(name)
        # self.ghostAgent.model = self.loadModel(name)
        self.isSaved = 0
        print '----------'
        print '############ HierarchicalQAgent ############'
        print 'Epsilon Decay = %f, Arbitrator Epsilon Decay = %f, Discount Factor = %.2f' % (self.decay, self.arbitratorDecay, self.discount)
        print 'Feature Count: Arbitrator = %d, Ghost = %d, Food = %d' % (self.nb_features, self.nb_ghostFeatures, self.nb_foodFeatures)
        print 'Rewards for Arbitrator: (Eat ghost) = %.2f, (Eat Food) = %.2f, (Death Penalty) = %.2f, (Time Penalty) = %.2f' % \
        (self.getArbitratorReward(50), self.getArbitratorReward(10), self.getArbitratorReward(-500), self.getArbitratorReward(-1))
        print 'Rewards for foodAgent: Time Penalty = %.2f, (Food Reward + Time Penalty) = %.2f, \
        (Food Reward + Time Penalty + LastReward) = %.2f' % (self.getFoodReward(TIME_PENALTY), \
        self.getFoodReward(TIME_PENALTY + FOOD_REWARD), self.getFoodReward(TIME_PENALTY + FOOD_REWARD + EAT_ALL_FOOD_REWARD))
        print 'Rewards for ghostAgent: Time Penalty = %.2f, (Death Penalty) = %.2f' % (self.getGhostReward(TIME_PENALTY), \
        self.getGhostReward(TIME_PENALTY + DIE_PENALTY))
        print '----------'


    def computeActionFromQValues(self, state):
        # legalActions = self.getLegalActions(state)
        if np.random.rand() < self.arbitratorEpsilon:
            self.arbitratorAction = random.randrange(2)
        else:
            self.arbitratorAction = self.arbitrator.getAction(state, [0, 1])

        legalActions = [ActionMapping.ActionToNumber[action] for action in self.getLegalActions(state)]

        action = self.subModules[self.arbitratorAction].getAction(state, legalActions)


        return ActionMapping.NumbertoAction[action]

    def getFoodReward(self, reward):
        MODIFIED_TIME_PENALTY = -10. # Number of points lost each round
        MODIFIED_FOOD_REWARD = 10
        MODIFIED_EAT_ALL_FOOD_REWARD = 50
        # MODIFIED_PUDDLE_PENALTY = 10

        if reward == TIME_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD
        elif reward == TIME_PENALTY + DIE_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_EAT_ALL_FOOD_REWARD + MODIFIED_FOOD_REWARD
        elif reward == TIME_PENALTY + DIE_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD

        elif reward == TIME_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_TIME_PENALTY

        return reward / 10.0

    def getGhostReward(self, reward):
        MODIFIED_TIME_PENALTY = -1 # Number of points lost each round
        MODIFIED_DIE_PENALTY = -20

        if reward == TIME_PENALTY:
            reward = -MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + FOOD_REWARD:
            reward = -MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY:
            reward = MODIFIED_DIE_PENALTY
        elif reward == TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD:
            reward = -MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY + FOOD_REWARD:
            reward = MODIFIED_DIE_PENALTY

        elif reward == TIME_PENALTY + PUDDLE_PENALTY:
            reward = -MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_DIE_PENALTY

        return reward / 10.0

    def getArbitratorReward(self, reward):

        MODIFIED_TIME_PENALTY = -10. # Number of points lost each round
        MODIFIED_FOOD_REWARD = 10
        MODIFIED_EAT_ALL_FOOD_REWARD = 50
        MODIFIED_DIE_PENALTY = -20

        if reward == TIME_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        if reward == TIME_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD
        if reward == TIME_PENALTY + DIE_PENALTY:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_DIE_PENALTY
        if reward == TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_EAT_ALL_FOOD_REWARD + MODIFIED_FOOD_REWARD
        if reward == TIME_PENALTY + DIE_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD + MODIFIED_DIE_PENALTY

        return reward / 10.0

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

    def update(self, state, action, nextState, reward):
        if self.selfTesting and self.currentTrainingEpisode > self.lastSavedWeights:
            self.saveModel(self.ghostAgent.model, 'ghostAgent_' + identifier + '_' + str(self.currentTrainingEpisode))
            self.saveModel(self.foodAgent.model, 'foodAgent_' + identifier + '_' + str(self.currentTrainingEpisode))
            self.saveModel(self.arbitrator.model, 'arbitrator_' + identifier + '_' + str(self.currentTrainingEpisode))
            self.lastSavedWeights = self.currentTrainingEpisode
        if self.alpha < 0.0001:
            return
        if self.epsilon > self.min_epsilon:
            self.epsilon = self.epsilon * self.decay

        if self.arbitratorEpsilon > self.min_epsilon:
            self.arbitratorEpsilon = self.arbitratorEpsilon * self.arbitratorDecay

        done = 1 if abs(reward) > 100 else 0
        # if done:
        #     print '------------------------------------------------------------'

        self.arbitrator.update(state, self.arbitratorAction, nextState, self.getArbitratorReward(reward), done)
        self.ghostAgent.update(state, ActionMapping.ActionToNumber[action], nextState, self.getGhostReward(reward), done)
        self.foodAgent.update(state, ActionMapping.ActionToNumber[action], nextState, self.getFoodReward(reward), done)

class PreTrainedHierarchicalQAgent(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', **args):
        PacmanQAgent.__init__(self, **args)
        self.nb_foodFeatures = 10
        self.nb_ghostFeatures = 9
        self.nb_actions = 5
        self.nb_features = 13
        self.epsilon = 1
        self.decay = .999
        self.min_epsilon = 0.01
        self.arbitratorDecay = .999
        self.arbitratorEpsilon = 1
        self.ghostAgent = DqnModule(nb_features = self.nb_ghostFeatures, featureExtractor = CustomizedExtractor().getGhostFeatures)
        self.foodAgent = DqnModule(nb_features = self.nb_foodFeatures, featureExtractor = CustomizedExtractor().getFoodFeatures)
        self.arbitrator = DqnModule(nb_features = self.nb_features, featureExtractor = CustomizedExtractor().getFeatures, nb_actions = 2)
        self.subModules = [self.ghostAgent, self.foodAgent]
        self.lastSavedWeights = -1
        self.ghostAgent.model = self.loadModel('ghostAgent_ghostTimePenalty1_1_1100')
        self.foodAgent.model = self.loadModel('foodAgent_ghostTimePenalty1_1_1100')
        self.isSaved = 0
        print '----------'
        print '############ HierarchicalQAgent ############'
        print 'Epsilon Decay = %f, Arbitrator Epsilon Decay = %f, Discount Factor = %.2f' % (self.decay, self.arbitratorDecay, self.discount)
        print 'Feature Count: Arbitrator = %d, Ghost = %d, Food = %d' % (self.nb_features, self.nb_ghostFeatures, self.nb_foodFeatures)
        print 'Rewards for Arbitrator: (Eat ghost) = %.2f, (Eat Food) = %.2f, (Death Penalty) = %.2f, (Time Penalty) = %.2f' % \
        (self.getArbitratorReward(50), self.getArbitratorReward(10), self.getArbitratorReward(-500), self.getArbitratorReward(-1))
        # print 'Rewards for foodAgent: Time Penalty = %.2f, (Food Reward + Time Penalty) = %.2f, \
        # (Food Reward + Time Penalty + LastReward) = %.2f' % (self.getFoodReward(TIME_PENALTY), \
        # self.getFoodReward(TIME_PENALTY + FOOD_REWARD), self.getFoodReward(TIME_PENALTY + FOOD_REWARD + EAT_ALL_FOOD_REWARD))
        # print 'Rewards for ghostAgent: Time Penalty = %.2f, (Death Penalty) = %.2f' % (self.getGhostReward(TIME_PENALTY), \
        # self.getGhostReward(TIME_PENALTY + DIE_PENALTY))
        print '----------'


    def computeActionFromQValues(self, state):
        # legalActions = self.getLegalActions(state)
        if np.random.rand() < self.arbitratorEpsilon:
            self.arbitratorAction = random.randrange(2)
        else:
            self.arbitratorAction = self.arbitrator.getAction(state, [0, 1])

        legalActions = [ActionMapping.ActionToNumber[action] for action in self.getLegalActions(state)]

        action = self.subModules[self.arbitratorAction].getAction(state, legalActions)


        return ActionMapping.NumbertoAction[action]


    def getArbitratorReward(self, reward):

        MODIFIED_TIME_PENALTY = -10. # Number of points lost each round
        MODIFIED_FOOD_REWARD = 10
        MODIFIED_EAT_ALL_FOOD_REWARD = 50
        MODIFIED_DIE_PENALTY = -20

        if reward == TIME_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        if reward == TIME_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD
        if reward == TIME_PENALTY + DIE_PENALTY:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_DIE_PENALTY
        if reward == TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_EAT_ALL_FOOD_REWARD + MODIFIED_FOOD_REWARD
        if reward == TIME_PENALTY + DIE_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD + MODIFIED_DIE_PENALTY

        return reward / 10.0

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

    def update(self, state, action, nextState, reward):
        if self.selfTesting and self.currentTrainingEpisode > self.lastSavedWeights:
            # self.saveModel(self.ghostAgent.model, 'ghostAgent_' + identifier + '_' + str(self.currentTrainingEpisode))
            # self.saveModel(self.foodAgent.model, 'foodAgent_' + identifier + '_' + str(self.currentTrainingEpisode))
            self.saveModel(self.arbitrator.model, 'arbitrator_' + identifier + '_' + str(self.currentTrainingEpisode))
            self.lastSavedWeights = self.currentTrainingEpisode
        if self.alpha < 0.0001:
            return
        if self.epsilon > self.min_epsilon:
            self.epsilon = self.epsilon * self.decay

        if self.arbitratorEpsilon > self.min_epsilon:
            self.arbitratorEpsilon = self.arbitratorEpsilon * self.arbitratorDecay

        done = 1 if abs(reward) > 100 else 0
        # if done:
        #     print '------------------------------------------------------------'

        self.arbitrator.update(state, self.arbitratorAction, nextState, self.getArbitratorReward(reward), done)
        # self.ghostAgent.update(state, ActionMapping.ActionToNumber[action], nextState, self.getGhostReward(reward), done)
        # self.foodAgent.update(state, ActionMapping.ActionToNumber[action], nextState, self.getFoodReward(reward), done)


class SequentialHierarchicalAgent(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', **args):
        PacmanQAgent.__init__(self, **args)
        self.nb_foodFeatures = 10
        self.nb_ghostFeatures = 8
        self.nb_actions = 5
        self.nb_features = 3
        self.epsilon = 1
        self.min_epsilon = 0.01
        self.decay = .999
        self.arbitratorDecay = .9995
        self.arbitratorEpsilon = 1
        self.ghostAgent = DqnModule(nb_features = self.nb_ghostFeatures, featureExtractor = CustomizedExtractor().getGhostFeatures)
        self.foodAgent = DqnModule(nb_features = self.nb_foodFeatures, featureExtractor = CustomizedExtractor().getFoodFeatures)
        self.arbitrator = DqnModule(nb_features = self.nb_features, featureExtractor = CustomizedExtractor().getFeatures, nb_actions = 2)
        self.subModules = [self.ghostAgent, self.foodAgent]
        self.lastSavedWeights = -1
        self.ghostAgent.model = self.loadModel('ghostAgent_DQNBaselineAgent_allActions_1_3500')
        self.foodAgent.model = self.loadModel('foodAgent_DQNBaselineAgent_allActions_1_3500')
        self.isSaved = 0

        print '----------'
        print '############ SequentialHierarchicalAgent ############'
        print 'Epsilon Decay = %f, Arbitrator Epsilon Decay = %f, Discount Factor = %.2f' % (self.decay, self.arbitratorDecay, self.discount)
        print 'Feature Count: Arbitrator = %d, Ghost = %d, Food = %d' % (self.nb_features, self.nb_ghostFeatures, self.nb_foodFeatures)
        print 'Rewards for Arbitrator: (Time Penalty) = %.2f, (Eat Food) = %.2f, (Die) = %.2f,  (Eat All Food) = %.2f' % \
        (self.getArbitratorReward(TIME_PENALTY), self.getArbitratorReward(TIME_PENALTY + FOOD_REWARD), self.getArbitratorReward(TIME_PENALTY + DIE_PENALTY), self.getArbitratorReward(TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD))
        print '----------'


    def computeActionFromQValues(self, state):
        # legalActions = self.getLegalActions(state)
        allActions = ['North', 'South', 'East', 'West', 'Stop']
        legalActions = [ActionMapping.ActionToNumber[action] for action in allActions]

        if np.random.rand() < self.arbitratorEpsilon:
            self.arbitratorAction = random.randrange(2)
        else:
            self.arbitratorAction = self.arbitrator.getAction(state, [0, 1])

        action = self.subModules[self.arbitratorAction].getAction(state, legalActions)

        return ActionMapping.NumbertoAction[action]

    def getArbitratorReward(self, reward):

        MODIFIED_TIME_PENALTY = -10. # Number of points lost each round
        MODIFIED_FOOD_REWARD = 10
        MODIFIED_EAT_ALL_FOOD_REWARD = 50
        MODIFIED_DIE_PENALTY = -20

        if reward == TIME_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        if reward == TIME_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD
        if reward == TIME_PENALTY + DIE_PENALTY:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_DIE_PENALTY
        if reward == TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_EAT_ALL_FOOD_REWARD + MODIFIED_FOOD_REWARD
        if reward == TIME_PENALTY + DIE_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD + MODIFIED_DIE_PENALTY

        return reward / 10.0

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

    def update(self, state, action, nextState, reward):
        if self.selfTesting and self.currentTrainingEpisode > self.lastSavedWeights:
            self.saveModel(self.arbitrator.model, 'arbitrator_' + identifier + '_' + str(self.currentTrainingEpisode))
            self.lastSavedWeights = self.currentTrainingEpisode
        if self.alpha < 0.0001:
            return

        if self.arbitratorEpsilon > self.min_epsilon:
            self.arbitratorEpsilon = self.arbitratorEpsilon * self.arbitratorDecay

        done = 1 if abs(reward) > 100 else 0
        # if done:
        #     print '------------------------------------------------------------'

        self.arbitrator.update(state, self.arbitratorAction, nextState, self.getArbitratorReward(reward), done)

class DDPGAgentPuddle(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', **args):
        PacmanQAgent.__init__(self, **args)
        self.nb_foodFeatures = 10
        self.nb_ghostFeatures = 9
        self.nb_puddleFeatures = 17
        self.nb_actions = 5
        self.nb_features = 26
        self.arbitrator_actions = 3
        self.epsilonDecay = .9995

        self.ghostAgent = DqnModule(nb_features = self.nb_ghostFeatures, featureExtractor = CustomizedExtractor().getGhostFeatures)
        self.foodAgent = DqnModule(nb_features = self.nb_foodFeatures, featureExtractor = CustomizedExtractor().getFoodFeatures)
        self.puddleAgent = DqnModule(nb_features = self.nb_puddleFeatures, featureExtractor = CustomizedExtractor().getPuddleFeatures)
        self.arbitrator = DDPGModule(nb_features = self.nb_features, featureExtractor = CustomizedExtractor().getFeatures, nb_actions = self.arbitrator_actions, decay = self.epsilonDecay)
        self.subModules = [self.ghostAgent, self.foodAgent]
        self.lastSavedWeights = -1
        self.ghostAgent.model = self.loadModel('ghostAgent_ghostTimePenalty1_1_1100')
        self.foodAgent.model = self.loadModel('foodAgent_ghostTimePenalty1_1_1100')
        self.puddleAgent.model = self.loadModel('puddleAgent__50')
        self.isSaved = 0

        print '----------'
        print '############ ActorCriticAgent ############'
        print 'Arbitrator Epsilon Decay = %f, Discount Factor = %.2f' % (self.arbitrator.decay, self.discount)
        print 'Feature Count: Arbitrator = %d, Ghost = %d, Food = %d' % (self.nb_features, self.nb_ghostFeatures, self.nb_foodFeatures)
        print 'Rewards for Arbitrator: (Time Penalty) = %.2f, (Eat Food) = %.2f, (Die) = %.2f,  (Eat All Food) = %.2f' % \
        (self.getArbitratorReward(TIME_PENALTY), self.getArbitratorReward(TIME_PENALTY + FOOD_REWARD), self.getArbitratorReward(TIME_PENALTY + DIE_PENALTY), self.getArbitratorReward(TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD))
        print '----------'


    def computeActionFromQValues(self, state):
        self.arbitratorAction = self.arbitrator.getAction(state)[0]
        scaleParameters = self.arbitratorAction

        # if self.currentTrainingEpisode > 300:
        # print state
        # print 'action = ', scaleParameter

        puddleQValues = self.puddleAgent.getQValues(state)
        ghostQValues = self.ghostAgent.getQValues(state)
        foodQValues = self.foodAgent.getQValues(state)
        # scalarizedQValues = scaleParameters[0] * ghostQValues/3 + scaleParameters[1] * foodQValues
        scalarizedQValues = scaleParameters[0] * ghostQValues + scaleParameters[1] * foodQValues + scaleParameters[2] * puddleQValues
        # scalarizedQValues = scaleParameter * ghostQValues + (1 - scaleParameter) * (foodQValues)
        bestAction = ActionMapping.NumbertoAction[np.argmax(scalarizedQValues)]
        return bestAction

    def getArbitratorReward(self, reward):

        MODIFIED_PUDDLE_PENALTY = .8
        MODIFIED_TIME_PENALTY = -1. # Number of points lost each round
        MODIFIED_FOOD_REWARD = 10
        MODIFIED_EAT_ALL_FOOD_REWARD = 0
        MODIFIED_DIE_PENALTY = -20

        if reward == TIME_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD
        elif reward == TIME_PENALTY + DIE_PENALTY:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_DIE_PENALTY
        elif reward == TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_EAT_ALL_FOOD_REWARD + MODIFIED_FOOD_REWARD
        elif reward == TIME_PENALTY + DIE_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD + MODIFIED_DIE_PENALTY

        elif reward == TIME_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_PUDDLE_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_DIE_PENALTY + MODIFIED_PUDDLE_PENALTY

        return reward / 10.0

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

    def update(self, state, action, nextState, reward):
        if self.selfTesting and self.currentTrainingEpisode > self.lastSavedWeights:
            self.saveModel(self.arbitrator.actor_model, 'actor_' + identifier + '_' + str(self.currentTrainingEpisode))
            # self.saveModel(self.arbitrator.model, 'arbitrator_' + identifier + '_' + str(self.currentTrainingEpisode))
            self.lastSavedWeights = self.currentTrainingEpisode
        if self.alpha < 0.0001:
            return

        done = 1 if abs(reward) > 100 else 0
        # if done:
        #     self.arbitrator.random_process.reset_states()
        #     print '------------------------------------------------------------'

        self.arbitrator.update(state, self.arbitratorAction, nextState, self.getArbitratorReward(reward), done)

class DDPGAgent(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', **args):
        PacmanQAgent.__init__(self, **args)
        self.nb_foodFeatures = 10
        self.nb_ghostFeatures = 9
        self.nb_actions = 5
        self.arbitrator_actions = 2
        self.nb_features = 13
        self.epsilonDecay = .995

        self.ghostAgent = DqnModule(nb_features = self.nb_ghostFeatures, featureExtractor = CustomizedExtractor().getGhostFeatures)
        # self.ghostAgent2 = DqnModule(nb_features = self.nb_ghostFeatures, featureExtractor = CustomizedExtractor().getGhostFeatures2)
        self.foodAgent = DqnModule(nb_features = self.nb_foodFeatures, featureExtractor = CustomizedExtractor().getFoodFeatures)
        self.arbitrator = DDPGModule(nb_features = self.nb_features, featureExtractor = CustomizedExtractor().getFeatures, nb_actions = self.arbitrator_actions, decay = self.epsilonDecay)
        self.subModules = [self.ghostAgent, self.foodAgent]
        self.lastSavedWeights = -1
        self.ghostAgent.model = self.loadModel('ghostAgent_ghostTimePenalty1_1_1100')
        # self.ghostAgent2.model = self.loadModel('ghostAgent_ghostTimePenalty1_1_1100')
        self.foodAgent.model = self.loadModel('foodAgent_ghostTimePenalty1_1_1100')
        self.isSaved = 0

        print '----------'
        print '############ ActorCriticAgent ############'
        print 'Arbitrator Epsilon Decay = %f, Discount Factor = %.2f' % (self.arbitrator.decay, self.discount)
        print 'Feature Count: Arbitrator = %d, Ghost = %d, Food = %d' % (self.nb_features, self.nb_ghostFeatures, self.nb_foodFeatures)
        print 'Rewards for Arbitrator: (Time Penalty) = %.2f, (Eat Food) = %.2f, (Die) = %.2f,  (Eat All Food) = %.2f' % \
        (self.getArbitratorReward(TIME_PENALTY), self.getArbitratorReward(TIME_PENALTY + FOOD_REWARD), self.getArbitratorReward(TIME_PENALTY + DIE_PENALTY), self.getArbitratorReward(TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD))
        print '----------'

    def computeActionFromQValues(self, state):
        self.arbitratorAction = self.arbitrator.getAction(state)[0]
        scaleParameters = self.arbitratorAction

        ghostQValues = self.ghostAgent.getQValues(state)
        # ghost2QValues = self.ghostAgent2.getQValues(state)
        foodQValues = self.foodAgent.getQValues(state)
        scalarizedQValues = scaleParameters[0] * (ghostQValues / 3) + scaleParameters[1] * foodQValues
        # scalarizedQValues = scaleParameter * ghostQValues + (1 - scaleParameter) * (foodQValues)

        bestAction = ActionMapping.NumbertoAction[np.argmax(scalarizedQValues)]
        return bestAction

    def getArbitratorReward(self, reward):

        MODIFIED_TIME_PENALTY = -1. # Number of points lost each round
        MODIFIED_FOOD_REWARD = 10
        MODIFIED_EAT_ALL_FOOD_REWARD = 50
        MODIFIED_DIE_PENALTY = -20

        if reward == TIME_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD
        elif reward == TIME_PENALTY + DIE_PENALTY:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_DIE_PENALTY
        elif reward == TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_EAT_ALL_FOOD_REWARD + MODIFIED_FOOD_REWARD
        elif reward == TIME_PENALTY + DIE_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD + MODIFIED_DIE_PENALTY

        return reward / 10.0

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

    def update(self, state, action, nextState, reward):
        # if self.selfTesting and self.currentTrainingEpisode > self.lastSavedWeights:
        #     self.saveModel(self.arbitrator.model, 'arbitrator_' + identifier + '_' + str(self.currentTrainingEpisode))
            # self.lastSavedWeights = self.currentTrainingEpisode
        if self.alpha < 0.0001:
            return

        done = 1 if abs(reward) > 100 else 0
        # if done:
        #     print '------------------------------------------------------------'

        self.arbitrator.update(state, self.arbitratorAction, nextState, self.getArbitratorReward(reward), done)

class DDPGAgent_ghostHand(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', **args):
        PacmanQAgent.__init__(self, **args)
        self.nb_foodFeatures = 10
        self.nb_ghostFeatures = 9
        self.nb_actions = 5
        self.arbitrator_actions = 2
        self.nb_features = 13
        self.epsilonDecay = .999

        self.ghostAgent = OptimalGhostAgent()
        # self.ghostAgent2 = DqnModule(nb_features = self.nb_ghostFeatures, featureExtractor = CustomizedExtractor().getGhostFeatures2)
        self.foodAgent = DqnModule(nb_features = self.nb_foodFeatures, featureExtractor = CustomizedExtractor().getFoodFeatures)
        self.arbitrator = DDPGModule(nb_features = self.nb_features, featureExtractor = CustomizedExtractor().getFeatures, nb_actions = self.arbitrator_actions, decay = self.epsilonDecay)
        self.subModules = [self.ghostAgent, self.foodAgent]
        self.lastSavedWeights = -1
        # self.ghostAgent.model = self.loadModel('ghostAgent_ghostTimePenalty1_1_1100')
        # # self.ghostAgent2.model = self.loadModel('ghostAgent_ghostTimePenalty1_1_1100')
        # self.foodAgent.model = self.loadModel('foodAgent_ghostTimePenalty1_1_1100')
        self.isSaved = 0

        print '----------'
        print '############ ActorCriticAgent ############'
        print 'Arbitrator Epsilon Decay = %f, Discount Factor = %.2f' % (self.arbitrator.decay, self.discount)
        print 'Feature Count: Arbitrator = %d, Ghost = %d, Food = %d' % (self.nb_features, self.nb_ghostFeatures, self.nb_foodFeatures)
        print 'Rewards for Arbitrator: (Time Penalty) = %.2f, (Eat Food) = %.2f, (Die) = %.2f,  (Eat All Food) = %.2f' % \
        (self.getArbitratorReward(TIME_PENALTY), self.getArbitratorReward(TIME_PENALTY + FOOD_REWARD), self.getArbitratorReward(TIME_PENALTY + DIE_PENALTY), self.getArbitratorReward(TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD))
        print '----------'

    def computeActionFromQValues(self, state):
        self.arbitratorAction = self.arbitrator.getAction(state)[0]
        scaleParameters = self.arbitratorAction

        ghostQValues = self.ghostAgent.getQValues(state)
        # ghost2QValues = self.ghostAgent2.getQValues(state)
        foodQValues = self.foodAgent.getQValues(state)
        scalarizedQValues = scaleParameters[0] * (ghostQValues) + scaleParameters[1] * foodQValues
        # scalarizedQValues = scaleParameter * ghostQValues + (1 - scaleParameter) * (foodQValues)

        bestAction = ActionMapping.NumbertoAction[np.argmax(scalarizedQValues)]
        return bestAction

    def getArbitratorReward(self, reward):

        MODIFIED_TIME_PENALTY = -1. # Number of points lost each round
        MODIFIED_FOOD_REWARD = 10
        MODIFIED_EAT_ALL_FOOD_REWARD = 50
        MODIFIED_DIE_PENALTY = -20

        if reward == TIME_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD
        elif reward == TIME_PENALTY + DIE_PENALTY:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_DIE_PENALTY
        elif reward == TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_EAT_ALL_FOOD_REWARD + MODIFIED_FOOD_REWARD
        elif reward == TIME_PENALTY + DIE_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD + MODIFIED_DIE_PENALTY

        return reward / 10.0

    def getFoodReward(self, reward):
        MODIFIED_TIME_PENALTY = -10. # Number of points lost each round
        MODIFIED_FOOD_REWARD = 10
        MODIFIED_EAT_ALL_FOOD_REWARD = 50

        if reward == TIME_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD
        elif reward == TIME_PENALTY + DIE_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_EAT_ALL_FOOD_REWARD + MODIFIED_FOOD_REWARD
        elif reward == TIME_PENALTY + DIE_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD

        return reward / 10.0

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

    def update(self, state, action, nextState, reward):
        # if self.selfTesting and self.currentTrainingEpisode > self.lastSavedWeights:
        #     self.saveModel(self.arbitrator.model, 'arbitrator_' + identifier + '_' + str(self.currentTrainingEpisode))
            # self.lastSavedWeights = self.currentTrainingEpisode
        if self.alpha < 0.0001:
            return

        done = 1 if abs(reward) > 100 else 0
        # if done:
        #     print '------------------------------------------------------------'
        self.foodAgent.update(state, ActionMapping.ActionToNumber[action], nextState, self.getFoodReward(reward), done)
        self.arbitrator.update(state, self.arbitratorAction, nextState, self.getArbitratorReward(reward), done)

class SequentialHierarchicalLinearFeaturesAgent(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', **args):
        PacmanQAgent.__init__(self, **args)
        self.nb_foodFeatures = 10
        self.nb_ghostFeatures = 8
        self.nb_actions = 5
        self.nb_features = 3
        self.epsilon = 1
        self.min_epsilon = 0.01
        self.decay = .999
        self.arbitratorDecay = .995
        self.arbitratorEpsilon = 1
        self.ghostAgent = DqnModule(nb_features = self.nb_ghostFeatures, featureExtractor = CustomizedExtractor().getGhostFeatures)
        self.foodAgent = DqnModule(nb_features = self.nb_foodFeatures, featureExtractor = CustomizedExtractor().getFoodFeatures)
        self.subModules = [self.ghostAgent, self.foodAgent]
        self.lastSavedWeights = -1
        self.ghostAgent.model = self.loadModel('ghostAgent_DQNBaselineAgent_allActions_1_3500')
        self.foodAgent.model = self.loadModel('foodAgent_DQNBaselineAgent_allActions_1_3500')
        self.isSaved = 0

        self.alpha = .1
        # self.weights = [util.Counter(), util.Counter()]
        self.valueTable = [[0., 0.], [0. ,0.]]
        self.featureExtractor = CustomizedExtractor().getLinearFeatures

        print '----------'
        print '############ SequentialHierarchicalLinearFeaturesAgent ############'
        print 'Epsilon Decay = %f, Arbitrator Epsilon Decay = %f, Discount Factor = %.2f' % (self.decay, self.arbitratorDecay, self.discount)
        print 'Feature Count: Arbitrator = %d, Ghost = %d, Food = %d' % (self.nb_features, self.nb_ghostFeatures, self.nb_foodFeatures)
        print 'Rewards for Arbitrator: (Time Penalty) = %.2f, (Eat Food) = %.2f, (Die) = %.2f,  (Eat All Food) = %.2f' % \
        (self.getArbitratorReward(TIME_PENALTY), self.getArbitratorReward(TIME_PENALTY + FOOD_REWARD), self.getArbitratorReward(TIME_PENALTY + DIE_PENALTY), self.getArbitratorReward(TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD))
        print '----------'

    def getStateValue(self, state):
        qv0 = self.getActionValue(state, 0)
        qv1 = self.getActionValue(state, 1)
        return qv0 if qv0 > qv1 else qv1

    def getArbitratorAction(self, state):
        qv0 = self.getActionValue(state, 0)
        qv1 = self.getActionValue(state, 1)
        return 0 if qv0 > qv1 else 1

    def getActionValue(self, state, action):
        f = self.featureExtractor(state)
        return self.valueTable[f][action]
        # qv = 0
        # for feature in f:
        #     qv = qv + self.weights[action][feature] * f[feature]
        # return qv

    def computeActionFromQValues(self, state):
        # legalActions = self.getLegalActions(state)
        allActions = ['North', 'South', 'East', 'West', 'Stop']
        legalActions = [ActionMapping.ActionToNumber[action] for action in allActions]

        if np.random.rand() < self.arbitratorEpsilon:
            self.arbitratorAction = random.randrange(2)
        else:
            self.arbitratorAction = self.getArbitratorAction(state)

        if np.random.rand() < 0.00:
            action = random.choice(legalActions)
        else:
            action = self.subModules[self.arbitratorAction].getAction(state, legalActions)

        return ActionMapping.NumbertoAction[action]

    def getArbitratorReward(self, reward):

        MODIFIED_TIME_PENALTY = -10. # Number of points lost each round
        MODIFIED_FOOD_REWARD = 10
        MODIFIED_EAT_ALL_FOOD_REWARD = 0
        MODIFIED_DIE_PENALTY = -20

        if reward == TIME_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        if reward == TIME_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD
        if reward == TIME_PENALTY + DIE_PENALTY:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_DIE_PENALTY
        if reward == TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_EAT_ALL_FOOD_REWARD + MODIFIED_FOOD_REWARD
        if reward == TIME_PENALTY + DIE_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD + MODIFIED_DIE_PENALTY

        return reward / 10.0

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

    def updateArbitrator(self, state, action, nextState, reward, done):
        alphadiff = self.alpha * ((reward + self.discount * self.getStateValue(nextState)) - self.getActionValue(state, action))
        f = self.featureExtractor(state)
        self.valueTable[f][action] += alphadiff
        # for feature in f.keys():
        #     self.weights[action][feature] = self.weights[action][feature] + alphadiff * f[feature]

    def update(self, state, action, nextState, reward):
        if self.selfTesting and self.currentTrainingEpisode > self.lastSavedWeights:
            print self.valueTable
            self.lastSavedWeights = self.currentTrainingEpisode
        if self.alpha < 0.0001:
            return

        if self.arbitratorEpsilon > self.min_epsilon:
            self.arbitratorEpsilon = self.arbitratorEpsilon * self.arbitratorDecay

        done = 1 if abs(reward) > 100 else 0
        # if done:
        #     print '------------------------------------------------------------'

        self.updateArbitrator(state, self.arbitratorAction, nextState, self.getArbitratorReward(reward), done)

class DDPGAgent2(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', **args):
        PacmanQAgent.__init__(self, **args)
        self.nb_foodFeatures = 10
        self.nb_ghostFeatures = 9
        self.nb_actions = 5
        self.arbitrator_actions = 5
        self.nb_features = 13
        self.nb_modules = 2
        self.epsilonDecay = .999
        self.min_epsilon = 0.01
        self.ghostAgent1 = DqnModule(nb_features = self.nb_ghostFeatures, featureExtractor = CustomizedExtractor().getGhostFeatures1)
        self.foodAgent = DqnModule(nb_features = self.nb_foodFeatures, featureExtractor = CustomizedExtractor().getFoodFeatures)
        self.arbitrator = ArbitratorModule(nb_features = self.nb_features, nb_modules = self.nb_modules, featureExtractor = CustomizedExtractor().getFeatures, nb_actions = self.arbitrator_actions, decay = self.epsilonDecay)
        self.subModules = [self.ghostAgent1, self.foodAgent]
        self.lastSavedWeights = -1
        self.ghostAgent1.model = self.loadModel('ghostAgent_ghostTimePenalty1_1_1100')
        self.foodAgent.model = self.loadModel('foodAgent_ghostTimePenalty1_1_1100')
        self.isSaved = 0

        print '----------'
        print '############ ActorCriticAgent2 ############'
        print 'Arbitrator Epsilon Decay = %f, Discount Factor = %.2f' % (self.arbitrator.decay, self.discount)
        print 'Feature Count: Arbitrator = %d, Ghost = %d, Food = %d' % (self.nb_features, self.nb_ghostFeatures, self.nb_foodFeatures)
        print 'Rewards for Arbitrator: (Time Penalty) = %.2f, (Eat Food) = %.2f, (Die) = %.2f,  (Eat All Food) = %.2f' % \
        (self.getArbitratorReward(TIME_PENALTY), self.getArbitratorReward(TIME_PENALTY + FOOD_REWARD), self.getArbitratorReward(TIME_PENALTY + DIE_PENALTY), self.getArbitratorReward(TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD))
        print '----------'

    def computeActionFromQValues(self, state):
        legalActions = ['North', 'South', 'East', 'West', 'Stop']
        if np.random.rand() < self.epsilon:
            return random.choice(legalActions)
        legalActions = [0, 1, 2, 3, 4]
        ghostQValues = self.ghostAgent1.getQValues(state)
        foodQValues = self.foodAgent.getQValues(state)
        modulesQvalues = np.concatenate((ghostQValues, foodQValues))
        self.arbitratorAction = self.arbitrator.getAction([state, modulesQvalues], legalActions)

        bestAction = ActionMapping.NumbertoAction[np.argmax(self.arbitratorAction)]
        return bestAction

    def getArbitratorReward(self, reward):

        MODIFIED_TIME_PENALTY = -1. # Number of points lost each round
        MODIFIED_FOOD_REWARD = 10
        MODIFIED_EAT_ALL_FOOD_REWARD = 50
        MODIFIED_DIE_PENALTY = -20

        if reward == TIME_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        if reward == TIME_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD
        if reward == TIME_PENALTY + DIE_PENALTY:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_DIE_PENALTY
        if reward == TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_EAT_ALL_FOOD_REWARD + MODIFIED_FOOD_REWARD
        if reward == TIME_PENALTY + DIE_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD + MODIFIED_DIE_PENALTY

        return reward / 10.0

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

    def update(self, state, action, nextState, reward):
        # if self.selfTesting and self.currentTrainingEpisode > self.lastSavedWeights:
        #     self.saveModel(self.arbitrator.model, 'arbitrator_' + identifier + '_' + str(self.currentTrainingEpisode))
            # self.lastSavedWeights = self.currentTrainingEpisode
        if self.alpha < 0.0001:
            return
        done = 1 if abs(reward) > 35 else 0
        # if done:
        #     print '------------------------------------------------------------'

        ghostQValues = self.ghostAgent1.getQValues(state)
        foodQValues = self.foodAgent.getQValues(state)
        modulesQvalues = np.concatenate((ghostQValues, foodQValues))

        ghostQValues = self.ghostAgent1.getQValues(nextState)
        foodQValues = self.foodAgent.getQValues(nextState)
        next_modulesQvalues = np.concatenate((ghostQValues, foodQValues))
        if self.epsilon > self.min_epsilon:
            self.epsilon = self.epsilon * self.epsilonDecay

        self.arbitrator.update([state, modulesQvalues], ActionMapping.ActionToNumber[action], [nextState, next_modulesQvalues], self.getArbitratorReward(reward), done)

class ArbitratorModule:
    def __init__(self, nb_features, nb_modules, nb_actions, featureExtractor, batch_size = 32, start_epsilon = 1, min_epsilon = 0.01, decay = 0.9995, discount = 0.8):
        self.batch_size = batch_size
        self.epsilon = start_epsilon
        self.min_epsilon = min_epsilon
        self.nb_features = nb_features
        self.nb_modules = nb_modules
        self.nb_actions = nb_actions
        self.decay = decay
        self.discount = discount
        self.model = self.create_model()
        self.replay_memory_buffer = deque(maxlen=50000)
        self.extractor = featureExtractor
        print '----------'
        print '### DqnModule ###'
        print 'Epsilon Decay = %s, Discount Factor = %.2f' % (self.decay, self.discount)
        print 'Input Features = %d' % (nb_features)
        print '----------'

    def create_model(self):
        state_input = Input(shape=(self.nb_features,))
        state_h1 = Dense(64, activation='relu')(state_input)
        state_h2 = Dense(128)(state_h1)

        action_input = Input(shape=(self.nb_modules * self.nb_actions,))
        action_h1 = Dense(64)(action_input)

        merged = Concatenate()([state_h2, action_h1])
        merged_h1 = Dense(64, activation='relu')(merged)
        output = Dense(self.nb_actions, activation='linear')(merged_h1)
        model = Model(input=[state_input,action_input], output=output)

        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return model

    def makeState(self, state):
        state = [[self.extractor(state[0])], [state[1]]]
        return state

    def getQValue(self, state, action):
        state = self.makeState(state)
        qValues =  self.model.predict(state, batch_size=1)[0]
        return qValues[action]

    def getAction(self, state, legalActions):
        state = self.makeState(state)
        # print "ACTUAL", state

        qValues = self.model.predict(state, batch_size=1)[0]
        maxQ, bestAction = float('-inf'), None
        for action in legalActions:
            if qValues[action] > maxQ:
                maxQ, bestAction = qValues[action], action
        return bestAction

    def getQValues(self, state):
        state = self.makeState(state)
        return self.model.predict(state, batch_size=1)[0]

    def update(self, state, action, nextState, reward, done):
        self.add_to_replay_memory(state, action, reward, nextState, done)
        self.replayExperience()

    def add_to_replay_memory(self, state, action, reward, next_state, done):
        self.replay_memory_buffer.append((self.makeState(state),
            action, reward, self.makeState(next_state), done))


    def replayExperience(self):
        if len(self.replay_memory_buffer) < self.batch_size:
            return
        random_sample = self.get_random_sample_from_replay_mem()
        indexes = np.array([i for i in range(self.batch_size)])
        states, actions, rewards, next_states, done_list = self.get_attribues_from_sample(random_sample)
        targets = rewards + self.discount * (self.model.predict_on_batch(next_states)[0, actions]) * (1 - done_list)

        target_vec = self.model.predict_on_batch(states)
        target_vec[[indexes], [actions]] = targets

        self.model.fit(states, target_vec, epochs=1, verbose=0)

    def get_random_sample_from_replay_mem(self):
        return random.sample(self.replay_memory_buffer, self.batch_size)

    def make_state_attributes(self, states):
        s0 = np.squeeze(np.array([s[0] for s in states]))
        s1 = np.squeeze(np.array([s[1] for s in states]))
        s = [s0, s1]
        return s

    def get_attribues_from_sample(self, random_sample):
        states = [i[0] for i in random_sample]
        states = self.make_state_attributes(states)

        actions = np.array([i[1] for i in random_sample])
        rewards = np.array([i[2] for i in random_sample])
        next_states = [i[3] for i in random_sample]
        next_states = self.make_state_attributes(next_states)

        done_list = np.array([i[4] for i in random_sample])
        return states, actions, rewards, next_states, done_list


class DDPGModule:
    def __init__(self, nb_features, featureExtractor, nb_actions, decay):
        self.sess = tf.Session()
        K.set_session(self.sess)
        self.alpha = 0.0005
        self.epsilon = .9
        self.min_epsilon = .01
        self.decay = decay
        self.gamma = .8
        self.tau   = .01
        self.batch_size = 32
        self.extractor = featureExtractor
        self.nb_features = nb_features
        self.nb_actions = nb_actions
        print '----------'
        print '### DDPG Module ###'
        print 'Epsilon Decay = %s, Discount Factor = %.2f, alpha = %f' % (self.decay, self.gamma, self.alpha)
        print 'Input Features = %d' % (self.nb_features)
        print '----------'

		# ===================================================================== #
		#                               Actor Model                             #
		# Chain rule: find the gradient of chaging the actor network params in  #
		# getting closest to the final value network predictions, i.e. de/dA    #
		# Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
		# ===================================================================== #

        self.replay_memory_buffer = deque(maxlen=50000)
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        self.actor_critic_grad = tf.placeholder(tf.float32,
	       [None, self.nb_actions]) # where we will feed de/dC (from critic)

        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output,
        	actor_model_weights, -self.actor_critic_grad) # dC/dA (from actor)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.alpha).apply_gradients(grads)

        # ===================================================================== #
        #                              Critic Model                             #
        # ===================================================================== #

        self.critic_state_input, self.critic_action_input, \
        	self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()

        self.critic_grads = tf.gradients(self.critic_model.output,
        	self.critic_action_input) # where we calcaulte de/dC for feeding above

    	# Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())

    # ========================================================================= #
    #                              Model Definitions                            #
    # ========================================================================= #

    def create_actor_model(self):
        state_input = Input(shape=(self.nb_features,))
        h1 = Dense(64, activation='relu')(state_input)
        h2 = Dense(128, activation='relu')(h1)
        h3 = Dense(64, activation='relu')(h2)
        output = Dense(self.nb_actions, activation='tanh')(h3)

        model = Model(input=state_input, output=output)
        adam  = Adam(lr=self.alpha)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        state_input = Input(shape=(self.nb_features,))
        state_h1 = Dense(64, activation='relu')(state_input)
        state_h2 = Dense(128)(state_h1)

        action_input = Input(shape=(self.nb_actions,))
        action_h1 = Dense(64)(action_input)

        merged = Concatenate()([state_h2, action_h1])
        merged_h1 = Dense(64, activation='relu')(merged)
        output = Dense(1, activation='linear')(merged_h1)
        model = Model(input=[state_input,action_input], output=output)

        adam = Adam(lr=self.alpha)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

	# ========================================================================= #
	#                               Model Training                              #
	# ========================================================================= #
    def update(self, state, action, nextState, reward, done):
        if self.alpha < 0.000001:
            return
        # action = (action - 0.5) * 2
        self.add_to_replay_memory(state, action, reward, nextState, done)
        self.train()
        self.update_target()

    def add_to_replay_memory(self, state, action, reward, next_state, done):
        self.replay_memory_buffer.append((self.extractor(state),
            action, reward, self.extractor(next_state), done))

    def _train_actor(self, samples):

        cur_states, actions, rewards, new_states, _ =  self.get_attributes_from_sample(samples)
        predicted_actions = self.actor_model.predict(cur_states)
        grads = self.sess.run(self.critic_grads, feed_dict={
        	self.critic_state_input:  cur_states,
        	self.critic_action_input: predicted_actions
        })[0]

        self.sess.run(self.optimize, feed_dict={
        	self.actor_state_input: cur_states,
        	self.actor_critic_grad: grads
        })

    def _train_critic(self, samples):

        cur_states, actions, rewards, new_states, dones = self.get_attributes_from_sample(samples)
        target_actions = self.target_actor_model.predict(new_states)
        future_rewards = self.target_critic_model.predict([new_states, target_actions])

        rewards += self.gamma * future_rewards * (1 - dones)

        evaluation = self.critic_model.fit([cur_states, actions], rewards, verbose=0)
        #print(evaluation.history)

    def train(self):
        if len(self.replay_memory_buffer) < self.batch_size:
            return

        rewards = []
        samples = random.sample(self.replay_memory_buffer, self.batch_size)
        self._train_critic(samples)
        self._train_actor(samples)

    # ========================================================================= #
    #                         Target Model Updating                             #
    # ========================================================================= #

    def _update_actor_target(self):
        actor_model_weights  = self.actor_model.get_weights()
        actor_target_weights = self.target_actor_model.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]*self.tau + actor_target_weights[i]*(1-self.tau)
        self.target_actor_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights  = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]*self.tau + critic_target_weights[i]*(1-self.tau)
        self.target_critic_model.set_weights(critic_target_weights)

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    def get_attributes_from_sample(self, random_sample):
        array = np.array(random_sample)

        current_states = np.stack(array[:,0]).reshape((array.shape[0],-1))
        actions = np.stack(array[:,1]).reshape((array.shape[0],-1))
        rewards = np.stack(array[:,2]).reshape((array.shape[0],-1))
        new_states = np.stack(array[:,3]).reshape((array.shape[0],-1))
        dones = np.stack(array[:,4]).reshape((array.shape[0],-1))

        return current_states, actions, rewards, new_states, dones

    # ========================================================================= #
    #                              Model Predictions                            #
    # ========================================================================= #

    def getAction(self, state):
        state = self.extractor(state).reshape((1, self.nb_features))
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay
        if np.random.random() < self.epsilon:
            noise = np.random.uniform(-1, 1, size = self.nb_actions)
            action = self.actor_model.predict(state) + noise
            return action
        action = self.actor_model.predict(state)
        return action

class HierarchicalDDPGAgent(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', **args):
        PacmanQAgent.__init__(self, **args)
        self.nb_foodFeatures = 10
        self.nb_ghostFeatures = 9
        # self.nb_puddleFeatures = 15
        self.nb_actions = 5
        self.nb_features = 13
        self.arbitrator_actions = 2
        # self.epsilon = 1
        # self.min_epsilon = 0.01
        self.decay = .999
        # self.puddleAgent = DqnModule(nb_features = self.nb_puddleFeatures, featureExtractor = CustomizedExtractor().getPuddleFeatures)
        self.ghostAgent = DqnModule(nb_features = self.nb_ghostFeatures, featureExtractor = CustomizedExtractor().getGhostFeatures)
        self.foodAgent = DqnModule(nb_features = self.nb_foodFeatures, featureExtractor = CustomizedExtractor().getFoodFeatures)
        self.arbitrator = DDPGModule(nb_features = self.nb_features, featureExtractor = CustomizedExtractor().getFeatures, nb_actions = self.arbitrator_actions, decay = self.decay)
        self.subModules = [self.ghostAgent, self.foodAgent]
        self.lastSavedWeights = -1
        # self.foodAgent.model = self.loadModel(name)
        # self.ghostAgent.model = self.loadModel(name)
        self.isSaved = 0
        print '----------'
        print '############ HierarchicalDDPGAgent ############'
        print 'Arbitrator Epsilon Decay = %f, Discount Factor = %.2f' % (self.arbitrator.decay, self.discount)
        print 'Feature Count: Arbitrator = %d, Ghost = %d, Food = %d' % (self.nb_features, self.nb_ghostFeatures, self.nb_foodFeatures)
        print 'Rewards for Arbitrator: (Eat ghost) = %.2f, (Eat Food) = %.2f, (Death Penalty) = %.2f, (Time Penalty) = %.2f' % \
        (self.getArbitratorReward(50), self.getArbitratorReward(10), self.getArbitratorReward(-500), self.getArbitratorReward(-1))
        print 'Rewards for foodAgent: Time Penalty = %.2f, (Food Reward + Time Penalty) = %.2f, \
        (Food Reward + Time Penalty + LastReward) = %.2f' % (self.getFoodReward(TIME_PENALTY), \
        self.getFoodReward(TIME_PENALTY + FOOD_REWARD), self.getFoodReward(TIME_PENALTY + FOOD_REWARD + EAT_ALL_FOOD_REWARD))
        print 'Rewards for ghostAgent: Time Penalty = %.2f, (Death Penalty) = %.2f' % (self.getGhostReward(TIME_PENALTY), \
        self.getGhostReward(TIME_PENALTY + DIE_PENALTY))
        print '----------'


    def computeActionFromQValues(self, state):
        self.arbitratorAction = self.arbitrator.getAction(state)[0]
        scaleParameters = self.arbitratorAction

        # if self.currentTrainingEpisode > 300:
        # print state
        # print 'action = ', scaleParameter

        # puddleQValues = self.puddleAgent.getQValues(state)
        ghostQValues = self.ghostAgent.getQValues(state)
        foodQValues = self.foodAgent.getQValues(state)
        scalarizedQValues = scaleParameters[0] * (ghostQValues) + scaleParameters[1] * foodQValues
        # scalarizedQValues = scaleParameters[0] * ghostQValues + scaleParameters[1] * foodQValues + scaleParameters[2] * puddleQValues
        # scalarizedQValues = scaleParameter * ghostQValues + (1 - scaleParameter) * (foodQValues)

        bestAction = ActionMapping.NumbertoAction[np.argmax(scalarizedQValues)]
        return bestAction

    def getPuddleReward(self, reward):

        MODIFIED_PUDDLE_PENALTY = -3.0
        MODIFIED_TIME_PENALTY = -1. # Number of points lost each round

        if reward == TIME_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_PUDDLE_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_PUDDLE_PENALTY
        else:
            reward = MODIFIED_TIME_PENALTY

        return reward / 10.0

    def getFoodReward(self, reward):
        MODIFIED_TIME_PENALTY = -10. # Number of points lost each round
        MODIFIED_FOOD_REWARD = 10
        MODIFIED_EAT_ALL_FOOD_REWARD = 50
        # MODIFIED_PUDDLE_PENALTY = 10

        if reward == TIME_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD
        elif reward == TIME_PENALTY + DIE_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_EAT_ALL_FOOD_REWARD + MODIFIED_FOOD_REWARD
        elif reward == TIME_PENALTY + DIE_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD

        elif reward == TIME_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_TIME_PENALTY

        return reward / 10.0

    def getGhostReward(self, reward):
        MODIFIED_TIME_PENALTY = -1 # Number of points lost each round
        MODIFIED_DIE_PENALTY = -20

        if reward == TIME_PENALTY:
            reward = -MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + FOOD_REWARD:
            reward = -MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY:
            reward = MODIFIED_DIE_PENALTY
        elif reward == TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD:
            reward = -MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY + FOOD_REWARD:
            reward = MODIFIED_DIE_PENALTY

        elif reward == TIME_PENALTY + PUDDLE_PENALTY:
            reward = -MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_DIE_PENALTY

        return reward / 10.0

    def getArbitratorReward(self, reward):

        MODIFIED_PUDDLE_PENALTY = .8
        MODIFIED_TIME_PENALTY = -1. # Number of points lost each round
        MODIFIED_FOOD_REWARD = 10
        MODIFIED_EAT_ALL_FOOD_REWARD = 50
        MODIFIED_DIE_PENALTY = -20

        if reward == TIME_PENALTY:
            reward = MODIFIED_TIME_PENALTY
        elif reward == TIME_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD
        elif reward == TIME_PENALTY + DIE_PENALTY:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_DIE_PENALTY
        elif reward == TIME_PENALTY + EAT_ALL_FOOD_REWARD + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_EAT_ALL_FOOD_REWARD + MODIFIED_FOOD_REWARD
        elif reward == TIME_PENALTY + DIE_PENALTY + FOOD_REWARD:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_FOOD_REWARD + MODIFIED_DIE_PENALTY

        elif reward == TIME_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_PUDDLE_PENALTY
        elif reward == TIME_PENALTY + DIE_PENALTY + PUDDLE_PENALTY:
            reward = MODIFIED_TIME_PENALTY + MODIFIED_DIE_PENALTY + MODIFIED_PUDDLE_PENALTY

        return reward / 10.0

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

    def update(self, state, action, nextState, reward):
        if self.selfTesting and self.currentTrainingEpisode > self.lastSavedWeights:
            #self.saveModel(self.ghostAgent.model, 'ghostAgent_' + identifier + '_' + str(self.currentTrainingEpisode))
            #self.saveModel(self.foodAgent.model, 'foodAgent_' + identifier + '_' + str(self.currentTrainingEpisode))
            # self.saveModel(self.puddleAgent.model, 'puddleAgent_' + identifier + '_' + str(self.currentTrainingEpisode))
            #self.saveModel(self.arbitrator.actor_model, 'actor_' + identifier + '_' + str(self.currentTrainingEpisode))
            #self.saveModel(self.arbitrator.critic_model, 'critic_' + identifier + '_' + str(self.currentTrainingEpisode))
            # self.saveModel(self.arbitrator.model, 'arbitrator_' + identifier + '_' + str(self.currentTrainingEpisode))
            self.lastSavedWeights = self.currentTrainingEpisode
        if self.alpha < 0.0001:
            return
        # if self.epsilon > self.min_epsilon:
        #     self.epsilon = self.epsilon * self.decay

        done = 1 if abs(reward) > 100 else 0
        # if done:
        #     print '------------------------------------------------------------'

        self.arbitrator.update(state, self.arbitratorAction, nextState, self.getArbitratorReward(reward), done)
        self.ghostAgent.update(state, ActionMapping.ActionToNumber[action], nextState, self.getGhostReward(reward), done)
        self.foodAgent.update(state, ActionMapping.ActionToNumber[action], nextState, self.getFoodReward(reward), done)
        # self.puddleAgent.update(state, ActionMapping.ActionToNumber[action], nextState, self.getPuddleReward(reward), done)

class TestingAgentArbiQ(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', **args):
        PacmanQAgent.__init__(self, **args)
        self.nb_foodFeatures = 10
        self.nb_ghostFeatures = 9
        self.nb_actions = 5
        self.nb_features = 13
        self.ghostAgent = DqnModule(nb_features = self.nb_ghostFeatures, featureExtractor = CustomizedExtractor().getGhostFeatures)
        self.foodAgent = DqnModule(nb_features = self.nb_foodFeatures, featureExtractor = CustomizedExtractor().getFoodFeatures)
        self.arbitrator = DqnModule(nb_features = self.nb_features, featureExtractor = CustomizedExtractor().getFeatures, nb_actions = 2)
        self.subModules = [self.ghostAgent, self.foodAgent]
        self.ghostAgent.model = self.loadModel('ghostAgent_ghostTimePenalty1_1_1100')
        self.foodAgent.model = self.loadModel('foodAgent_ghostTimePenalty1_1_1100')
        self.arbitrator.model = self.loadModel('arbitrator_1_3000')

        # self.foodAgent = self.loadModel('foodAgent_HierarchicalDDPGAgent_3_500')

    def computeActionFromQValues(self, state):
        self.arbitratorAction = self.arbitrator.getAction(state, [0, 1])
        # print self.arbitratorAction
        legalActions = [ActionMapping.ActionToNumber[action] for action in self.getLegalActions(state)]
        action = self.subModules[self.arbitratorAction].getAction(state, legalActions)
        return ActionMapping.NumbertoAction[action]

    def loadModel(self, file_name):
        json_file = open('weights/' + file_name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights('weights/' + file_name + '.h5')
        return loaded_model

    def update(self, state, action, nextState, reward):
        return

class TestingAgent(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', **args):
        PacmanQAgent.__init__(self, **args)
        # self.arbitratorAgent = self.loadPretrainedModel('arbirator_SeqHeir_allActions_eps_999_2_8000')
        # self.optAgent = self.loadPretrainedModel('DQNBaselineAgent_1_1000')
        # self.ghostAgent = self.loadModel('ghostAgent_1_1000')
        # self.foodAgent = self.loadModel('foodAgent_1_1000')
        # self.puddleAgent = self.loadModel('puddleAgent__50')
        self.ghostAgent = self.loadModel('ghostAgent_ghostTimePenalty1_1_1100')
        self.foodAgent = self.loadModel('foodAgent_ghostTimePenalty1_1_1100')
        # self.puddleAgent = self.loadModel('puddleAgent_shaped_100')
        # self.puddleAgent = self.loadModel('puddleAgent_puddle_in_100')
        self.puddleAgent = self.loadModel('puddleAgent__50')

        # self.foodAgent = self.loadModel('foodAgent_HierarchicalDDPGAgent_3_500')

    def computeActionFromQValues(self, state):
        # ghostFeatures = CustomizedExtractor().getGhostFeatures(state)
        # qValues1 = self.ghostAgent.predict(np.array([ghostFeatures]), batch_size=1)[0]
        # foodFeatures = CustomizedExtractor().getFoodFeatures(state)
        # qValues2 = self.foodAgent.predict(np.array([foodFeatures]), batch_size=1)[0]
        puddleFeatures = CustomizedExtractor().getPuddleFeatures(state)
        qValues3 = self.puddleAgent.predict(np.array([puddleFeatures]), batch_size=1)[0]
        # qValues = (qValues1) + (qValues2) + (qValues3)
        qValues = (qValues3)
        bestAction = ActionMapping.NumbertoAction[np.argmax(qValues)]
        print qValues, bestAction
        return bestAction

    def loadModel(self, file_name):
        json_file = open('weights/' + file_name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights('weights/' + file_name + '.h5')
        return loaded_model

    def update(self, state, action, nextState, reward):
        print reward
        return

class TestingAgentDDPG(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', **args):
        PacmanQAgent.__init__(self, **args)
        # self.arbitratorAgent = self.loadPretrainedModel('arbirator_SeqHeir_allActions_eps_999_2_8000')
        # self.optAgent = self.loadPretrainedModel('DQNBaselineAgent_1_1000')
        # self.actor = self.loadModel('actor_3_newReward_1100')
        # self.actor = self.loadModel('actor_1_positiveScale_50')
        # self.actor = self.loadModel('actor_2_positiveScale_1980')
        self.actor = self.loadModel('actor_3_1000')

        # self.actor = self.loadModel('actor_3_positiveScale_1700')

        self.ghostAgent = self.loadModel('ghostAgent_ghostTimePenalty1_1_1100')
        self.foodAgent = self.loadModel('foodAgent_ghostTimePenalty1_1_1100')
        self.puddleAgent = self.loadModel('puddleAgent__50')
        # self.foodAgent = self.loadModel('foodAgent_HierarchicalDDPGAgent_3_500')

    def computeActionFromQValues(self, state):

        ddpgState = CustomizedExtractor().getFeatures(state).reshape((1, 26))
        scaleParameters = self.actor.predict(ddpgState)[0]
        print scaleParameters
        ghostFeatures = CustomizedExtractor().getGhostFeatures(state)
        ghostQValues = self.ghostAgent.predict(np.array([ghostFeatures]), batch_size=1)[0]
        foodFeatures = CustomizedExtractor().getFoodFeatures(state)
        foodQValues = self.foodAgent.predict(np.array([foodFeatures]), batch_size=1)[0]
        puddleFeatures = CustomizedExtractor().getPuddleFeatures(state)
        # puddleQValues = self.puddleAgent.predict(np.array([puddleFeatures]), batch_size=1)[0] * 5
        # for i in range(100000):
            # pass
        puddleQValues = self.puddleAgent.predict(np.array([puddleFeatures]), batch_size=1)[0]

        # qValues = scaleParameters[0] * ghostQValues + scaleParameters[1] * foodQValues
        qValues = scaleParameters[0] * ghostQValues + scaleParameters[1] * foodQValues + scaleParameters[2] * puddleQValues

        bestAction = ActionMapping.NumbertoAction[np.argmax(qValues)]
        # print scaleParameters, bestAction
        # print puddleQValues
        # print state.getPacmanPosition()
        return bestAction

    def loadModel(self, file_name):
        json_file = open('weights/' + file_name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights('weights/' + file_name + '.h5')
        return loaded_model

    def update(self, state, action, nextState, reward):
        return


class OptimalSequentialHeirarchicalAgent():
    def __init__(self):
        '''
            0 for ghostAgent,
            1 for foodAgent.
        '''
        self.subModules = [OptimalGhostAgent(), OptimalFoodAgent()]

    def getArbitratorAction(self, state):
        x, y = state.getPacmanPosition()
        ghosts = state.getGhostPositions()
        walls = state.getWalls()
        dist = closestGhost((x, y), ghosts, walls)
        return 1 if dist > 1 else 0

        return self.valueTable[f][action]

    def getAction(self, state):
        arbitratorAction = self.getArbitratorAction(state)
        action = self.subModules[arbitratorAction].getAction(state)
        return action

class OptimalFoodAgent:

    def getAction(self, state):
        # print state.getPacmanState().configuration, state.data.layout.walls
        legalActions = Actions.getPossibleActions( state.getPacmanState().configuration, state.data.layout.walls )
        minDistance, bestAction = float('inf'), None
        x, y = state.getPacmanPosition()
        food = state.getFood()
        walls = state.getWalls()

        for action in legalActions:
            dx, dy = Actions.directionToVector(action)
            next_x, next_y = int(x + dx), int(y + dy)
            dist = closestFood((next_x, next_y), food, walls)
            if dist != None and dist < minDistance:
                minDistance, bestAction = dist, action
        return bestAction

class OptimalGhostAgent:

    def getAction(self, state):
        # print state.getPacmanState().configuration, state.data.layout.walls
        legalActions = Actions.getPossibleActions( state.getPacmanState().configuration, state.data.layout.walls )
        maxDistance, bestAction = float('-inf'), None
        x, y = state.getPacmanPosition()
        ghosts = state.getGhostPositions()
        walls = state.getWalls()
        dist = closestGhost((x, y), ghosts, walls)
        if dist > 1:
            return 'Stop'
        for action in legalActions:
            dx, dy = Actions.directionToVector(action)
            next_x, next_y = int(x + dx), int(y + dy)
            dist = closestGhost((next_x, next_y), ghosts, walls)
            if dist != None and dist > maxDistance:
                maxDistance, bestAction = dist, action
        return bestAction

    def getQValues(self, state):
        x, y = state.getPacmanPosition()
        ghosts = state.getGhostPositions()
        walls = state.getWalls()
        dist = closestGhost((x, y), ghosts, walls)

        if dist > 1.:
            return np.array([1., 1., 1., 1., 1.])

        legalActions = Actions.getPossibleActions( state.getPacmanState().configuration, state.data.layout.walls )
        distances = np.array([0., 0., 0., 0., 0.])

        for action in legalActions:
            dx, dy = Actions.directionToVector(action)
            next_x, next_y = int(x + dx), int(y + dy)
            dist = closestGhost((next_x, next_y), ghosts, walls)
            if dist != None:
                distances[ActionMapping.ActionToNumber[action]] = dist
        maxDistance = np.argmax(distances)
        EPS = .01
        for i in range(5):
            distances[i] = 1 if distances[i] + EPS < maxDistance else 0
        return distances


class NeuralNetwork:
    def __init__(self, input_size, nb_actions):
        self.input_size = input_size
        self.nb_actions = nb_actions

        self.model = Sequential()
        self.model.add(Dense(64, init='lecun_uniform', input_shape=(self.input_size,)))
        self.model.add(Activation('relu'))

        self.model.add(Dense(64, init='lecun_uniform'))
        self.model.add(Activation('relu'))

        self.model.add(Dense(32, init='lecun_uniform'))
        self.model.add(Activation('relu'))

        self.model.add(Dense(self.nb_actions, init='lecun_uniform'))
        self.model.add(Activation('linear'))

        # rms = RMSprop(lr=0.000001, rho=0.6)
        adamOptimizer = Adam(lr=0.0002)

        self.model.compile(loss='mse', optimizer=adamOptimizer)
