# coding: utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import math
import time
import random
import numpy as np
import sys

import layout_parser
from variables import *
from qLearningAgents import *
from visualizer import Visualizer
from environment import Environment
from rewardScales import *

if len(sys.argv) > 1: TRIAL_NUM = int(sys.argv[1])
else: TRIAL_NUM = -1

layout_name = 'f1'
layout = layout_parser.getLayout( layout_name )
visuals = Visualizer(layout)
env = Environment(layout)
'''
Choose you agent here.
'''
# agent = HierarchicalDDPGAgent(layout = layout, trialNum = TRIAL_NUM)
# agent = DQNBaselineAgent(layout = layout)
#agent = GmQAgent(layout = layout, trialNum = TRIAL_NUM)
# agent = TestingAgent(layout = layout)
# agent = CollisionAgent(layout = layout)
# agent = SequentialArbiQAgent(layout = layout)
#agent = SequentialDDPGAgent(layout = layout)
agent = DDPG_finishPre(layout = layout)
# ################################################################################
# ################################################################################
#
# env.reset()
# state = env.start()
# done = False
# reward = -1
# action = (1,1)
# while 1:
#     if done:
#         print "----------------------------DONE----------------------------"
#         env.reset()
#         reward = -1
#         action = (1,1)
#         state = env.start()
#     state, reward, done = env.step(state, action)
#     alpha = [(-1,-1),(-1,0),(0,-1),(-1,1),(0,0),(1,-1),(0,1),(1,0),(1,1)]
#     action = random.choice(alpha)
#     # action = (1,1)
#     # print 'state', state, 'action', action
#     for i in range(100000): pass
#     visuals.visualize_racetrack(state)
#
# ################################################################################
# ################################################################################
def run_episode(agent, env, visuals, testing = False):
    env.reset()
    state = env.start()
    episode_score, episode_steps = 0, 0
    gameOver, done = False, False
    while not gameOver:
        # for i in range(10000000):
        #     pass
        # visuals.visualize_racetrack(state)
        episode_steps += 1
        action = agent.getAction(state, testing)
        next_state, reward, done = env.step(state, action)
        episode_score += reward
        if not testing:
            shapedReward = env.getShapedReward(state, next_state)
            agent.update(state, action, next_state, (reward + shapedReward), done)
        state = next_state

        testing_limit_exceed = (episode_steps > TESTING_STEP_LIMIT) and testing
        training_limit_exceed = (episode_steps > TRAINING_STEP_LIMIT) and not testing

        gameOver = done or testing_limit_exceed or training_limit_exceed

    # if done: print "-----------------------DONE-------------------------"
    # else: print "---------------------CRASHED------------------------"

    return episode_score, episode_steps, done

def begin_testing(agent, env, visuals):
    score_list = []
    total_steps, finish_count = 0, 0
    for test_num in range(testRepitition):
        episode_score, episode_steps, done = run_episode(agent, env, visuals, testing = True)
        total_steps += episode_steps
        finish_count += done
        score_list.append(episode_score)
    return score_list, total_steps, finish_count


numEpisodes = 2000
def start_it_baby():
    current_training_average, training_averages, training_steps = 0., [], 0.
    test_scores, test_averages, test_finishes = [], [], []
    for episode_num in range(numEpisodes):

        agent.training_episode_num = episode_num
        episode_score, episode_steps, done = run_episode(agent, env, visuals)
        current_training_average += episode_score
        training_steps += episode_steps

        if ((episode_num+1) % testInterval == 0) and episode_num > START_TESTING_FROM:
            current_training_average /= float(testInterval)
            training_averages.append(current_training_average)
            # print 'Episodes Completed = %d and training_steps = %d, average = %.2f' % (episode_num, training_steps, current_training_average)
            # print training_rewards
            current_training_average = 0.
            score_list, total_steps, finish_count = begin_testing(agent, env, visuals)
            test_scores += score_list
            test_finishes.append(finish_count)
            test_averages.append(sum(score_list) / float(len(score_list)))
            print score_list, total_steps, finish_count
            # print test_averages
    if TRIAL_NUM != -1:
        print "TRIAL NUM-", TRIAL_NUM, rewardScales[TRIAL_NUM]
    print 'TRAINING AVERAGE', training_averages
    print 'TESTING AVERAGE', test_averages
    print "---------------------------------------------------"

start_it_baby()
