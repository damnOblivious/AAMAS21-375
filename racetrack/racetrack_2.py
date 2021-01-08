# coding: utf-8

import math
import time
import random
import numpy as np

import layout_parser
from variables import *
from qLearningAgents import *
from visualizer import Visualizer
from environment import Environment

layout_name = 'f1'
layout = layout_parser.getLayout( layout_name )
visuals = Visualizer(layout)
env = Environment(layout)
# agent = HierarchicalDDPGAgent(layout = layout)
# agent = DQNBaselineAgent(layout = layout)
# agent = GmQAgent(layout = layout)
# agent = TestingAgent(layout = layout)
# agent = CollisionAgent(layout = layout)
# agent = SequentialArbiQAgent(layout = layout)
# agent = SequentialDDPGAgent(layout = layout)
agent = DDPGGraphAgent(layout = layout)

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

def findDifference(q_arbi, q_joint):
    return abs(q_arbi - q_joint)

def run_episode(agent, env, visuals, testing = False):
    env.reset()
    state = env.start()
    episode_score, episode_steps = 0, 0
    # valueDifference = 0.
    all_q_values = [0., 0., 0.]
    gameOver, done = False, False
    while not gameOver:
        episode_steps += 1
        if testing:
            action, q_arbi, q_joint, dqn_q = agent.getActionWithQValues(state, testing)
            # difference = findDifference(q_arbi, q_joint)
            # valueDifference += difference
            all_q_values[0] += q_arbi
            all_q_values[1] += q_joint
            all_q_values[2] += dqn_q
        else:
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

    if testing:
        # return episode_score, episode_steps, done, valueDifference
        return episode_score, episode_steps, done, all_q_values
    else:
        return episode_score, episode_steps, done

def begin_testing(agent, env, visuals):
    score_list = []
    total_steps, finish_count = 0, 0
    # q_differences = 0.
    all_q_values = [0., 0., 0.]
    for test_num in range(testRepitition):
        episode_score, episode_steps, done, all_q_value = run_episode(agent, env, visuals, testing = True)
        total_steps += episode_steps
        finish_count += done
        score_list.append(episode_score)
        all_q_values[0] = all_q_value[0]
        all_q_values[1] = all_q_value[1]
        all_q_values[2] = all_q_value[2]
        # q_differences += q_difference
    all_q_values = [round(x / testRepitition, 3) for x in all_q_values]
    # return score_list, total_steps, finish_count, round(q_differences / testRepitition, 3)
    return score_list, total_steps, finish_count, all_q_values


numEpisodes = 2000
def start_it_baby():
    current_training_average, training_averages, training_steps = 0., [], 0.
    test_scores, test_averages, test_finishes = [], [], []
    q_differences = []
    q_values = [[], [], []]
    for episode_num in range(numEpisodes):

        agent.training_episode_num = episode_num
        episode_score, episode_steps, done = run_episode(agent, env, visuals)
        current_training_average += episode_score
        training_steps += episode_steps

        # Start Testing Right away
        if ((episode_num+1) % testInterval == 0) and episode_num > 0:
            current_training_average /= float(testInterval)
            training_averages.append(current_training_average)
            print 'Episodes Completed = %d and training_steps = %d, average = %.2f' % (episode_num, training_steps, current_training_average)
            # print training_rewards
            current_training_average = 0.
            score_list, total_steps, finish_count, q_value = begin_testing(agent, env, visuals)
            test_scores += score_list
            test_finishes.append(finish_count)
            test_averages.append(sum(score_list) / float(len(score_list)))
            q_values[0].append(q_value[0])
            q_values[1].append(q_value[1])
            q_values[2].append(q_value[2])
            # q_differences.append(round(q_difference, 3))
            print score_list, total_steps, finish_count
            # print test_averages, q_values


    print 'TRAINING AVERAGE', training_averages
    print 'TESTING AVERAGE', test_averages
    print 'Q DIFFERENCE', q_values
start_it_baby()
