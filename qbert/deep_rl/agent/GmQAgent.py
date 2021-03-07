#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from ..utils import *
import time
from .BaseAgent import *
from .DQN_module import *
from .rewardFunctions import *

class GmQAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.numModules = 4
        self.config = config
        self.modules = [DQNModule(self.config) for i in range(self.numModules)]
        self.task = config.task_fn()
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.teacher = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.criterion = nn.MSELoss()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.policy = config.policy_fn()
        self.total_steps = 0
        self.intervalAsked = 0
        self.intervalGot = 0

    def get_final_q_values(self, state):
        finalQValues = self.modules[0].get_q_values(state) + self.modules[1].get_q_values(state) + self.modules[2].get_q_values(state) + self.modules[3].get_q_values(state)
        return finalQValues

    def own_select_action(self, state):
        eps = self.policy.epsilon(0)

        if np.random.uniform() < eps:
            action = np.random.randint(0, self.task.action_dim)
        else:
            q_values = self.get_final_q_values(state)
            action = np.argmax(q_values)
        return action


    def episode(self, deterministic=False):
        episode_start_time = time.time()
        state = self.task.reset()
        total_reward = 0.0
        steps = 0
        while True:
            value = self.get_final_q_values(state)
            if deterministic:
                if np.random.random() > 0.05:
                    action = np.argmax(value)
                else:
                    action = np.random.randint(0, len(value))

            elif self.total_steps < self.config.exploration_steps:
                action = np.random.randint(0, self.task.action_dim)
            else:
                action = np.argmax(value)

            next_state, reward, done, _ = self.task.step(action)
            total_reward += reward
            reward = self.config.reward_normalizer(reward)
            if not deterministic:
                self.modules[0].update(state, action, change_reward0(reward), next_state, done, deterministic)
                self.modules[1].update(state, action, change_reward1(reward), next_state, done, deterministic)
                self.modules[2].update(state, action, change_reward2(reward), next_state, done, deterministic)
                self.modules[3].update(state, action, change_reward3(reward, done), next_state, done, deterministic)
                self.total_steps += 1
            steps += 1

            if self.total_steps % (self.config.test_interval * 25) == 0 and self.total_steps > 0:
                torch.save(self.network,self.task.name + "_"+str(self.total_steps)+".net")

            state = next_state
            #

            if done:
                break

        episode_time = time.time() - episode_start_time
        self.config.logger.debug('episode steps %d, episode time %f, time per step %f' %
                          (steps, episode_time, episode_time / float(steps)))
        return total_reward, steps
