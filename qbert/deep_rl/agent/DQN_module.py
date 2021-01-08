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

class DQNModule(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
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

    def get_q_values(self, state):
        myQvalues = self.network.predict(np.stack([self.config.state_normalizer(state)]), True).flatten()
        return myQvalues

    def own_select_action(self, state):
        eps = self.policy.epsilon(0)
        if np.random.uniform() < eps:
            action = np.random.randint(0, self.task.action_dim)
        else:
            q_values, asked,  got = self.get_final_q_values(state)
            self.intervalGot += got
            self.intervalAsked += asked
            action = np.argmax(q_values)
        return action


    def update(self, state, action, reward, next_state, done, deterministic):
        reward = self.config.reward_normalizer(reward)
        self.total_steps += 1
        if not deterministic:
            self.replay.feed([state, action, reward, next_state, int(done)])

        if self.total_steps % (self.config.test_interval * 25) == 0 and self.total_steps > 0:
            torch.save(self.network,self.task.name + "_"+str(self.total_steps)+".net")

        state = next_state
        if not deterministic and self.total_steps > self.config.exploration_steps \
                and self.total_steps % self.config.sgd_update_frequency == 0:
            experiences = self.replay.sample()
            states, actions, rewards, next_states, terminals = experiences
            states = self.config.state_normalizer(states)
            next_states = self.config.state_normalizer(next_states)
            q_next = self.target_network.predict(next_states, False).detach()
            if self.config.double_q:
                _, best_actions = self.network.predict(next_states).detach().max(1)
                q_next = q_next.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            else:
                q_next, _ = q_next.max(1)
            terminals = tensor(terminals)
            rewards = tensor(rewards)
            q_next = self.config.discount * q_next * (1 - terminals)
            q_next.add_(rewards)
            actions = tensor(actions).unsqueeze(1).long()
            q = self.network.predict(states, False)
            q = q.gather(1, actions).squeeze(1)
            loss = self.criterion(q, q_next)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            self.optimizer.step()

        self.evaluate()
        if not deterministic and self.total_steps % self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        if not deterministic and self.total_steps > self.config.exploration_steps:
            self.policy.update_epsilon()
