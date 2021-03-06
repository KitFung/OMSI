import numpy as np
from collections import defaultdict

MIN_PULL_FOR_ACC = 200

class Policy(object):
    # Select Max
    def choose(self, agent):
        m = np.zeros(agent.n, dtype=bool)
        m[agent.avail_idx] = True
        indices = np.ma.array(agent.indices, mask=m)
        action = np.argmax(indices)
        check = np.where(indices == indices[action])[0]
        if len(check) == 1:
            return action
        else:
            return np.random.choice(check)

# Select Discount UCB as our algorithm
class NonStationaryBanditAgent(object):
    def __init__(self, policy, n_option, option_names):
        self.policy = policy
        self.n = n_option
        assert len(option_names) == n_option
        self.option_name_map = {name: i for i, name in enumerate(option_names)}
        self.avail_idx = list(range(n_option))

        self.step = 0
        self.discount_factor = 0.8
        self.confidence_scale = 0.5
        self.cum_reward = np.zeros(n_option)
        self.disc_cum_reward = np.zeros(n_option)
        self.pull_cnt = np.zeros(n_option)
        self.disc_cnt = np.zeros(n_option)
        self.indices = np.zeros(n_option)

        self.cache_disc_cum_reward = defaultdict(float)
        self.cache_disc_cnt = defaultdict(float)
        self.cache_indices = defaultdict(float)

        self.join_step = defaultdict(int)
        self.explore_threshold = 1000

    def set_explore_threshold(self, explore_threshold):
        self.explore_threshold = explore_threshold

    def print_status(self):
        print("indices:")
        print(self.indices)
        rewards = self.cum_reward / self.pull_cnt
        print("rewards:")
        print(rewards)
        print(self.option_name_map)
        print('-------')

    def add_option(self, option_name):
        self.print_status()

        if len(self.avail_idx) == 0:
            print("Cannot add option since no place leave")
            return False
        print("Add option %s" % option_name)
        self.join_step[option_name] = self.step
        idx = self.avail_idx.pop()

        self.option_name_map[option_name] = idx
        self.cum_reward[idx] = 0
        self.disc_cum_reward[idx] = self.cache_disc_cum_reward[option_name]
        self.pull_cnt[idx] = 0
        self.disc_cnt[idx] = self.cache_disc_cnt[option_name]
        self.indices[idx] = self.cache_indices[option_name]
        return True

    def kick_option(self, option_name):
        self.print_status()

        target = self.option_name_map[option_name]
        self.cache_disc_cum_reward[option_name] = self.disc_cum_reward[target]
        self.cache_disc_cnt[option_name] = self.disc_cnt[target]
        self.cache_indices[option_name] = self.indices[target]

        self.cum_reward[target] = 0
        self.disc_cum_reward[target] = 0
        self.pull_cnt[target] = 0
        self.disc_cnt[target] = 0
        self.indices[target] = 0
        print("Remove option %s" % option_name)
        del self.option_name_map[option_name]
        self.avail_idx.append(target)

    def choose(self):
        return self.policy.choose(self)

    def observe(self, name, reward):
        target = self.option_name_map[name]
        self.cum_reward[target] += reward
        self.pull_cnt[target] += 1.0
        self.step += 1
        self.disc_cum_reward[:] *= self.discount_factor
        self.disc_cum_reward[target] += reward
        self.disc_cnt[:] *= self.discount_factor
        self.disc_cnt[target] += 1.0
        self.indices = self.disc_cum_reward / self.disc_cnt + 2 * np.sqrt(
            self.confidence_scale * np.log(self.disc_cnt.sum()) / self.disc_cnt)

    def initialized(self):
        return len(self.pull_cnt[self.pull_cnt == 0]) == 0

    def vote_the_worst(self):
        rewards = self.cum_reward / self.pull_cnt
        mean_reward = np.mean(rewards[self.pull_cnt > MIN_PULL_FOR_ACC])
        bad_ind = (self.pull_cnt > MIN_PULL_FOR_ACC) & (rewards < mean_reward)
        for name, idx in self.option_name_map.items():
            if bad_ind[idx] and self.step - self.join_step[name] > self.explore_threshold:
                return idx
        return None
