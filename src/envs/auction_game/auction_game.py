from envs.multiagentenv import MultiAgentEnv
from copy import deepcopy
from utils.rl_utils import softmax
import numpy as np
from absl import logging


class AuctionGame(MultiAgentEnv):
    def __init__(self, **kwargs):
        self.n_agents = kwargs["n_agents"]
        self.episode_limit = kwargs["limit"]
        self.budget_ratio = kwargs["budget_ratio"]
        self.auction_type = kwargs["auction_type"]
        self.coop = kwargs["coop"]
        self.ratios = kwargs["ratios"]

        # Observations and state
        self.obs_last_win_rate = kwargs["obs_last_win_rate"]
        self.obs_instead_of_state = kwargs["obs_instead_of_state"]

        self.state_last_action = kwargs["state_last_action"]

        # Rewards args
        self.reward_death_value = kwargs["reward_death_value"]  # penalty agent who run out of budget too early
        self.reward_negative_scale = kwargs["reward_negative_scale"]
        self.reward_scale = kwargs["reward_scale"]
        self.reward_scale_rate = kwargs["reward_scale_rate"]

        # Other
        self._seed = kwargs["seed"]
        self.debug = kwargs["debug"]

        # actions
        self.n_actions = kwargs["n_actions"]
        self.auc_action = np.linspace(0, 1.0, self.n_actions)

        # budget ratios
        if self.n_agents == 2:
            self.ratios = [self.ratios, 1 - self.ratios]
        else:
            self.ratios = [1 / self.n_agents for _ in range(self.n_agents)]

        # map info
        self.max_rewards = None
        self.agents = {}
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self._obs = None
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        self.value_dist = kwargs["value_dist"]  # distribution of value
        self.slot_price = kwargs["slot_price"]

    def reset(self):
        """Reset the environment. Required after each full episode.
        Returns initial observations and states.
        """
        self._episode_steps = 0

        self.episode_resources = self._get_episode_data()
        self.traffic_owner = softmax(self.episode_resources, axis=1, temp=0.3)

        self.max_traffic_value = 1  # TODO vary across agents

        # information kept for counting the reward
        self.death_tracker_units = np.zeros(self.n_agents)
        self.previous_units = None

        self.last_action = np.zeros((self.n_agents, self.n_actions))

        self.init_units()

        if self.debug:
            logging.debug("Started Episode {}"
                          .format(self._episode_count).center(60, "*"))

        return self.get_obs(), self.get_state()

    def _get_episode_data(self):
        if self.value_dist == "random":
            # 流量分布随机
            episode_resources = np.random.uniform(low=0.0, high=1.0, size=[self.episode_limit + 1, self.n_agents])
        elif self.value_dist == "inverted":
            # 流量分布相反
            episode_resources = np.zeros((self.episode_limit + 1, self.n_agents))
            data = np.random.normal(0.5, 0.01, size=self.episode_limit + 1)
            flun = np.random.normal(0, 0.3, size=self.episode_limit + 1)

            episode_resources[:, 0] = np.clip(data - flun, 0.01, 0.99)
            episode_resources[:, 1] = np.clip(data + flun, 0.01, 0.99)
            assert self.n_agents == 2  # only support 2 agents scenario
        elif self.value_dist == "same":
            # 流量分布完全一致
            episode_resources = np.zeros((self.episode_limit + 1, self.n_agents))
            data = np.random.normal(0.5, 0.01, size=self.episode_limit + 1)
            for i in range(self.n_agents):
                episode_resources[:, i] = data
        elif self.value_dist == 'high_low':
            # 流量分布一高一低，scale不同
            episode_resources = np.zeros((self.episode_limit + 1, self.n_agents))
            data1 = np.random.normal(0.8, 0.01, size=self.episode_limit + 1)
            data2 = np.random.normal(0.2, 0.01, size=self.episode_limit + 1)
            episode_resources[:, 0] = np.clip(data1, 0.01, 0.99)
            episode_resources[:, 1] = np.clip(data2, 0.01, 0.99)
        else:
            raise ValueError("value_dist not recognized")

        episode_resources = episode_resources / episode_resources.sum() * (self.episode_limit + 1)
        return episode_resources

    def init_units(self):
        self.agents = {}

        for i in range(self.n_agents):
            self.agents[i] = {
                "initial_budget": self.episode_limit * (1 / self.budget_ratio) * self.ratios[i],
                "remain_budget" : self.episode_limit * (1 / self.budget_ratio) * self.ratios[i],
                "traffic_value" : self.episode_resources[0][i],  # pctr
                "last_bid"      : 0,
                "last_win_rate" : 0,
                "n_good_bids"   : 0,
                "n_wins"        : 0,
            }
            if self.debug:
                logging.debug(
                    "Unit {}, remain budget = {:.2f}, value = {:.4f}".format(
                        len(self.agents),
                        self.agents[i]["remain_budget"],
                        self.agents[i]["traffic_value"],
                    )
                )
        self.max_rewards = self.episode_resources[:-1].sum(axis=0)

    def step(self, actions):
        """ Returns reward, terminated, info """
        if self.debug:
            logging.debug("".center(60, "-"))
            logging.debug("episode={}, t={}".format(self._episode_count, self._episode_steps).center(60, "-"))
            logging.debug("".center(60, "-"))
            logging.debug("Agent".center(60, "-"))
            for i in range(self.n_agents):
                logging.debug("Unit {}, bid {}".format(i, self.auc_action[actions[i]]))

        actions_int = [int(a) for a in actions]

        self.last_action = np.eye(self.n_actions)[np.array(actions_int)]

        self._total_steps += 1
        self._episode_steps += 1

        # Update units
        rewards = self.update_units(actions_int)  # update units' state after taking actions_int

        terminated = False
        info = {}

        # count units that are still alive
        dead_units = 0
        for id, unit in self.agents.items():
            if unit["remain_budget"] < self.auc_action[1]:
                dead_units += 1
            info[f"remain_budget_ratio{id}"] = unit["remain_budget"] / unit["initial_budget"]

        if self._episode_steps >= self.episode_limit:
            terminated = True

            info["dead_units"] = dead_units
            info["max_rewards"] = self.max_rewards
            for id, unit in self.agents.items():
                info[f"total_good_bids{id}"] = unit["n_good_bids"]
                info[f"mean_good_bids{id}"] = unit["n_good_bids"] / (unit["n_wins"] + 1e-8)
                info[f"n_wins{id}"] = unit["n_wins"]

        return rewards, terminated, info

    def update_units(self, actions_int):
        """
        update each unit's status, including
        - remain_budget
        - traffic_value
        - last_win_rate
        """
        self.previous_units = deepcopy(self.agents)
        rewards = np.zeros(self.n_agents + 1)  # individual rewards for n agents plus a joint reward of the platform

        # all bidders' bid prices
        auc_actions = [self.auc_action[x] + float(np.random.rand(1)) * 1e-5 for x in
                       actions_int]  # bid price with small fluctuation
        max_price = max(auc_actions)  # max bid price

        value_won = 0
        # update units
        for a_id, action in enumerate(actions_int):
            win = (auc_actions[a_id] >= max_price and self.auc_action[actions_int[a_id]] > self.slot_price)
            if win:
                if self.auction_type == 'first':
                    price = self.auc_action[action]  # first pricce auction
                    self.agents[a_id]["remain_budget"] -= max(self.slot_price, price)
                elif self.auction_type == 'GSP':
                    price = np.sort(self.auc_action[actions_int])[-2]
                    self.agents[a_id]["remain_budget"] -= max(self.slot_price, price)  # GSP
                else:
                    raise ValueError("auction mechanism not recognized")
                rewards[a_id] = self.agents[a_id]["traffic_value"]
                value_won = self.agents[a_id]["traffic_value"]
                rewards[-1] = max(self.slot_price, price)  # platform's reward
                self.agents[a_id]["last_win_rate"] = 1.0
                self.agents[a_id]["n_good_bids"] += self.traffic_owner[self._episode_steps - 1][a_id]  # soft
                self.agents[a_id]["n_wins"] += 1
            else:
                self.agents[a_id]["last_win_rate"] = 0.0

            self.agents[a_id]["last_bid"] = self.auc_action[action]
            self.agents[a_id]["last_win_price"] = max_price  # winning price is only observed by winner
            self.agents[a_id]["traffic_value"] = self.episode_resources[self._episode_steps, a_id]

        if self.coop == 0:
            train_reward = rewards[:-1].copy()
        else:  # (co!=0)
            train_reward = softmax(np.array(auc_actions), axis=0, temp=self.coop) * value_won
        rewards = np.concatenate((train_reward, rewards), axis=0)

        return rewards

    def get_obs(self):
        """ Returns all agent observations in a list """
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        unit = self.agents[agent_id]

        agent_obs = np.array([unit["remain_budget"] / unit["initial_budget"],
                              1 - self._episode_steps / self.episode_limit,
                              unit["traffic_value"]])

        if self.obs_last_win_rate:
            agent_obs = np.append(unit["last_win_rate"])

        if self.debug:
            logging.debug("Obs Agent: {}".format(agent_id).center(60, "-"))
            logging.debug("Avail actions {}".format(
                self.get_avail_agent_actions(agent_id)))
            logging.debug(
                "obs_remain_budget: {:.3f}, obs_t_left: {:.3f}, obs_traffic_value: {:.3f}".format(agent_obs[0],
                                                                                                  1 - self._episode_steps / self.episode_limit,
                                                                                                  agent_obs[2]))
        return agent_obs

    def get_obs_size(self):
        """ Returns the shape of the observation """
        size = 3  # remain budget, timestep left and traffic_value
        if self.obs_last_win_rate:
            size += 1
        return size

    def get_state(self):
        """
        state consists of
        - remain budget
        - timestep left
        - value
        - last_action (optional)
        each dimension is normalized with its max value
        """
        if self.obs_instead_of_state:
            obs_concat = np.concatenate(self.get_obs(), axis=0).astype(
                np.float32
            )
            return obs_concat

        nf_al = 3
        state = np.zeros((self.n_agents, nf_al))
        for id, unit in self.agents.items():
            state[id, 0] = unit["remain_budget"] / unit["initial_budget"]
            state[id, 1] = 1 - self._episode_steps / self.episode_limit  # timesteps left
            state[id, 2] = unit["traffic_value"] / self.max_traffic_value

        state = state.flatten()

        if self.state_last_action:
            state = np.append(state, self.last_action.flatten())

        state = state.astype(dtype=np.float32)

        if self.debug:
            logging.debug("STATE".center(60, "-"))
            logging.debug("state (remain bgt, t left, v, last action(opt), win price(opt)): {}".format(state))
            if self.state_last_action:
                logging.debug("Last actions {}".format(self.last_action))

        return state

    def get_state_size(self):
        """ Returns the shape of the state"""
        if self.obs_instead_of_state:
            return self.get_obs_size() * self.n_agents

        size = 3 * self.n_agents  # remain budget, timesteps left and traffic_value
        if self.state_last_action:
            size += self.n_agents * self.n_actions
        return size

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        unit = self.agents[agent_id]
        avail_actions = [1] * self.n_actions

        for idx, action in enumerate(self.auc_action):
            if unit["remain_budget"] < action:
                avail_actions[idx] = 0
        return avail_actions

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return self.n_actions

    def get_env_info(self):
        env_info = {"state_shape"  : self.get_state_size(),
                    "obs_shape"    : self.get_obs_size(),
                    "n_actions"    : self.get_total_actions(),
                    "n_agents"     : self.n_agents,
                    "episode_limit": self.episode_limit,
                    "auc_action"   : self.auc_action,
                    }
        return env_info

    def get_stats(self):
        stats = {}
        return stats

    def close(self):
        print("Game over")
