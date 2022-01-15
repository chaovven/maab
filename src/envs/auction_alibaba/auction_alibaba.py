from envs.multiagentenv import MultiAgentEnv
import warnings
from copy import deepcopy
from utils.rl_utils import softmax
import numpy as np
from absl import logging
import pandas as pd
from sklearn import preprocessing

pd.options.mode.chained_assignment = None  # default='warn'


class AuctionGameAli(MultiAgentEnv):
    def __init__(self, **kwargs):
        self.columns = ["session_id", "adgroup_id", "pctr", "pcvr", "pdemand", "ocpc_option",
                        "time_stamp", "init_bid", "pctr_his", "pcvr_his", "pcfr_his", "pre_rankscore"]

        # self._preprocessing()

        # data
        self.train_data = pd.read_csv(kwargs["train_src"], delimiter=',', usecols=self.columns)
        self.test_data = pd.read_csv(kwargs["test_src"], delimiter=',', usecols=self.columns)

        self.goals = []
        for goal in [4, 1, 96]:
            goal_idx = self.train_data.groupby("ocpc_option").groups[goal]
            self.goals.append(set(self.train_data.iloc[goal_idx].groupby("adgroup_id").groups))

        # pointer
        self.train_data_ptr = 0
        self.test_data_ptr = 0
        self.n_agents = kwargs["n_agents"]
        self.episode_limit = kwargs["limit"]
        self.budget_ratio = kwargs["budget_ratio"]
        self.coop = kwargs["coop"]
        self.ratios = kwargs["ratios"]
        self.marl = kwargs["marl"]  # use multi-agent
        self.run_baseline = kwargs["run_baseline"]

        # Observations and state
        self.obs_last_win_rate = kwargs["obs_last_win_rate"]
        self.obs_instead_of_state = kwargs["obs_instead_of_state"]

        # Rewards args
        self.reward_death_value = kwargs["reward_death_value"]  # penalty agent who run out of budget too early
        self.reward_negative_scale = kwargs["reward_negative_scale"]
        self.reward_scale = kwargs["reward_scale"]
        self.reward_scale_rate = kwargs["reward_scale_rate"]

        # Other
        self._seed = kwargs["seed"]
        self.debug = kwargs["debug"]
        self.action_mapping = 5

        # actions
        self.n_actions = kwargs["n_actions"]
        self.auc_action = np.linspace(0, 1.0, self.n_actions)  # Note: Here raw action of the agent is alpha

        # build train and test data
        self.minute_level = kwargs["minute_level"] * 60
        min_stamp_train = self.train_data["time_stamp"].min()
        min_stamp_test = self.test_data["time_stamp"].min()
        self.train_data["timestep"] = ((self.train_data["time_stamp"] - min_stamp_train) // self.minute_level)
        self.test_data["timestep"] = ((self.test_data["time_stamp"] - min_stamp_test) // self.minute_level)

        pctr_idx = (self.train_data["ocpc_option"] == 4)
        pctr_pcvr_idx = (self.train_data["ocpc_option"] == 1)
        pctr_pdem_idx = (self.train_data["ocpc_option"] == 96)

        self.scaler_pctr = preprocessing.QuantileTransformer(output_distribution='normal').fit(
            np.array(self.train_data[pctr_idx]['pctr']).reshape(-1, 1))
        self.scaler_pctr_pcvr = preprocessing.QuantileTransformer(output_distribution='normal').fit(
            np.array(self.train_data[pctr_pcvr_idx]['pctr'] * self.train_data[pctr_pcvr_idx]['pcvr']).reshape(-1, 1))
        self.scaler_pctr_pdem = preprocessing.QuantileTransformer(output_distribution='normal').fit(
            np.array(self.train_data[pctr_pdem_idx]['pctr'] * self.train_data[pctr_pdem_idx]['pdemand']).reshape(-1, 1))

        # current episode's number
        self.cur_episode_train = 0
        self.cur_episode_test = 0
        self.n_episodes_train = self.train_data["timestep"].max() // self.episode_limit
        self.n_episodes_test = self.test_data["timestep"].max() // self.episode_limit

        self.train_data["n_episode"] = self.train_data["timestep"] // self.episode_limit
        self.test_data["n_episode"] = self.test_data["timestep"] // self.episode_limit

        # map info
        self.agents = {}
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self._obs = None
        self.last_action = np.zeros((self.n_agents, self.n_actions))

    def reset(self, test_mode=False):
        """Reset the environment. Required after each full episode.
        Returns initial observations and states.
        """
        self._episode_steps = 0

        self.episode_resources, self.ep_cost = self._get_episode_data(test_mode=test_mode)

        # information kept for counting the reward
        self.death_tracker_units = np.zeros(self.n_agents)

        self.last_action = np.zeros((self.n_agents, self.n_actions))

        self.init_units()

        if self.debug:
            logging.debug("Started Episode {}"
                          .format(self._episode_count).center(60, "*"))

        return self.get_obs(), self.get_state()

    def _get_episode_data(self, test_mode=False):
        episode_resources = []
        # we have different train_data & test_data
        if test_mode:
            # episode_idxs = self.train_data["timestep"] % self.episode_limit
            cur_epi_idx = (self.test_data["n_episode"] == self.cur_episode_test)
            self.cur_episode_test = (self.cur_episode_test + 1) % self.n_episodes_test
            data = self.test_data[cur_epi_idx]
        else:
            cur_epi_idx = (self.train_data["n_episode"] == self.cur_episode_train)
            self.cur_episode_train = (self.cur_episode_train + 1) % self.n_episodes_train
            data = self.train_data[cur_epi_idx]

        # normalized pvalue
        pctr_idx = (data["ocpc_option"] == 4)
        pctr_pcvr_idx = (data["ocpc_option"] == 1)
        pctr_pdem_idx = (data["ocpc_option"] == 96)
        self.pctr_norm = data[pctr_idx]['pctr'].sum()
        self.pctr_cvr_norm = (data[pctr_pcvr_idx]['pctr'] * data[pctr_pcvr_idx]['pcvr']).sum()
        self.pctr_pdem_norm = (data[pctr_pdem_idx]['pctr'] * data[pctr_pdem_idx]['pdemand']).sum()
        data["pctr_norm"] = data["pctr"] / self.pctr_norm * 100
        data["pctr_pcvr_norm"] = (data["pcvr"] * data["pctr"]) / self.pctr_cvr_norm * 100
        data["pctr_pdem_norm"] = (data["pdemand"] * data["pctr"]) / self.pctr_pdem_norm * 100
        #
        # transform
        if not test_mode:
            trans_pctr = self.scaler_pctr.transform(data["pctr"].values.reshape(-1, 1))
            trans_pctr_pcvr = self.scaler_pctr_pcvr.transform((data["pctr"] * data["pcvr"]).values.reshape(-1, 1))
            trans_pctr_pdem = self.scaler_pctr_pdem.transform((data["pctr"] * data["pdemand"]).values.reshape(-1, 1))
            data["pctr_norm"] = pd.DataFrame(trans_pctr, index=data.index)
            data["pctr_pcvr_norm"] = pd.DataFrame(trans_pctr_pcvr, index=data.index)
            data["pctr_pdem_norm"] = pd.DataFrame(trans_pctr_pdem, index=data.index)

        ep_cost = []
        for t in range(self.episode_limit):
            ads_t = data[((data["timestep"] % self.episode_limit) == t)]

            if ads_t.shape[0] == 0:
                t_delta = 1  # TODO
                while ads_t.shape[0] == 0:
                    t_delta += 1
                    ads_t = data[((data["timestep"] % self.episode_limit) == ((t + t_delta) % self.episode_limit))]

            ep_cost.append(ads_t.iloc[1::6]["pctr"].sum() * self.action_mapping)  # cost of a timestep t
            episode_resources.append(ads_t)

        episode_resources.append(episode_resources[0])  # extend one timestep for handling terminated episode
        return episode_resources, sum(ep_cost)

    def init_units(self):
        self.agents = {"traffic_value": self.episode_resources[0]}  # shared

        for i, ocpc_id, pvalue_id in [(0, 4, "pctr"), (1, 1, "pcvr"), (2, 96, "pdemand")]:
            idx = (self.agents['traffic_value']["ocpc_option"] == ocpc_id)
            mean_value_i = self.agents['traffic_value'][idx][pvalue_id].mean()
            if ocpc_id != 4:  # pcvr or demand
                mean_value_i = mean_value_i * self.agents['traffic_value'][idx]["pctr"].mean()
            self.agents[i] = {
                "init_bgt_ader" : pd.DataFrame.from_dict(dict.fromkeys(self.goals[i],
                                                                       self.ep_cost * 2 / self.budget_ratio),
                                                         orient='index'),
                "rem_bgt_ader"  : pd.DataFrame.from_dict(dict.fromkeys(self.goals[i],
                                                                       self.ep_cost * 2 / self.budget_ratio),
                                                         orient='index'),
                "initial_budget": self.ep_cost / self.budget_ratio * self.ratios[i],
                "remain_budget" : self.ep_cost / self.budget_ratio * self.ratios[i],
                "last_bid"      : 0,
                "last_win_rate" : 0,
                "n_good_bids"   : 0,
                "n_wins"        : 0,
                "mean_value"    : mean_value_i,
            }

    def step(self, actions, test_mode=False):
        """ Returns reward, terminated, info """
        if self.debug:
            logging.debug("".center(80, "-"))
            logging.debug("episode={}, t={}".format(self._episode_count, self._episode_steps).center(80, "-"))
            logging.debug("".center(80, "-"))
            logging.debug("Agent".center(80, "-"))
            logging.debug("b0: {:.3f}, b1: {:.3f}, b2: {:.3f}".format(self.auc_action[actions[0]],
                                                                      self.auc_action[actions[1]],
                                                                      self.auc_action[actions[2]]))

        actions_int = [int(a) for a in actions]

        self.last_action = np.eye(self.n_actions)[np.array(actions_int)]

        self._total_steps += 1
        self._episode_steps += 1

        # Update units
        if test_mode or self.marl:
            rewards = self.update_units(actions_int)  # update units' state after taking actions_int
        else:
            rewards = self.update_units_single_agent(actions_int)  # update units' state after taking actions_int

        terminated = False
        info = {}

        # count units that are still alive
        for i in range(self.n_agents):
            info[f"remain_budget_ratio{i}"] = self.agents[i]["remain_budget"] / self.agents[i]["initial_budget"]

        if self._episode_steps >= self.episode_limit:
            terminated = True
            for i in range(self.n_agents):
                info[f"n_wins{i}"] = self.agents[i]["n_wins"]

        return rewards, terminated, info

    def update_units(self, actions_int):
        # alpha set by RL agents
        joint_action = [self.auc_action[x] for x in actions_int]  # bid price with small fluctuation

        # initialize rewards
        rewards = np.zeros(self.n_agents + 1)  # individual rewards for n agents and joint reward of the platform
        train_rewards = np.zeros(self.n_agents)

        # log
        sess_set = set(self.agents["traffic_value"]["session_id"].unique())

        rankscore = []
        for i, ocpc_id, pvalue_id in [(0, 4, "pctr"), (1, 1, "pcvr"), (2, 96, "pdemand")]:
            agent_idx = (self.agents["traffic_value"]["ocpc_option"] == ocpc_id)

            if self.run_baseline:  # baseline: MSB
                bid = self.agents["traffic_value"][agent_idx]["init_bid"] * 0.01
            else:  # MARL
                pvalue = self.agents["traffic_value"][agent_idx][pvalue_id]  # pvalue
                if ocpc_id != 4:  # pcvr & pdemand
                    pvalue = pvalue * self.agents["traffic_value"][agent_idx]["pctr"]  # pcvr * pctr (pdemand * pctr)
                relative_pvalue = pvalue / self.agents[i]['mean_value']
                bid = relative_pvalue * joint_action[i] * self.action_mapping  # bid * pvalue / mean_pvalue
            rankscore.append(bid * self.agents["traffic_value"][agent_idx]["pctr"])  # bid * pctr
            # print(ocpc_id)
            # print(self.agents["traffic_value"][agent_idx]["pctr"].mean())
        self.agents["traffic_value"]["rankscore"] = pd.concat(rankscore)

        # identify winner and cost
        self.agents["traffic_value"] = self.agents["traffic_value"].groupby("session_id").apply(
            pd.DataFrame.sort_values, "rankscore", ascending=False)
        winner = self.agents["traffic_value"][::6]  # first one in a session
        cost = self.agents["traffic_value"][1::6]  # second one in a session
        cost.index = winner.index
        winner["cost"] = cost["rankscore"]

        winner_pctr = winner[(winner["ocpc_option"] == 4) & (winner["rankscore"] != 0)]["pctr_norm"]
        winner_pcvr = winner[(winner["ocpc_option"] == 1) & (winner["rankscore"] != 0)]["pctr_pcvr_norm"]
        winner_pdem = winner[(winner["ocpc_option"] == 96) & (winner["rankscore"] != 0)]["pctr_pdem_norm"]

        winners = [winner_pctr, winner_pcvr, winner_pdem]

        # update reward
        rewards[0] = winner_pctr.sum()
        rewards[1] = winner_pcvr.sum()
        rewards[2] = winner_pdem.sum()
        rewards[-1] = cost["rankscore"].sum()

        for i in range(self.n_agents):
            train_rewards[i] = rewards[i] / len(sess_set)

        # update unit's info
        self.agents["traffic_value"] = self.episode_resources[self._episode_steps]
        for i, ocpc_id, pvalue_id in [(0, 4, "pctr"), (1, 1, "pcvr"), (2, 96, "pdemand")]:
            self.agents[i]["remain_budget"] -= winner[winner["ocpc_option"] == ocpc_id]["cost"].sum()
            self.agents[i]["n_wins"] += winners[i].shape[0]
            self.agents[i]["last_win_rate"] = self.agents[i]["n_wins"] / len(sess_set) if len(sess_set) > 0 else 0
            self.agents[i]["last_bid"] = self.auc_action[actions_int[i]]

            # mean value
            idx = (self.agents['traffic_value']["ocpc_option"] == ocpc_id)
            mean_value_i = self.agents['traffic_value'][idx][pvalue_id].mean()  # mean pvalue
            if ocpc_id != 4:  # if pcvr or demand, multiplied mean_value_i by pctr
                mean_value_i = mean_value_i * self.agents['traffic_value'][idx]["pctr"].mean()
            self.agents[i]["mean_value"] = mean_value_i

        if self.coop != 0:
            train_rewards = softmax(np.array(joint_action), axis=0, temp=self.coop) * train_rewards.sum()
        rewards = np.concatenate((train_rewards, rewards), axis=0)

        if np.isnan(rewards.sum()):
            raise ValueError("reward is nan")

        if self.debug:
            logging.debug(
                "tr0: {:.3f}, tr1: {:.3f}, tr2: {:.3f}, r0: {:.3f}, r1: {:.3f}, r2: {:.3f}, r_plat: {:.3f}".format(
                    rewards[0], rewards[1], rewards[2],
                    rewards[3], rewards[4], rewards[5],
                    rewards[-1]))
        return rewards

    def update_units_single_agent(self, actions_int):
        # alpha set by RL agents
        joint_action = [self.auc_action[x] for x in actions_int]  # bid price with small fluctuation

        # initialize rewards
        rewards = np.zeros(self.n_agents + 1)  # individual rewards for n agents and joint reward of the platform
        train_rewards = np.zeros(self.n_agents)

        # log
        sess_set = set(self.agents["traffic_value"]["session_id"].unique())

        rankscore = []
        pre_rankscore = []
        for i, ocpc_id, pvalue_id in [(0, 4, "pctr"), (1, 1, "pcvr"), (2, 96, "pdemand")]:
            agent_idx = (self.agents["traffic_value"]["ocpc_option"] == ocpc_id)

            pvalue = self.agents["traffic_value"][agent_idx][pvalue_id]  # pvalue
            if ocpc_id != 4:  # pcvr & pdemand
                pvalue = pvalue * self.agents["traffic_value"][agent_idx]["pctr"]  # pcvr * pctr (pdemand * pctr)

            relative_pvalue = pvalue / self.agents[i]['mean_value']
            bid = relative_pvalue * joint_action[i] * 5  # bid * pvalue / mean_pvalue
            rankscore.append(bid * self.agents["traffic_value"][agent_idx]["pctr"])  # bid * pctr
            pre_rankscore.append(self.agents["traffic_value"][agent_idx]["init_bid"] *
                                 self.agents["traffic_value"][agent_idx]["pctr"] * 0.01)
        self.agents["traffic_value"]["rankscore"] = pd.concat(rankscore)
        self.agents["traffic_value"]["pre_rankscore"] = pd.concat(pre_rankscore)

        # for single RL
        self.agents["traffic_value"]["rankscore_pctr"] = self.agents["traffic_value"]["pre_rankscore"]
        self.agents["traffic_value"]["rankscore_pcvr"] = self.agents["traffic_value"]["pre_rankscore"]
        self.agents["traffic_value"]["rankscore_pdem"] = self.agents["traffic_value"]["pre_rankscore"]

        pctr_idx = (self.agents["traffic_value"]["ocpc_option"] == 4)
        pcvr_idx = (self.agents["traffic_value"]["ocpc_option"] == 1)
        pdem_idx = (self.agents["traffic_value"]["ocpc_option"] == 96)

        self.agents["traffic_value"].loc[pctr_idx, "rankscore_pctr"] = self.agents["traffic_value"][pctr_idx][
            "rankscore"]
        self.agents["traffic_value"].loc[pcvr_idx, "rankscore_pcvr"] = self.agents["traffic_value"][pcvr_idx][
            "rankscore"]
        self.agents["traffic_value"].loc[pdem_idx, "rankscore_pdem"] = self.agents["traffic_value"][pdem_idx][
            "rankscore"]

        # identify winner and cost
        sess_gb = self.agents["traffic_value"].groupby("session_id")
        self.agents["traffic_value_pctr"] = sess_gb.apply(
            pd.DataFrame.sort_values, "rankscore_pctr", ascending=False)
        self.agents["traffic_value_pcvr"] = sess_gb.apply(
            pd.DataFrame.sort_values, "rankscore_pcvr", ascending=False)
        self.agents["traffic_value_pdem"] = sess_gb.apply(
            pd.DataFrame.sort_values, "rankscore_pdem", ascending=False)

        winner_ito_pctr = self.agents["traffic_value_pctr"][::6]
        winner_ito_pcvr = self.agents["traffic_value_pcvr"][::6]
        winner_ito_pdem = self.agents["traffic_value_pdem"][::6]

        # build cost
        cost_ito_pctr = self.agents["traffic_value_pctr"][1::6]
        cost_ito_pcvr = self.agents["traffic_value_pcvr"][1::6]
        cost_ito_pdem = self.agents["traffic_value_pdem"][1::6]
        cost_ito_pctr.index = winner_ito_pctr.index
        cost_ito_pcvr.index = winner_ito_pcvr.index
        cost_ito_pdem.index = winner_ito_pdem.index
        winner_ito_pctr["cost"] = cost_ito_pctr["rankscore_pctr"]
        winner_ito_pcvr["cost"] = cost_ito_pcvr["rankscore_pcvr"]
        winner_ito_pdem["cost"] = cost_ito_pdem["rankscore_pdem"]

        winner_ito_list = [winner_ito_pctr, winner_ito_pcvr, winner_ito_pdem]

        winner_pctr = winner_ito_pctr[(winner_ito_pctr["ocpc_option"] == 4) &
                                      (winner_ito_pctr["rankscore_pctr"] != 0)]["pctr"]
        winner_pcvr = winner_ito_pcvr[(winner_ito_pcvr["ocpc_option"] == 1) & (winner_ito_pcvr["rankscore_pcvr"] != 0)]
        winner_pcvr = winner_pcvr["pcvr"] * winner_pcvr["pctr"]

        winner_pdem = winner_ito_pdem[(winner_ito_pdem["ocpc_option"] == 96) & (winner_ito_pdem["rankscore_pdem"] != 0)]
        winner_pdem = winner_pdem["pdemand"] * winner_pdem["pctr"]

        winners = [winner_pctr, winner_pcvr, winner_pdem]

        # update reward
        rewards[0] = winner_pctr.sum()
        rewards[1] = winner_pcvr.sum()
        rewards[2] = winner_pdem.sum()

        # train reward normalization
        for i in range(3):
            train_rewards[i] = rewards[i] / len(sess_set)

        # update unit's info
        self.agents["traffic_value"] = self.episode_resources[self._episode_steps]
        for i, ocpc_id, pvalue_id in [(0, 4, "pctr"), (1, 1, "pcvr"), (2, 96, "pdemand")]:
            self.agents[i]["remain_budget"] -= winner_ito_list[i][winner_ito_list[i]["ocpc_option"] == ocpc_id][
                'cost'].sum()
            self.agents[i]["n_wins"] += winners[i].shape[0]
            self.agents[i]["last_win_rate"] = self.agents[i]["n_wins"] / len(sess_set) if len(sess_set) > 0 else 0
            self.agents[i]["last_bid"] = self.auc_action[actions_int[i]]
            # update mean value
            idx = (self.agents['traffic_value']["ocpc_option"] == ocpc_id)
            mean_value_i = self.agents['traffic_value'][idx][pvalue_id].mean()  # mean pvalue
            if ocpc_id != 4:  # if pcvr or demand, multiplied mean_value_i by pctr
                mean_value_i = mean_value_i * self.agents['traffic_value'][idx]["pctr"].mean()
            self.agents[i]["mean_value"] = mean_value_i

        rewards = np.concatenate((train_rewards, rewards), axis=0)

        if self.debug:
            logging.debug(
                "tr0: {:.3f}, tr1: {:.3f}, tr2: {:.3f}, r0: {:.3f}, r1: {:.3f}, r2: {:.3f}, r_plat: {:.3f}".format(
                    rewards[0], rewards[1], rewards[2],
                    rewards[3], rewards[4], rewards[5],
                    rewards[-1]))
        return rewards

    def get_obs(self):
        """ Returns all agent observations in a list """
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        if np.isnan(np.concatenate(agents_obs).sum()):
            print("Error: agent_obs is nan")
        return agents_obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        unit = self.agents[agent_id]

        agent_obs = np.array([unit["remain_budget"] / unit["initial_budget"],
                              1 - self._episode_steps / self.episode_limit,
                              unit["mean_value"]])

        if self.obs_last_win_rate:
            agent_obs = np.append(unit["last_win_rate"])

        if self.debug:
            logging.debug("Obs Agent: {}".format(agent_id).center(60, "-"))
            logging.debug("Avail actions {}".format(
                self.get_avail_agent_actions(agent_id)))
            logging.debug("obs_rem_bgt: {:.3f}, obs_t_left: {:.3f}, obs_v: {:.3f}".format(agent_obs[0],
                                                                                          1 - self._episode_steps / self.episode_limit,
                                                                                          agent_obs[2]))
        return agent_obs

    def get_obs_size(self):
        """ Returns the shape of the observation """
        size = 3  # remain budget, timestep left, and mean impression value
        if self.obs_last_win_rate:
            size += 1
        return size

    def get_state(self):
        obs_concat = np.concatenate(self.get_obs(), axis=0).astype(np.float32)
        return obs_concat

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.get_obs_size() * self.n_agents

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
            if idx == 0:  # skip index 0
                continue
            if unit["remain_budget"] <= 0:
                avail_actions[idx] = 0
        return avail_actions

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
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

    def _preprocessing(self):
        col = ["session_id", "adgroup_id", "pctr", "pcvr", "pdemand", "ocpc_option",
               "time_stamp", "init_bid", "pctr_his", "pcvr_his", "pcfr_his", "pre_rankscore"]
        self.data_full = pd.read_csv(
            '/home/aaron/PycharmProjects/pymarl_auction/src/envs/auction_alibaba/data/large/kdd_20201118_full_test_filter.csv',
            # '/home/aaron/PycharmProjects/pymarl_auction/src/envs/auction_alibaba/data/large/kdd_data_20201118_member_id_full.csv',
            # '/home/aaron/PycharmProjects/pymarl_auction/src/envs/auction_alibaba/data/small/kdd_20201118_1m_filter.csv',
            delimiter=',', usecols=col)

        # data_gb = self.data_full.groupby("session_id")
        # selected_idx = np.random.choice(range(data_gb.ngroups), data_gb.ngroups // 30, replace=False)
        #
        # np_keys = np.array(list(data_gb.groups.keys()))
        # selected_key = list(np_keys[selected_idx])
        # self.data_30 = self.data_full[self.data_full["session_id"].isin(selected_key)]
        # self.data_30.to_csv('./kdd_20201118_5min_test.csv', index=False)

        # add pre_rankscore
        pctr_idx = (self.data["ocpc_option"] == 4)
        pcvr_idx = (self.data["ocpc_option"] == 1)
        pdem_idx = (self.data["ocpc_option"] == 96)

        pctr_norm = (self.data["pctr"] / self.data["pctr_his"] * self.data["pctr"]).clip(0, 2)[pctr_idx]
        pcvr_norm = (self.data["pcvr"] / self.data["pcvr_his"] * self.data["pctr"]).clip(0, 2)[pcvr_idx]
        pdem_norm = (self.data["pdemand"] / self.data["pcfr_his"] * self.data["pctr"]).clip(0, 2)[pdem_idx]

        pctr_norm[np.isinf(pctr_norm)] = 1
        pcvr_norm[np.isinf(pcvr_norm)] = 1
        pdem_norm[np.isinf(pdem_norm)] = 1

        pre_rankscore = pd.concat([pctr_norm * self.data["init_bid"][pctr_idx] * 0.01,
                                   pcvr_norm * self.data["init_bid"][pcvr_idx] * 0.01,
                                   pdem_norm * self.data["init_bid"][pdem_idx] * 0.01])
        self.data["pre_rankscore"] = pre_rankscore

        # delete session with only one row
        filt_data = []
        sess_gb = self.data.groupby("session_id")
        i = 0
        j = 0
        for sess_id, sess_data in sess_gb:
            j += 1
            if sess_data.shape[0] >= 2:
                sess_4 = sess_data[sess_data["ocpc_option"] == 4].nlargest(2, "pre_rankscore")
                sess_1 = sess_data[sess_data["ocpc_option"] == 1].nlargest(2, "pre_rankscore")
                sess_96 = sess_data[sess_data["ocpc_option"] == 96].nlargest(2, "pre_rankscore")
                if sess_4.shape[0] == 2 and sess_1.shape[0] == 2 and sess_96.shape[0] == 2:
                    sess_data_filt = pd.concat([sess_4, sess_1, sess_96]).sort_values("pre_rankscore", ascending=False)
                    filt_data.append(sess_data_filt)
                else:
                    i += 1
                    print('{}/{}:存在数量小于2'.format(i, j))
            else:
                print("H")
        self.data = pd.concat(filt_data)
        self.data.to_csv('./kdd_20201118_sampel05_filter.csv', index=False)
        print("finish deleting session")
