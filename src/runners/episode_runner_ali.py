from envs import REGISTRY as env_REGISTRY
from utils.rl_utils import softmax
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch as th


class EpisodeRunnerAli:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.env_info = self.env.get_env_info()
        self.auc_action = self.env_info["auc_action"]
        self.n_agents = args.env_args['n_agents']
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000
        self.log_action_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac, mac_v=None):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.mac_v = mac_v

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self, test_mode):
        self.batch = self.new_batch()
        self.env.reset(test_mode)
        self.t = 0

    def run(self, test_mode=False):
        self.reset(test_mode)

        terminated = False
        if self.args.v_mode == 'rl' or self.args.v_mode == 'fix':
            episode_returns = np.zeros(self.args.n_agents * 3 + 1)
        else:
            episode_returns = np.zeros(self.args.n_agents * 2 + 1)  # for 'None'
        self.mac.init_hidden(batch_size=self.batch_size)
        if self.args.v_mode == 'rl':
            self.mac_v.init_hidden(batch_size=self.batch_size)

        while not terminated:
            pre_transition_data = {
                "state"        : [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs"          : [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode).cpu()

            if test_mode:
                action_v = th.zeros_like(actions)
            else:
                if self.args.v_mode == 'rl':
                    action_v = self.mac_v.select_actions(self.batch, t_ep=self.t, t_env=self.t_env,
                                                         test_mode=test_mode).cpu()
                elif self.args.v_mode == 'fix':
                    action_v = th.ones_like(actions, dtype=th.float) * self.args.v_threshold
                elif self.args.v_mode is None:
                    action_v = th.zeros_like(actions)
                else:
                    raise ValueError("v_mode not recognized error")

            rewards, terminated, env_info = self.env.step(actions[0], test_mode)

            rewards = self.build_rewards(actions[0], action_v[0], rewards)

            episode_returns += rewards

            post_transition_data = {
                "actions"   : actions,
                "reward"    : rewards,
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            post_transition_data["action_v"] = action_v

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state"        : [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs"          : [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)
        if self.args.v_mode == 'rl':
            if test_mode == False:
                action_v = self.mac_v.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            else:
                action_v = th.zeros_like(actions)
            self.batch.update({"action_v": action_v}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_returns)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            if self.args.v_mode == 'rl':
                self.logger.log_stat("epsilon_v", self.mac_v.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env
        # log the info of each timestep across each episode
        if not test_mode and self.t_env - self.log_action_stats_t >= self.args.ep_log_interval:
            for t in range(self.args.env_args["limit"]):
                for i in range(self.env.n_agents):
                    self.logger.log_stat(f"tenv_{self.t_env}_obs{i}_rembudget",
                                         self.batch['obs'][0, t, i, 0] * self.env.agents[i]['initial_budget'],
                                         t)
                    self.logger.log_stat(f"tenv_{self.t_env}_obs{i}_v",
                                         self.batch['obs'][0, t, i, 2], t)
                    self.logger.log_stat(f"tenv_{self.t_env}_act{i}",
                                         self.env.auc_action[int(self.batch["actions"][0, t, i])], t)
                    self.logger.log_stat(f"tenv_{self.t_env}_rew{i}", self.batch["reward"][0, t, i], t)
                    if self.args.v_mode is not None:
                        if self.args.v_mode == 'fix':
                            self.logger.log_stat(f"tenv_{self.t_env}_act_v{i}", self.batch["action_v"][0, t, i], t)
                        else:
                            self.logger.log_stat(f"tenv_{self.t_env}_act_v{i}",
                                                 self.env.auc_action[int(self.batch["action_v"][0, t, i])], t)
                        self.logger.log_stat(f"tenv_{self.t_env}_rew_v{i}",
                                             self.batch["reward"][0, t][-self.args.n_agents:][i], t)
            self.log_action_stats_t = self.t_env

        return self.batch

    def build_rewards(self, actions, action_v, rewards_env):
        # advertiser's bid
        bids = self.auc_action[actions]

        if self.args.v_mode is not None:
            # virtual agent's bid
            if self.args.v_mode == 'rl':
                bids_v = self.auc_action[np.array(action_v)]
            elif self.args.v_mode == 'fix':
                bids_v = np.array(action_v)

            use_reward = np.array(bids >= bids_v)
            reward_train = use_reward * rewards_env[: self.n_agents]

            # softmax virtual agent's reward
            # softmax_r_v = softmax(bids, axis=0, temp=1e3) * rewards_env[-1]
            r_v = use_reward * rewards_env[-1] / use_reward.size
            rewards = np.concatenate((reward_train,
                                      np.array(rewards_env[self.n_agents:-1]),
                                      np.array([rewards_env[-1]]),
                                      r_v),
                                     axis=0)
        else:
            rewards = rewards_env
        return rewards

    def _log(self, returns, stats, prefix):
        for i in range(self.args.env_args['n_agents']):  # platform's return
            self.logger.log_stat(prefix + f"penalized_return_mean{i}", np.mean([x[i] for x in returns]), self.t_env)
            self.logger.log_stat(prefix + f"return_mean{i}", np.mean([x[i + self.args.n_agents] for x in returns]),
                                 self.t_env)
        # social welfare
        self.logger.log_stat(prefix + f"return_mean_sw",
                             np.mean(np.sum([x[self.args.n_agents:self.args.n_agents + 3] for x in returns], axis=1)),
                             self.t_env)
        if self.args.v_mode is None:
            self.logger.log_stat(prefix + f"return_mean{self.args.n_agents}", np.mean([x[-1] for x in returns]),
                                 self.t_env)  # platform's return
        else:
            self.logger.log_stat(prefix + f"return_mean{self.args.n_agents}",
                                 np.mean([x[self.args.n_agents * 2] for x in returns]),
                                 self.t_env)  # platform's return
            for i in range(self.args.n_agents):
                self.logger.log_stat(prefix + f"return_mean_v{i}",
                                     np.mean([x[self.args.n_agents * 2 + i + 1] for x in returns]), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                if k != 'max_rewards':
                    self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
                else:
                    for i in range(self.args.env_args["n_agents"]):
                        self.logger.log_stat(prefix + k + "_mean", v[i] / stats["n_episodes"], self.t_env)
        stats.clear()
