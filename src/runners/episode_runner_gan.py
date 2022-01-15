from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch as th


class EpisodeRunnerGan:

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

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = np.zeros(self.args.n_agents * 2 + 1)
        self.mac.init_hidden(batch_size=self.batch_size)

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

            reward, terminated, env_info = self.env.step(actions[0, 0])
            episode_return += reward

            post_transition_data = {
                "actions"   : actions[:, 0],
                "action_v"  : actions[:, 1],
                "reward"    : reward,
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

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
        self.batch.update({"actions": actions[:, 0]}, ts=self.t)
        self.batch.update({"actions": actions[:, 1]}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
            # log the info of each timestep across each episode
        if not test_mode and self.t_env - self.log_action_stats_t >= self.args.ep_log_interval:
            for t in range(self.args.env_args["limit"]):
                for i in range(self.env.n_agents):
                    self.logger.log_stat(f"tenv_{self.t_env}_obs{i}_rembudget",
                                         self.batch['obs'][0, t, i, 0] * self.env.agents[i]['initial_budget'], t)
                    self.logger.log_stat(f"tenv_{self.t_env}_obs{i}_v",
                                         self.batch['obs'][0, t, i, 2], t)
                    self.logger.log_stat(f"tenv_{self.t_env}_act{i}",
                                         self.env.auc_action[int(self.batch["actions"][0, t, i])], t)
                    self.logger.log_stat(f"tenv_{self.t_env}_rew{i}", self.batch["reward"][0, t, i], t)
                    self.logger.log_stat(f"tenv_{self.t_env}_act_v{i}",
                                         self.env.auc_action[int(self.batch["action_v"][0, t, i])], t)
                self.logger.log_stat(f"tenv_{self.t_env}_rew_v", int(self.batch["reward"][0, t, -1]), t)
            self.log_action_stats_t = self.t_env
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        for i in range(self.args.env_args['n_agents']):  # platform's return
            self.logger.log_stat(prefix + f"penalized_return_mean{i}", np.mean([x[i] for x in returns]), self.t_env)
            self.logger.log_stat(prefix + f"return_mean{i}", np.mean([x[i + self.args.n_agents] for x in returns]),
                                 self.t_env)
            # self.logger.log_stat(prefix + f"return_std{i}", np.std([x[i + self.args.n_agents] for x in returns]),
            #                      self.t_env)
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
