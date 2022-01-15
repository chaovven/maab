from envs import REGISTRY as env_REGISTRY
from utils.rl_utils import softmax
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        self.n_agents = args.env_args['n_agents']

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        self.ps = [
            Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, **self.args.env_args))))
            for worker_conn in self.worker_conns]

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]
        self.auc_action = self.env_info["auc_action"]

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000
        self.log_action_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac, mac_v=None):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.mac_v = mac_v
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state"        : [],
            "avail_actions": [],
            "obs"          : []
        }
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        args = self.args
        self.reset()

        all_terminated = False
        if self.args.v_mode == 'rl' or args.v_mode == 'fix':
            episode_returns = [np.zeros(self.n_agents * 3 + 1) for _ in range(self.batch_size)]
        else:
            episode_returns = [np.zeros(self.n_agents * 2 + 1) for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        if self.args.v_mode == 'rl':
            self.mac_v.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated,
                                              test_mode=test_mode).cpu()
            if test_mode:
                action_v = th.zeros_like(actions)
            else:
                if self.args.v_mode == 'rl':
                    action_v = self.mac_v.select_actions(self.batch, t_ep=self.t, t_env=self.t_env,
                                                         test_mode=test_mode).cpu()
                elif args.v_mode == 'fix':
                    action_v = th.ones_like(actions) * self.args.v_threshold
                elif self.args.v_mode is None:
                    action_v = th.zeros_like(actions)
                else:
                    raise ValueError("v_mode not recognized error")

            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1),
            }
            if self.args.v_mode is not None:
                actions_chosen["action_v"] = action_v.unsqueeze(1)

            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:  # We produced actions for this env
                    if not terminated[idx]:  # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1  # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward"    : [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state"        : [],
                "avail_actions": [],
                "obs"          : []
            }

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()

                    rewards = self.build_rewards(actions[idx], action_v[idx], data["reward"])

                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((rewards,))

                    episode_returns[idx] += rewards
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))
                    post_transition_data["action_v"] = action_v

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos
        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        for i in range(self.args.env_args['n_agents']):  # platform's return
            self.logger.log_stat(prefix + f"penalized_return_mean{i}", np.mean([x[i] for x in returns]), self.t_env)
            self.logger.log_stat(prefix + f"return_mean{i}", np.mean([x[i + self.n_agents] for x in returns]),
                                 self.t_env)
        if self.args.v_mode is None:
            self.logger.log_stat(prefix + f"return_mean{self.n_agents}", np.mean([x[-1] for x in returns]),
                                 self.t_env)  # platform's return
        else:
            self.logger.log_stat(prefix + f"return_mean{self.n_agents}",
                                 np.mean([x[self.n_agents * 2] for x in returns]),
                                 self.t_env)  # platform's return
            for i in range(self.n_agents):
                self.logger.log_stat(prefix + f"return_mean_v{i}",
                                     np.mean([x[self.n_agents * 2 + i + 1] for x in returns]), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                if k != 'max_rewards':
                    self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
                else:
                    for i in range(self.args.env_args["n_agents"]):
                        self.logger.log_stat(prefix + k + "_mean", v[i] / stats["n_episodes"], self.t_env)
        stats.clear()

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
            rewards = np.concatenate((reward_train,  # 训练各个agent的reward
                                      np.array(rewards_env[self.n_agents:-1]),  # 各个agent实际的收益
                                      np.array([rewards_env[-1]]),  # 平台实际的收益
                                      r_v),  # 实际训练虚拟agent的reward
                                     axis=0)
        else:
            rewards = rewards_env
        return rewards


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state"        : state,
                "avail_actions": avail_actions,
                "obs"          : obs,
                # Rest of the data for the current timestep
                "reward"       : reward,
                "terminated"   : terminated,
                "info"         : env_info
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state"        : env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs"          : env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)
