import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.qmix import IndQMixer
import torch as th
from torch.optim import RMSprop


class QLearnerV:
    def __init__(self, mac, scheme, logger, args, mac_v=None):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = IndQMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.mac_v = mac_v
        self.params_v = list(mac_v.parameters())
        self.optimiser_v = RMSprop(params=self.params_v, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.target_mac_v = copy.deepcopy(mac_v)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1].squeeze(-1)
        r_ind = rewards[:, :, :self.args.n_agents]  # with penality
        actions = batch["actions"]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        loss_v = self.train_v(batch, t_env, episode_num)  # SHOULD DELAY UPDATING

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out, dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals_ext = chosen_action_qvals.unsqueeze(-2).repeat(1, 1, self.args.n_agents, 1)
            target_max_qvals_ext = chosen_action_qvals.unsqueeze(-2).repeat(1, 1, self.args.n_agents, 1)
            for i in range(self.args.n_agents):
                target_max_qvals_ext[:, :, i, i] = target_max_qvals[:, :, i]

            chosen_action_qvals = self.mixer(chosen_action_qvals_ext, batch)
            target_max_qvals = self.target_mixer(target_max_qvals_ext, batch)

        # Calculate 1-step Q-Learning targets
        targets = r_ind + self.args.gamma * (1 - terminated) * target_max_qvals[:, 1:]

        # Td-error
        td_error = (chosen_action_qvals[:, :-1] - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("loss_v", loss_v.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            # self.logger.log_stat("q_taken_mean",
            #                      (chosen_action_qvals[:, :-1] * mask).sum().item() / (mask_elems * self.args.n_agents),
            #                      t_env)
            # self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
            #                      t_env)
            self.log_stats_t = t_env

    def train_v(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        r_v = batch["reward"][:, :-1, -self.args.n_agents:]
        action_v = batch["action_v"]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        # Calculate estimated Q-Values
        mac_out_v = []

        self.mac_v.init_hidden(batch.batch_size)

        for t in range(batch.max_seq_length):
            agent_outs_v = self.mac_v.forward(batch, t=t)
            mac_out_v.append(agent_outs_v)

        mac_out_v = th.stack(mac_out_v, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals_v = th.gather(mac_out_v, dim=3, index=action_v).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out_v = []

        self.target_mac_v.init_hidden(batch.batch_size)

        for t in range(batch.max_seq_length):
            target_agent_outs_v = self.target_mac_v.forward(batch, t=t)
            target_mac_out_v.append(target_agent_outs_v)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out_v = th.stack(target_mac_out_v, dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out_v[avail_actions == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach_v = mac_out_v.clone().detach()
            mac_out_detach_v[avail_actions == 0] = -9999999
            cur_max_actions_v = mac_out_detach_v.max(dim=3, keepdim=True)[1]
            target_max_qvals_v = th.gather(target_mac_out_v, 3, cur_max_actions_v).squeeze(3)
        else:
            target_max_qvals_v = target_mac_out_v.max(dim=3)[0]

        target_v = r_v + self.args.gamma * (1 - terminated) * target_max_qvals_v[:, 1:]
        td_error_v = chosen_action_qvals_v[:, :-1] - target_v.detach()
        mask_v = mask.expand_as(td_error_v)
        masked_td_error_v = td_error_v * mask_v
        loss_v = (masked_td_error_v ** 2).sum() / mask_v.sum()

        # Optimise virtual agent
        self.optimiser_v.zero_grad()
        loss_v.backward()
        grad_norm_v = th.nn.utils.clip_grad_norm_(self.params_v, self.args.grad_norm_clip)
        self.optimiser_v.step()

        return loss_v

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_mac_v.load_state(self.mac_v)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.mac_v.cuda()
        self.target_mac.cuda()
        self.target_mac_v.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        self.mac_v.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        th.save(self.optimiser_v.state_dict(), "{}/opt_v.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.mac_v.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.target_mac_v.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser_v.load_state_dict(th.load("{}/opt_v.th".format(path), map_location=lambda storage, loc: storage))
