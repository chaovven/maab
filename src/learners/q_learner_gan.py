import copy
from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop


class QLearnerGan:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        bs = batch.batch_size
        t_max = batch.max_seq_length
        n_agents = self.args.env_args['n_agents']
        n_actions = self.args.env_args['n_actions']

        rewards = batch["reward"][:, :-1].squeeze(-1)
        r_ind = rewards[:, :, :self.args.n_agents]  # with penality
        r_tot = rewards[:, :, -1:]
        actions = batch["actions"]
        action_v = batch["action_v"]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :, 0], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_action_qvals_v = th.gather(mac_out[:, :, 1], dim=3, index=action_v).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

        # Mask out unavailable actions
        avail_actions = avail_actions.unsqueeze(-3).repeat(1, 1, 2, 1, 1)
        target_mac_out[avail_actions == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=-1, keepdim=True)[1]

            remax_idx = (cur_max_actions[:, :, 0] < cur_max_actions[:, :, 1]).unsqueeze(-3).repeat(1, 1, 2, 1, 1).long()

            # constrain platform and bidders' actions
            action_index = th.Tensor([[[[_ for _ in range(n_actions)]]]]).repeat(bs, t_max, n_agents, 1).float()
            action_index = action_index.float().to(batch.device)
            max_actions = cur_max_actions[:, :, 0].repeat(1, 1, 1, n_actions)
            max_action_v = cur_max_actions[:, :, 1].repeat(1, 1, 1, n_actions)
            avail_actions[:, :, 1][action_index > max_actions.float()] = 0  # constrain platform's actions
            avail_actions[:, :, 0][action_index < max_action_v.float()] = 0  # constrain bidder's actions

            mac_out_detach[avail_actions == 0] = -9999999
            re_cur_max_actions = mac_out_detach.max(dim=-1, keepdim=True)[1]

            final_max_actions = cur_max_actions * (1 - remax_idx) + remax_idx * re_cur_max_actions

            target_max_qvals = th.gather(target_mac_out, -1, final_max_actions).squeeze(-1)

        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]  # TODO

        # Calculate 1-step Q-Learning targets
        targets0 = r_ind + self.args.gamma * (1 - terminated) * target_max_qvals[:, 1:, 0]
        targets1 = r_tot.repeat(1, 1, n_agents) + self.args.gamma * (1 - terminated) * target_max_qvals[:, 1:, 1]

        # Td-error
        td_error0 = (chosen_action_qvals[:, :-1] - targets0.detach())
        td_error1 = (chosen_action_qvals_v[:, :-1] - targets1.detach())

        mask0 = mask.expand_as(td_error0)
        mask1 = mask.expand_as(td_error1)

        # 0-out the targets that came from padded data
        masked_td_error0 = td_error0 * mask0
        masked_td_error1 = td_error1 * mask1

        # Normal L2 loss, take mean over actual data
        loss0 = (masked_td_error0 ** 2).sum() / mask0.sum()
        loss1 = (masked_td_error1 ** 2).sum() / mask1.sum()
        loss = loss0 + loss1

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
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs0", (masked_td_error0.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals[:, :-1] * mask0).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.logger.log_stat("target_mean", (targets0 * mask0).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
