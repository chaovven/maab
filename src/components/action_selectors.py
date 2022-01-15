import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule

REGISTRY = {}


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        if self.args.run_baseline:
            masked_q_values[:, :, 10] = 99999999  # should never be selected!
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector


class EpsilonGreedyActionSelectorAli():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        avail_actions_rep = avail_actions.unsqueeze(-2).repeat(1, 1, self.args.env_args["top_n"], 1)
        avail_actions_rep = avail_actions_rep.view(masked_q_values.shape)
        masked_q_values[avail_actions_rep == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        try:
            random_actions = Categorical(avail_actions_rep.float()).sample().long()
        except:
            print("H")

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["epsilon_greedy_ali"] = EpsilonGreedyActionSelectorAli


class ConstrainedEpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)
        bs = agent_inputs.shape[0]

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        avail_actions = avail_actions.unsqueeze(-3).repeat(1, 2, 1, 1)
        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=-1)[1]

        repick = (picked_actions[:, 0] < picked_actions[:, 1]).unsqueeze(-2).repeat(1, 2, 1)

        # constrain bidders' actions and platform's action
        action_index = th.Tensor([[_ for _ in range(20)]]).unsqueeze(0).repeat(bs, self.args.env_args["n_agents"],
                                                                               1).float().to(picked_actions.device)
        picked_actions_bidder = picked_actions[:, 0].unsqueeze(-1).repeat(bs, 1, 20).float()
        picked_actions_plat = picked_actions[:, 1].unsqueeze(-1).repeat(bs, 1, 20).float()
        avail_actions[:, 1][action_index > picked_actions_bidder] = 0  # constrain platform's actions
        avail_actions[:, 0][action_index < picked_actions_plat] = 0  # constrain bidder's actions

        masked_q_values[avail_actions == 0] = -float("inf")

        random_numbers = th.rand_like(agent_inputs[:, :, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        repicked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=-1)[1]
        final_actions = repicked_actions * repick.long() + picked_actions * (1 - repick.long())

        assert int((final_actions[:, 0] < final_actions[:, 1]).sum()) == 0

        return final_actions


REGISTRY["constrained_epsilon_greedy"] = ConstrainedEpsilonGreedyActionSelector
