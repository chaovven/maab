from functools import partial
from envs.multiagentenv import MultiAgentEnv
from envs.auction_game.auction_game import AuctionGame
from envs.auction_alibaba.auction_alibaba import AuctionGameAli
import sys
import os


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["auction"] = partial(env_fn, env=AuctionGame)
REGISTRY["auction_ali"] = partial(env_fn, env=AuctionGameAli)