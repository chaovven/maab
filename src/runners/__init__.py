REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .episode_runner_gan import EpisodeRunnerGan
REGISTRY["episode_gan"] = EpisodeRunnerGan

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .episode_runner_ali import EpisodeRunnerAli
REGISTRY["episode_ali"] = EpisodeRunnerAli