import gc
from pathlib import Path

import torch

from alphaproof.core.config import Config
from alphaproof.core.network import Network
from alphaproof.inference.parallel import ParallelSearchEngine
from alphaproof.training.actor_phase import run_actor_phase
from alphaproof.training.matchmaker import Matchmaker
from alphaproof.training.randomness import seed_everything
from alphaproof.training.replay_buffer import ReplayBuffer
from alphaproof.training.run_logger import RunLogger
from alphaproof.training.shared_storage import SharedStorage


REPLAY_FILE = 'replay_buffer.jsonl'


def train_network(
    config: Config,
    network: Network,
    storage: SharedStorage,
    replay_buffer: ReplayBuffer,
    start_step: int,
    num_steps: int,
    transition_target: int,
    logger: RunLogger,
) -> int:
    """Run one learner phase and return the latest global step."""
    logger.log_learner_start(start_step, num_steps, transition_target)
    validation_batch = replay_buffer.validation_batch(
        config.validation_batch_size
    )
    step = start_step
    for _ in range(num_steps):
        step += 1
        train_loss = None
        oom_message = ''
        try:
            train_loss = network.update(replay_buffer.sample_batch())
        except torch.OutOfMemoryError as error:
            oom_message = str(error)

        if train_loss is None:
            network.optimizer.zero_grad(set_to_none=True)
            gc.collect()
            if network.device.type == 'cuda':
                torch.cuda.empty_cache()
            print(
                f'WARNING: OOM in learner step {step}; skipped the batch and '
                f'cleared the CUDA cache. {oom_message}',
                flush=True,
            )
            continue

        validation_loss = None
        if validation_batch and step % config.validation_interval == 0:
            validation_loss = network.evaluate(validation_batch)
        if step % config.log_interval == 0 or validation_loss is not None:
            logger.log_training(
                step,
                train_loss,
                validation_loss,
                len(replay_buffer),
            )
        if step % config.checkpoint_interval == 0:
            storage.save_checkpoint(step, network)

    if network.device.type == 'cuda':
        torch.cuda.empty_cache()
    return step


def alphaproof_train(
    config: Config,
    run_dir: Path,
    resume: bool,
    logger: RunLogger,
) -> Network:
    """Coordinate resumable actor jobs and learner updates."""
    print(f'Training seed: {config.seed}', flush=True)
    seed_everything(config.seed)

    storage = SharedStorage(run_dir)
    replay_buffer = ReplayBuffer(config, run_dir / REPLAY_FILE)
    matchmaker = Matchmaker(config)
    network = Network(config)

    if resume:
        start_step = storage.load_latest_checkpoint(network)
    else:
        if config.initial_params_path is None:
            raise ValueError('An SFT run is required for a new RL run.')
        network.load_params(config.initial_params_path)
        start_step = 0
        storage.save_checkpoint(start_step, network)
        storage.save_latest_checkpoint(start_step, network)

    num_iterations = (
        config.training_steps + config.training_steps_per_iteration - 1
    ) // config.training_steps_per_iteration
    transition_target = 0
    step = start_step

    with (
        ParallelSearchEngine(config, network) as engine,
        ParallelSearchEngine(config, network) as validation_engine,
    ):
        for iteration in range(num_iterations):
            transition_target += config.transitions_per_iteration
            run_actor_phase(
                config,
                run_dir,
                resume,
                engine,
                validation_engine,
                replay_buffer,
                matchmaker,
                logger,
                transition_target,
            )

            step_target = min(
                (iteration + 1) * config.training_steps_per_iteration,
                config.training_steps,
            )
            steps_to_run = step_target - step
            if (
                steps_to_run > 0
                and len(replay_buffer) >= replay_buffer.replay_batch_size
            ):
                step = train_network(
                    config,
                    network,
                    storage,
                    replay_buffer,
                    step,
                    steps_to_run,
                    transition_target,
                    logger,
                )
                storage.save_latest_checkpoint(step, network)
    if step % config.checkpoint_interval != 0:
        storage.save_checkpoint(step, network)
    return network
