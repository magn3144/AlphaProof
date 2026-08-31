import argparse
import gc
import math
import uuid
from pathlib import Path

import torch

from alphaproof.core.config import (
    Config,
    RL_PRECISIONS,
    load_experiment_config,
    rl_config_from_dict,
)
from alphaproof.core.network import Network
from alphaproof.core.paths import RUNS_DIR
from alphaproof.inference.parallel import ParallelSearchEngine
from alphaproof.training.actor_phase import run_actor_phase
from alphaproof.training.matchmaker import Matchmaker
from alphaproof.training.randomness import seed_everything
from alphaproof.training.replay_buffer import ReplayBuffer
from alphaproof.training.run_config import (
    CONFIG_FILE,
    load_run_config,
    save_run_config,
)
from alphaproof.training.run_logger import RunLogger, initialize_wandb
from alphaproof.training.shared_storage import SharedStorage


REPLAY_FILE = 'replay_buffer.jsonl'


def train_network(
    config: Config,
    network: Network,
    storage: SharedStorage,
    replay_buffer: ReplayBuffer,
    start_step: int,
    num_steps: int,
    logger: RunLogger,
) -> int:
    """Run one learner phase and return the latest global step."""
    logger.log_learner_start(start_step, num_steps)
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
    total_games = config.num_actors * config.num_games
    if total_games % config.training_iterations != 0:
        raise ValueError('Actor games must be divisible by training iterations.')
    if config.training_steps % config.training_iterations != 0:
        raise ValueError('Training steps must be divisible by training iterations.')

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

    games_per_iteration = total_games // config.training_iterations
    steps_per_iteration = config.training_steps // config.training_iterations
    step = start_step

    with (
        ParallelSearchEngine(config, network) as engine,
        ParallelSearchEngine(config, network) as validation_engine,
    ):
        for iteration in range(config.training_iterations):
            game_target = (iteration + 1) * games_per_iteration
            run_actor_phase(
                config,
                run_dir,
                resume,
                engine,
                validation_engine,
                replay_buffer,
                matchmaker,
                logger,
                game_target,
            )

            step_target = (iteration + 1) * steps_per_iteration
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
                    logger,
                )
            if iteration + 1 < config.training_iterations:
                engine.resume()

    if step % config.checkpoint_interval != 0:
        storage.save_checkpoint(step, network)
    return network


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse RL training arguments."""
    parser = argparse.ArgumentParser(description='Train AlphaProof with RL.')
    parser.add_argument('run_name', help='Directory name under data/runs.')
    parser.add_argument('config_path', type=Path, nargs='?')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if Path(args.run_name).name != args.run_name:
        parser.error('run_name must be a single directory name')
    if args.resume and args.config_path is not None:
        parser.error('CONFIG.yaml must be omitted when resuming')
    if not args.resume and args.config_path is None:
        parser.error('CONFIG.yaml is required for a new run')
    if args.config_path is not None and not args.config_path.is_file():
        parser.error(f'experiment YAML does not exist: {args.config_path}')
    return args


def validate_config(config: Config) -> None:
    """Validate relationships in a resolved RL configuration."""
    positive = (
        'num_simulations',
        'batch_size',
        'num_actors',
        'num_games_per_actor',
        'inference_batch_size',
        'num_sampled_actions',
        'rollout_max_action_length',
        'training_steps',
        'training_iterations',
        'checkpoint_interval',
    )
    for name in positive:
        if getattr(config, name) < 1:
            raise ValueError(f'{name} must be positive.')
    if config.lr <= 0:
        raise ValueError('lr must be positive.')
    if config.value_weight < 0:
        raise ValueError('value_weight cannot be negative.')
    if config.inference_batch_timeout < 0:
        raise ValueError('inference_batch_timeout cannot be negative.')
    if not 0 <= config.disprove_rate <= 1:
        raise ValueError('disprove_rate must be between zero and one.')
    if not 0 < config.sft_fraction < 1:
        raise ValueError('sft_fraction must be between zero and one.')
    if config.dtype not in RL_PRECISIONS:
        raise ValueError(f'dtype must be one of {RL_PRECISIONS}.')
    if config.wandb_mode not in ('online', 'offline', 'disabled'):
        raise ValueError('wandb_mode must be online, offline, or disabled.')
    if not math.isclose(
        config.batch_size * config.sft_fraction,
        round(config.batch_size * config.sft_fraction),
    ):
        raise ValueError('batch_size * sft_fraction must be a whole number.')


def prepare_run(
    args: argparse.Namespace,
) -> tuple[Config, Path, str]:
    """Create a new run or restore its saved configuration."""
    run_dir = RUNS_DIR / args.run_name

    if args.resume:
        saved = load_run_config(run_dir)
        config = rl_config_from_dict(saved['config'], args.run_name)
        validate_config(config)
        return config, run_dir, saved['wandb_run_id']

    if (run_dir / CONFIG_FILE).exists():
        raise FileExistsError(f'Run already exists: {run_dir}')
    config = load_experiment_config(args.config_path, args.run_name).rl
    validate_config(config)
    if config.sft_run_dir is None:
        raise ValueError('Set sft_run_dir in the RL configuration.')
    for dataset_path in (
        config.train_dataset_path,
        config.validation_dataset_path,
        config.test_dataset_path,
    ):
        if not dataset_path.is_file():
            raise FileNotFoundError(
                f'Theorem dataset split does not exist: {dataset_path}'
            )
    if not config.sft_dataset_path.is_file():
        raise FileNotFoundError(
            f'SFT dataset does not exist: {config.sft_dataset_path}'
        )
    if not (config.sft_run_dir / 'model_source').is_dir():
        raise FileNotFoundError('SFT model_source directory does not exist.')
    if not (config.sft_run_dir / 'network_params.pt').is_file():
        raise FileNotFoundError('SFT network_params.pt does not exist.')
    run_dir.mkdir(parents=True, exist_ok=True)
    wandb_run_id = uuid.uuid4().hex
    save_run_config(run_dir, config, wandb_run_id)
    return config, run_dir, wandb_run_id


def main() -> None:
    """Run or resume AlphaProof reinforcement learning."""
    args = parse_args()
    config, run_dir, wandb_run_id = prepare_run(args)
    logger = RunLogger(
        run_dir,
        config.reward_window,
        initialize_wandb(
            args.run_name,
            run_dir,
            args.resume,
            wandb_run_id,
            config,
        ),
    )
    try:
        alphaproof_train(
            config,
            run_dir,
            args.resume,
            logger,
        )
    except BaseException as error:
        logger.log_crash(error)
        logger.finish(exit_code=1)
        raise
    logger.finish(exit_code=0)


if __name__ == '__main__':
    main()
