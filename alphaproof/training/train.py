import argparse
import gc
import json
import math
import uuid
from pathlib import Path
from typing import Any

import torch

from alphaproof.core.config import Config
from alphaproof.core.network import Network
from alphaproof.core.paths import RUNS_DIR
from alphaproof.training.actor_phase import run_actor_phase
from alphaproof.training.matchmaker import Matchmaker
from alphaproof.training.randomness import seed_everything
from alphaproof.training.replay_buffer import ReplayBuffer
from alphaproof.training.run_config import CONFIG_FILE, save_run_config
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

    for iteration in range(config.training_iterations):
        game_target = (iteration + 1) * games_per_iteration
        run_actor_phase(
            config,
            run_dir,
            resume,
            network,
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

    if step % config.checkpoint_interval != 0:
        storage.save_checkpoint(step, network)
    return network


def make_config(
    args: argparse.Namespace,
    saved_config: dict[str, Any] | None = None,
) -> Config:
    """Build the AlphaProof configuration from CLI arguments."""
    config = Config(
        num_simulations=args.num_simulations,
        batch_size=args.batch_size,
        dataset_dir=args.dataset_dir,
        sft_dataset_path=args.sft_dataset_path,
        sft_fraction=args.sft_fraction,
        disprove_rate=args.disprove_rate,
        num_actors=args.num_actors,
        num_games=args.num_games,
        inference_batch_size=args.inference_batch_size,
        inference_batch_timeout=args.inference_batch_timeout,
        seed=args.seed,
        debug=args.debug,
        lr=args.learning_rate,
        run_id=args.run_name,
        training_steps=args.training_steps,
        training_iterations=args.training_iterations,
        checkpoint_interval=args.checkpoint_interval,
        value_weight=args.value_weight,
    )
    if saved_config is not None:
        for name, value in saved_config.items():
            if name in (
                'dataset_dir',
                'train_dataset_path',
                'validation_dataset_path',
                'test_dataset_path',
                'sft_dataset_path',
                'sft_run_dir',
                'initial_params_path',
            ):
                value = Path(value) if value is not None else None
            setattr(config, name, value)
    return config


def positive_int(value: str) -> int:
    """Parse a positive integer."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError('value must be positive')
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse RL training arguments."""
    defaults = Config()
    parser = argparse.ArgumentParser(description='Train AlphaProof with RL.')
    parser.add_argument('run_name', help='Directory name under data/runs.')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', type=int, default=defaults.seed)
    parser.add_argument('--debug', action='store_true', default=defaults.debug)
    parser.add_argument(
        '--dataset-dir', type=Path, default=defaults.dataset_dir
    )
    parser.add_argument(
        '--sft-dataset-path', type=Path, default=defaults.sft_dataset_path
    )
    parser.add_argument(
        '--sft-fraction', type=float, default=defaults.sft_fraction
    )
    parser.add_argument(
        '--disprove-rate', type=float, default=defaults.mm_disprove_rate
    )
    parser.add_argument(
        '--num-simulations', type=positive_int, default=defaults.num_simulations
    )
    parser.add_argument(
        '--num-actors', type=positive_int, default=defaults.num_actors
    )
    parser.add_argument(
        '--num-games', type=positive_int, default=defaults.num_games
    )
    parser.add_argument(
        '--inference-batch-size',
        type=positive_int,
        default=defaults.inference_batch_size,
    )
    parser.add_argument(
        '--inference-batch-timeout',
        type=float,
        default=defaults.inference_batch_timeout,
    )
    parser.add_argument(
        '--batch-size', type=positive_int, default=defaults.batch_size
    )
    parser.add_argument('--learning-rate', type=float, default=defaults.lr)
    parser.add_argument(
        '--training-steps', type=positive_int, default=defaults.training_steps
    )
    parser.add_argument(
        '--training-iterations',
        type=positive_int,
        default=defaults.training_iterations,
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=positive_int,
        default=defaults.checkpoint_interval,
    )
    parser.add_argument('--value-weight', type=float, default=defaults.value_weight)
    parser.add_argument('--wandb-name')
    parser.add_argument(
        '--wandb-mode',
        choices=('online', 'offline', 'disabled'),
        default='disabled',
    )
    return parser.parse_args()


def prepare_run(
    args: argparse.Namespace,
) -> tuple[argparse.Namespace, Path, dict[str, Any] | None]:
    """Create a new run or restore its saved CLI configuration."""
    if Path(args.run_name).name != args.run_name:
        raise ValueError('run_name must be a single directory name.')
    run_dir = RUNS_DIR / args.run_name
    config_path = run_dir / CONFIG_FILE

    if args.resume:
        if not config_path.is_file():
            raise FileNotFoundError(f'Run configuration does not exist: {config_path}')
        with config_path.open(encoding='utf-8') as config_file:
            saved = json.load(config_file)
        saved_args = saved['args']
        saved_args['resume'] = True
        return argparse.Namespace(**saved_args), run_dir, saved['config']

    if run_dir.exists():
        raise FileExistsError(f'Run already exists: {run_dir}')
    args.wandb_run_id = uuid.uuid4().hex
    config = make_config(args)
    if config.sft_run_dir is None:
        raise ValueError('Set sft_run_dir in Config before starting RL.')
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
    if args.learning_rate <= 0:
        raise ValueError('--learning-rate must be positive.')
    if args.value_weight < 0:
        raise ValueError('--value-weight cannot be negative.')
    if args.inference_batch_timeout < 0:
        raise ValueError('--inference-batch-timeout cannot be negative.')
    if not 0 <= args.disprove_rate <= 1:
        raise ValueError('--disprove-rate must be between zero and one.')
    if not 0 < args.sft_fraction < 1:
        raise ValueError('--sft-fraction must be between zero and one.')
    if not math.isclose(
        args.batch_size * args.sft_fraction,
        round(args.batch_size * args.sft_fraction),
    ):
        raise ValueError(
            '--batch-size * --sft-fraction must be a whole number.'
        )

    run_dir.mkdir(parents=True)
    return args, run_dir, None


def main() -> None:
    """Run or resume AlphaProof reinforcement learning."""
    args, run_dir, saved_config = prepare_run(parse_args())
    config = make_config(args, saved_config)
    if not args.resume:
        save_run_config(run_dir, args, config)
    logger = RunLogger(
        run_dir,
        config.reward_window,
        initialize_wandb(args, config),
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
