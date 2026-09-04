import argparse
import json
from pathlib import Path
from typing import Any, cast

import torch

from alphaproof.core.config import (
    DEFAULT_EXPERIMENT_PATH,
    Config,
    load_experiment_config,
    rl_config_from_dict,
    serializable_config,
    sft_config_from_dict,
)
from alphaproof.core.environment import Environment
from alphaproof.core.game import extract_proof_script
from alphaproof.core.network import Network, Params
from alphaproof.core.paths import LEAN_PROJECT_DIR
from alphaproof.inference.parallel import ParallelSearchEngine, SearchRequest
from alphaproof.inference.search_tree import serialize_search_tree
from alphaproof.training.randomness import seed_everything
from alphaproof.training.run_config import load_run_config
from leantree import LeanProject


def make_config(args: argparse.Namespace) -> Config:
    """Build search configuration for an SFT or RL run."""
    run_data = load_run_config(args.run_dir)
    saved_config = run_data['config']
    if 'num_simulations' in saved_config:
        config = rl_config_from_dict(saved_config, args.run_dir.name)
    else:
        sft_config = sft_config_from_dict(saved_config)
        values = serializable_config(
            load_experiment_config(DEFAULT_EXPERIMENT_PATH).rl
        )
        values.update(
            {
                'lr': sft_config.learning_rate,
                'sft_run_dir': str(args.run_dir),
                'max_state_length': sft_config.max_state_length,
                'max_action_length': sft_config.max_action_length,
                'rollout_max_action_length': (
                    sft_config.rollout_max_action_length
                ),
                'num_value_bins': sft_config.num_value_bins,
            }
        )
        config = rl_config_from_dict(values, args.run_dir.name)
    config.num_simulations = args.num_simulations
    config.batch_size = 1
    config.num_actors = args.parallel_searches
    config.num_games = 1
    config.num_games_per_actor = 1
    config.inference_batch_size = args.inference_batch_size
    config.inference_batch_timeout = args.inference_batch_timeout
    config.num_sampled_actions = args.num_sampled_actions
    config.tactic_timeout = args.tactic_timeout
    config.seed = args.seed
    return config


def load_network_checkpoint(run_dir: Path, network: Network) -> Path:
    """Load the latest RL checkpoint or the SFT network parameters."""
    checkpoints = sorted((run_dir / 'checkpoints').glob('step_*.pt'))
    if checkpoints:
        checkpoint_path = checkpoints[-1]
        checkpoint = torch.load(
            checkpoint_path,
            map_location='cpu',
            weights_only=True,
        )
        checkpoint = cast(dict[str, Any], checkpoint)
        network.params = cast(Params, checkpoint['network_params'])
        return checkpoint_path

    checkpoint_path = run_dir / 'network_params.pt'
    network.load_params(checkpoint_path)
    return checkpoint_path


def theorem_text(record: dict[str, Any]) -> str:
    """Combine an optional Lean header with a theorem request."""
    header = str(record.get('header') or '')
    return f'{header}\n{record["theorem"]}' if header else str(record['theorem'])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse paths and shared settings for JSONL inference."""
    defaults = load_experiment_config(DEFAULT_EXPERIMENT_PATH).rl
    default_run_dir = defaults.sft_run_dir
    parser = argparse.ArgumentParser(
        description='Search for verified Lean proofs from a JSONL batch.'
    )
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--run-dir',
        type=Path,
        default=default_run_dir,
        help=(
            'SFT or RL run containing trained network parameters '
            f'(default: {default_run_dir}).'
        ),
    )
    parser.add_argument('--lean-project', type=Path, default=LEAN_PROJECT_DIR)
    parser.add_argument('--import', dest='imports', action='append')
    parser.add_argument(
        '--num-simulations', type=int, default=defaults.num_simulations
    )
    parser.add_argument(
        '--num-sampled-actions', type=int, default=defaults.num_sampled_actions
    )
    parser.add_argument(
        '--tactic-timeout', type=float, default=defaults.tactic_timeout
    )
    parser.add_argument(
        '--parallel-searches', type=int, default=defaults.num_actors
    )
    parser.add_argument(
        '--inference-batch-size',
        type=int,
        default=defaults.inference_batch_size,
    )
    parser.add_argument(
        '--inference-batch-timeout',
        type=float,
        default=defaults.inference_batch_timeout,
    )
    parser.add_argument('--seed', type=int, default=defaults.seed)
    parser.add_argument(
        '--stop-on-solution',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args(argv)

    if not args.input.is_file():
        parser.error(f'Input does not exist: {args.input}')
    if not args.run_dir.is_dir():
        parser.error(f'Run does not exist: {args.run_dir}')
    has_sft_params = (args.run_dir / 'network_params.pt').is_file()
    has_rl_params = any((args.run_dir / 'checkpoints').glob('step_*.pt'))
    if not has_sft_params and not has_rl_params:
        parser.error(f'Run contains no network parameters: {args.run_dir}')
    if not args.lean_project.is_dir():
        parser.error(f'Lean project does not exist: {args.lean_project}')
    if args.num_simulations < 1:
        parser.error('--num-simulations must be positive')
    if args.num_sampled_actions < 1:
        parser.error('--num-sampled-actions must be positive')
    if args.tactic_timeout <= 0:
        parser.error('--tactic-timeout must be positive')
    if args.parallel_searches < 1:
        parser.error('--parallel-searches must be positive')
    if args.inference_batch_size < 1:
        parser.error('--inference-batch-size must be positive')
    if args.inference_batch_timeout < 0:
        parser.error('--inference-batch-timeout cannot be negative')
    return args


def main() -> None:
    """Load one network and search a JSONL request batch in parallel."""
    args = parse_args()
    seed_everything(args.seed)
    config = make_config(args)
    imports = tuple(args.imports or ('Mathlib',))
    config.environment_ctor = lambda: Environment(
        LeanProject(str(args.lean_project)),
        imports=imports,
    )
    network = Network(config)
    load_network_checkpoint(args.run_dir, network)

    with args.input.open(encoding='utf-8') as input_file:
        records = [json.loads(line) for line in input_file if line.strip()]
    requests = [
        SearchRequest(
            request_id=str(record['request_id']),
            theorem=theorem_text(record),
            disprove=bool(record.get('disprove', False)),
            num_simulations=args.num_simulations,
            stop_on_solution=args.stop_on_solution,
        )
        for record in records
    ]
    with ParallelSearchEngine(config, network) as engine:
        search_results = engine.search(requests)
        inference_stats = engine.inference_stats

    print(
        f'Inference: {inference_stats.batch_count} batches, average size '
        f'{inference_stats.average_batch_size:.2f}, model time '
        f'{inference_stats.model_seconds:.1f}s.',
        flush=True,
    )

    with args.output.open('w', encoding='utf-8') as output_file:
        for search_result in search_results:
            game = search_result.game
            proof_lines = (
                extract_proof_script(game.root)
                if game.root is not None and game.root.is_optimal
                else None
            )
            status = 'rejected' if search_result.rejection is not None else (
                'proved' if proof_lines is not None else 'failed'
            )
            result = {
                'request_id': search_result.request.request_id,
                'status': status,
                'proof': (
                    '\n'.join(proof_lines)
                    if proof_lines is not None
                    else None
                ),
                'error': game.error,
                'duration_seconds': search_result.duration_seconds,
                'tree': (
                    None
                    if search_result.rejection is not None
                    else serialize_search_tree(game.root)
                ),
            }
            output_file.write(json.dumps(result) + '\n')
            output_file.flush()


if __name__ == '__main__':
    main()
