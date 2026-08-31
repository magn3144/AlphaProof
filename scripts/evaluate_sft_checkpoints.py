"""Evaluate every SFT epoch checkpoint on one fixed theorem set."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import torch

from alphaproof.core.config import DEFAULT_EXPERIMENT_PATH, load_experiment_config
from alphaproof.core.network import Network, Params
from alphaproof.inference.infer import make_config
from alphaproof.inference.parallel import ParallelSearchEngine, SearchRequest
from alphaproof.training.randomness import seed_everything

def parse_args() -> argparse.Namespace:
    """Parse checkpoint evaluation arguments."""
    defaults = load_experiment_config(DEFAULT_EXPERIMENT_PATH).rl
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('dataset_path', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument(
        '--num-simulations', type=int, default=defaults.num_simulations
    )
    parser.add_argument(
        '--num-sampled-actions', type=int, default=defaults.num_sampled_actions
    )
    parser.add_argument(
        '--tactic-timeout', type=float, default=defaults.tactic_timeout
    )
    parser.add_argument('--seed', type=int, default=defaults.seed)
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
    return parser.parse_args()


def load_problems(dataset_path: Path) -> list[dict[str, Any]]:
    """Load the fixed evaluation problem set."""
    with dataset_path.open(encoding='utf-8') as dataset_file:
        problems = [
            json.loads(line)
            for line in dataset_file
            if line.strip()
        ]
    if len(problems) != 30:
        raise ValueError('The evaluation dataset must contain exactly 30 problems.')
    return problems


def load_checkpoint(checkpoint_path: Path, network: Network) -> float:
    """Load one resumable SFT checkpoint into the inference network."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location='cpu',
        weights_only=True,
    )
    checkpoint = cast(dict[str, Any], checkpoint)
    network.params = cast(Params, checkpoint['network_params'])
    return float(checkpoint['epoch'])


def write_json(path: Path, value: Any) -> None:
    """Write one readable JSON artifact."""
    with path.open('w', encoding='utf-8') as output_file:
        json.dump(value, output_file, indent=2)
        output_file.write('\n')


def main() -> None:
    """Evaluate each checkpoint on the same selected problems."""
    args = parse_args()
    seed_everything(args.seed)
    checkpoints = sorted(
        (args.run_dir / 'checkpoints').glob('checkpoint_epoch_*.pt')
    )
    if not checkpoints:
        raise FileNotFoundError(f'No SFT checkpoints found in {args.run_dir}')

    problems = load_problems(args.dataset_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / 'problems.json', problems)
    write_json(
        args.output_dir / 'config.json',
        {
            'run_dir': str(args.run_dir.resolve()),
            'dataset_path': str(args.dataset_path.resolve()),
            'checkpoints': [str(path.resolve()) for path in checkpoints],
            'num_problems': len(problems),
            'num_simulations': args.num_simulations,
            'num_sampled_actions': args.num_sampled_actions,
            'tactic_timeout': args.tactic_timeout,
            'seed': args.seed,
            'parallel_searches': args.parallel_searches,
            'inference_batch_size': args.inference_batch_size,
            'inference_batch_timeout': args.inference_batch_timeout,
        },
    )

    config_args = argparse.Namespace(
        run_dir=args.run_dir,
        num_simulations=args.num_simulations,
        num_sampled_actions=args.num_sampled_actions,
        tactic_timeout=args.tactic_timeout,
        parallel_searches=args.parallel_searches,
        inference_batch_size=args.inference_batch_size,
        inference_batch_timeout=args.inference_batch_timeout,
        seed=args.seed,
    )
    config = make_config(config_args)
    network = Network(config)

    summaries = []
    results_path = args.output_dir / 'results.jsonl'
    with results_path.open('w', encoding='utf-8') as results_file:
        for checkpoint_path in checkpoints:
            epoch = load_checkpoint(checkpoint_path, network)
            solved = 0
            print(f'Evaluating {checkpoint_path.name}', flush=True)
            checkpoint_started = perf_counter()
            requests = [
                SearchRequest(
                    request_id=str(problem_index),
                    theorem=problem['theorem'],
                    disprove=False,
                    num_simulations=config.num_simulations,
                    stop_on_solution=True,
                )
                for problem_index, problem in enumerate(problems)
            ]
            with ParallelSearchEngine(config, network) as engine:
                search_results = engine.search(requests)
            checkpoint_seconds = perf_counter() - checkpoint_started

            for problem_index, (problem, search_result) in enumerate(
                zip(problems, search_results)
            ):
                game = search_result.game
                elapsed_seconds = search_result.duration_seconds
                solved += int(game.root.is_optimal)

                result = {
                    'checkpoint': checkpoint_path.name,
                    'epoch': epoch,
                    'problem_index': problem_index,
                    'problem_id': problem['id'],
                    'difficulty': problem['difficulty'],
                    'seed': args.seed,
                    'solved': game.root.is_optimal,
                    'rejected': search_result.rejection is not None,
                    'proof': game.final_proof,
                    'error': game.error,
                    'elapsed_seconds': elapsed_seconds,
                    'expansions': len(game.timings.tactic_generations),
                    'timings': game.timings.record(),
                }
                results_file.write(json.dumps(result) + '\n')
                results_file.flush()
                print(
                    f'  {problem_index + 1:02d}/{len(problems)} '
                    f'{problem["id"]}: {"solved" if game.root.is_optimal else "failed"} '
                    f'({elapsed_seconds:.1f}s)',
                    flush=True,
                )

            summaries.append({
                'checkpoint': checkpoint_path.name,
                'epoch': epoch,
                'solved': solved,
                'total': len(problems),
                'success_rate': solved / len(problems),
                'elapsed_seconds': checkpoint_seconds,
            })
            write_json(args.output_dir / 'summary.json', summaries)

    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
