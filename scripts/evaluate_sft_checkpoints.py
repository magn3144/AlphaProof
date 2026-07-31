"""Evaluate every SFT epoch checkpoint on one fixed theorem set."""

import argparse
import json
import random
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import torch

from alphaproof.core.network import Network, Params
from alphaproof.inference.infer import make_config, prove




def parse_args() -> argparse.Namespace:
    """Parse checkpoint evaluation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('dataset_path', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--num-simulations', type=int, default=32)
    parser.add_argument('--num-sampled-actions', type=int, default=4)
    parser.add_argument('--tactic-timeout', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
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
        },
    )

    config_args = argparse.Namespace(
        run_dir=args.run_dir,
        num_simulations=args.num_simulations,
        tactic_timeout=args.tactic_timeout,
    )
    config = make_config(config_args)
    network = Network(config)
    network.num_sampled_actions = args.num_sampled_actions

    summaries = []
    results_path = args.output_dir / 'results.jsonl'
    with results_path.open('w', encoding='utf-8') as results_file:
        for checkpoint_path in checkpoints:
            epoch = load_checkpoint(checkpoint_path, network)
            solved = 0
            checkpoint_seconds = 0.0
            print(f'Evaluating {checkpoint_path.name}', flush=True)

            for problem_index, problem in enumerate(problems):
                problem_seed = args.seed + problem_index
                random.seed(problem_seed)
                torch.manual_seed(problem_seed)

                start = perf_counter()
                game = prove(problem['theorem'], config, network)
                elapsed_seconds = perf_counter() - start
                checkpoint_seconds += elapsed_seconds
                solved += int(game.root.is_optimal)

                result = {
                    'checkpoint': checkpoint_path.name,
                    'epoch': epoch,
                    'problem_index': problem_index,
                    'problem_id': problem['id'],
                    'difficulty': problem['difficulty'],
                    'seed': problem_seed,
                    'solved': game.root.is_optimal,
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
