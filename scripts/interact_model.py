"""Interactively sample tactics from a trained AlphaProof network."""

import argparse
import random
from pathlib import Path

import torch

from alphaproof.core.config import DEFAULT_EXPERIMENT_PATH, load_experiment_config
from alphaproof.core.network import Network
from alphaproof.inference.infer import load_network_checkpoint, make_config
from alphaproof.training.sft import resolve_device


def parse_args() -> argparse.Namespace:
    """Parse model interaction arguments."""
    defaults = load_experiment_config(DEFAULT_EXPERIMENT_PATH).rl
    default_run_dir = defaults.sft_run_dir
    parser = argparse.ArgumentParser(
        description=(
            'Sample tactics directly from a trained Salesforce CodeT5+ model.'
        )
    )
    parser.add_argument(
        '--run-dir',
        type=Path,
        default=default_run_dir,
        help=(
            'SFT or RL run containing trained network parameters '
            f'(default: {default_run_dir}).'
        ),
    )
    parser.add_argument(
        '--num-sampled-actions', type=int, default=defaults.num_sampled_actions
    )
    parser.add_argument(
        '--device',
        choices=('auto', 'cpu', 'cuda', 'mps'),
        default='auto',
    )
    parser.add_argument('--seed', type=int, default=defaults.seed)
    args = parser.parse_args()

    if not args.run_dir.is_dir():
        parser.error(f'Run does not exist: {args.run_dir}')
    has_sft_params = (args.run_dir / 'network_params.pt').is_file()
    has_rl_params = any((args.run_dir / 'checkpoints').glob('step_*.pt'))
    if not has_sft_params and not has_rl_params:
        parser.error(f'Run contains no network parameters: {args.run_dir}')
    if args.num_sampled_actions < 1:
        parser.error('--num-sampled-actions must be positive')
    return args


def read_state() -> str | None:
    """Read one multiline Lean state, returning None when the user exits."""
    lines = []
    while True:
        try:
            line = input('state> ' if not lines else '     | ')
        except EOFError:
            print()
            return '\n'.join(lines) if lines else None
        if not line:
            return '\n'.join(lines) if lines else None
        lines.append(line)


def main() -> None:
    """Load a trained network and repeatedly sample tactics for Lean states."""
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    defaults = load_experiment_config(DEFAULT_EXPERIMENT_PATH).rl
    config_args = argparse.Namespace(
        run_dir=args.run_dir,
        num_simulations=defaults.num_simulations,
        num_sampled_actions=args.num_sampled_actions,
        tactic_timeout=defaults.tactic_timeout,
        parallel_searches=defaults.num_actors,
        inference_batch_size=defaults.inference_batch_size,
        inference_batch_timeout=defaults.inference_batch_timeout,
        seed=args.seed,
    )
    config = make_config(config_args)
    network = Network(config)
    checkpoint_path = load_network_checkpoint(args.run_dir, network)
    network.device = resolve_device(args.device)
    network.to(network.device)

    print(f'Loaded {checkpoint_path} on {network.device}.')
    print('Paste a Lean state and finish it with an empty line.')
    print('Submit an empty state or press Ctrl-D to exit.')

    while state := read_state():
        output = network.sample(state)
        print(f'\nValue: {output.value:.3f}')
        print('Tactics (log probability):')
        for action, logprob in sorted(
            output.action_logprobs.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(f'{logprob:9.3f}  {action}')
        print()


if __name__ == '__main__':
    main()
