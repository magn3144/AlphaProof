import argparse
import json
import random
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import torch

from alphaproof.core.actors import run_mcts
from alphaproof.core.config import Config
from alphaproof.core.environment import Environment, NodeType
from alphaproof.core.game import Game, Node, extract_proof_script, final_check
from alphaproof.core.network import Network, Params
from alphaproof.core.paths import LEAN_PROJECT_DIR
from alphaproof.inference.search_tree import serialize_search_tree
from leantree import LeanProject


def load_run_config(run_dir: Path) -> dict[str, Any]:
    """Load optional settings saved alongside a network checkpoint."""
    config_path = run_dir / 'config.json'
    if not config_path.exists():
        return {}
    with config_path.open(encoding='utf-8') as file:
        return json.load(file)


def make_config(args: argparse.Namespace) -> Config:
    """Build search configuration for an SFT or RL run."""
    run_data = load_run_config(args.run_dir)
    saved_config = run_data.get('config', {})
    if saved_config:
        model_run_dir = Path(saved_config['sft_run_dir'])
        learning_rate = float(saved_config['lr'])
    else:
        model_run_dir = args.run_dir
        learning_rate = float(run_data.get('learning_rate', 5e-5))

    config = Config(
        num_simulations=args.num_simulations,
        batch_size=1,
        num_actors=1,
        num_games=1,
        lr=learning_rate,
        sft_run_dir=model_run_dir,
        max_state_length=int(
            saved_config.get(
                'max_state_length', run_data.get('max_state_length', 640)
            )
        ),
        max_action_length=int(
            saved_config.get(
                'max_action_length', run_data.get('max_action_length', 128)
            )
        ),
    )
    for name in (
        'pb_c_base',
        'pb_c_init',
        'value_discount',
        'prior_temperature',
        'no_legal_actions_value',
        'ps_c',
        'ps_alpha',
        'num_value_bins',
    ):
        if name in saved_config:
            setattr(config, name, saved_config[name])
    config.tactic_timeout = args.tactic_timeout
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


def prove(
    theorem: str,
    config: Config,
    network: Network,
    disprove: bool = False,
    stop_on_solution: bool = True,
) -> Game:
    """Search for and verify a proof of one theorem."""
    game = Game(theorem, disprove, config.num_simulations)
    with config.environment_ctor() as environment:
        state = environment.initial_state(theorem)
        if disprove:
            state = environment.step(state.id, 'disprove')
        game.root = Node(
            action=None,
            observation=state.observation,
            prior=1.0,
            state_id=state.id,
            node_type=NodeType.OR,
            reward=state.reward,
            is_optimal=state.terminal,
            is_terminal=state.terminal,
        )
        run_mcts(
            config,
            game,
            network,
            environment,
            stop_on_solution=stop_on_solution,
        )

    if game.root.is_optimal:
        game.root.is_optimal = final_check(game, config.final_check_timeout)
    return game


def theorem_text(record: dict[str, Any]) -> str:
    """Combine an optional Lean header with a theorem request."""
    header = str(record.get('header') or '')
    return f'{header}\n{record["theorem"]}' if header else str(record['theorem'])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse paths and shared settings for JSONL inference."""
    default_run_dir = Config().sft_run_dir
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
    parser.add_argument('--num-simulations', type=int, default=16)
    parser.add_argument('--num-sampled-actions', type=int, default=4)
    parser.add_argument('--tactic-timeout', type=float, default=1.0)
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
    return args


def main() -> None:
    """Load one network and run one tree search per JSONL request."""
    args = parse_args()
    config = make_config(args)
    imports = tuple(args.imports or ('Mathlib',))
    config.environment_ctor = lambda: Environment(
        LeanProject(str(args.lean_project)),
        imports=imports,
    )
    network = Network(config)
    load_network_checkpoint(args.run_dir, network)
    network.num_sampled_actions = args.num_sampled_actions

    with (
        args.input.open(encoding='utf-8') as input_file,
        args.output.open('w', encoding='utf-8') as output_file,
    ):
        for line in input_file:
            record = json.loads(line)
            seed = int(record['seed'])
            random.seed(seed)
            torch.manual_seed(seed)
            started = perf_counter()
            game = prove(
                theorem_text(record),
                config,
                network,
                stop_on_solution=args.stop_on_solution,
            )
            proof_lines = (
                extract_proof_script(game.root)
                if game.root.is_optimal
                else None
            )
            result = {
                'request_id': record['request_id'],
                'status': 'proved' if proof_lines is not None else 'failed',
                'proof': (
                    '\n'.join(proof_lines)
                    if proof_lines is not None
                    else None
                ),
                'duration_seconds': perf_counter() - started,
                'tree': serialize_search_tree(game.root),
            }
            output_file.write(json.dumps(result) + '\n')
            output_file.flush()


if __name__ == '__main__':
    main()
