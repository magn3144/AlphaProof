import json
import random
from pathlib import Path

from alphaproof.core.config import Config
from alphaproof.core.game import Game
from alphaproof.core.network import Network
from alphaproof.inference.parallel import ParallelSearchEngine, SearchRequest


VALIDATION_THEOREMS_FILE = 'validation_theorems.json'


def load_validation_theorems(
    config: Config,
    run_dir: Path,
    resume: bool,
) -> list[str]:
    """Draw fixed validation theorems once and restore them on resume."""
    path = run_dir / VALIDATION_THEOREMS_FILE
    if path.exists():
        with path.open(encoding='utf-8') as validation_file:
            return json.load(validation_file)
    if resume:
        raise FileNotFoundError(f'Validation theorems do not exist: {path}')

    with config.validation_dataset_path.open(encoding='utf-8') as dataset_file:
        theorems = [
            json.loads(line)['theorem']
            for line in dataset_file
            if line.strip()
        ]
    if len(theorems) < config.theorem_validation_num_theorems:
        raise ValueError(
            'Validation dataset contains fewer theorems than requested.'
        )
    selected = random.Random(config.seed).sample(
        theorems,
        config.theorem_validation_num_theorems,
    )
    with path.open('w', encoding='utf-8') as validation_file:
        json.dump(selected, validation_file, indent=2)
        validation_file.write('\n')
    return selected


def validate_theorems(
    config: Config,
    network: Network,
    theorems: list[str],
) -> list[Game]:
    """Search the fixed validation theorem set without training side effects."""
    requests = [
        SearchRequest(
            request_id=f'validation-{index}',
            theorem=theorem,
            disprove=False,
            num_simulations=config.num_simulations,
            stop_on_solution=True,
        )
        for index, theorem in enumerate(theorems)
    ]
    with ParallelSearchEngine(config, network) as engine:
        return [result.game for result in engine.search(requests)]
