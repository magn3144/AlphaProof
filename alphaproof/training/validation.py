import json
import random
from pathlib import Path

from alphaproof.core.actors import play_game
from alphaproof.core.config import Config
from alphaproof.core.game import Game, ProofVerifier
from alphaproof.core.network import Network
from alphaproof.training.matchmaker import Matchmaker
from alphaproof.training.shared_storage import SharedStorage


VALIDATION_THEOREMS_FILE = 'validation_theorems.json'


class ValidationMatchmaker(Matchmaker):
    """Provide one fixed proof objective without recording curriculum state."""

    def __init__(self, config: Config, theorem: str):
        self.config = config
        self.theorem = theorem

    def get_start_position(self) -> Game:
        """Return the configured validation theorem."""
        return Game(
            theorem=self.theorem,
            disprove=False,
            num_simulations=self.config.num_simulations,
        )

    def reject_theorem(self, theorem: str) -> None:
        """Leave fixed validation objectives unchanged."""


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
    storage: SharedStorage,
    theorems: list[str],
) -> list[Game | None]:
    """Play the fixed validation theorem set without training side effects."""
    network = Network(config)
    network.params = storage.latest_params()
    games: list[Game | None] = []
    with ProofVerifier(config.final_check_timeout) as verifier:
        for theorem in theorems:
            matchmaker = ValidationMatchmaker(config, theorem)
            games.append(play_game(config, network, matchmaker, verifier))
    return games
