import dataclasses
import json
import random
from pathlib import Path

from alphaproof.core.config import Config
from alphaproof.core.environment import Theorem
from alphaproof.core.game import Game
from alphaproof.core.paths import RUNS_DIR


MATCHMAKER_STATS_FILE = 'matchmaker_stats.json'


class Matchmaker:
    """Chooses theorem objectives and records their outcomes."""

    @dataclasses.dataclass
    class Stats:
        """Statistics for a theorem."""
        # List of (disprove, result) tuples:
        # Disprove is True iff this was an attempt to disprove the theorem.
        # Result is True iff the attempt was successful.
        attempts: list[tuple[bool, bool]] = dataclasses.field(default_factory=list)
        invalid: bool = False

        def update(self, game: Game):
            """Update statistics with the results of a game."""
            self.attempts.append((game.disprove, game.root.is_optimal))

        def weight(self, config: Config) -> float:
            """Compute weight of this theorem."""
            if self.invalid:
                return 0.0
            if not self.attempts:
                return 1.0
            disproved = any(
                    disprove and success for (disprove, success) in self.attempts
            )
            proved = any(
                    (not disprove) and success for (disprove, success) in self.attempts
            )
            if disproved:
                return 0.0
            elif len(self.attempts) < config.mm_trust_count:
                return 1.0
            elif not disproved and not proved:
                # Never managed to prove or disprove.
                return config.mm_undecided_weight
            else:
                latest = self.attempts[-config.mm_fully_decided_trust_count :]
                if all((not disprove) and success for (disprove, success) in latest):
                    # Consistently proved.
                    return config.mm_proved_weight
            return 1.0

    def __init__(self, config: Config):
        """Initialize theorem statistics from the backing store."""
        self.config = config
        self.run_dir = RUNS_DIR / str(config.run_id)
        self.stats_path = self.run_dir / MATCHMAKER_STATS_FILE
        self.theorem_stats = self._load_theorem_stats(config.dataset_path)
        self._save_theorem_stats()

    def _load_theorem_stats(self, dataset_path: Path) -> dict[Theorem, Stats]:
        """Load theorem starts from a JSONL dataset and resume run statistics."""
        theorem_stats: dict[Theorem, Matchmaker.Stats] = {}
        with dataset_path.open() as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                theorem = record['theorem']
                theorem_stats[theorem] = Matchmaker.Stats()

        if not theorem_stats:
            raise ValueError(f'No theorems found in {dataset_path}.')

        if self.stats_path.exists():
            with self.stats_path.open() as file:
                saved_stats = json.load(file)
            for record in saved_stats.get('theorem_stats', []):
                theorem = record['theorem']
                if theorem in theorem_stats:
                    theorem_stats[theorem] = Matchmaker.Stats(
                            attempts=[
                                    (bool(disprove), bool(success))
                                    for disprove, success in record.get('attempts', [])
                            ],
                            invalid=bool(record['invalid']),
                    )
        return theorem_stats

    def _save_theorem_stats(self):
        """Persist theorem statistics for this run."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        records = [
                {
                        'theorem': theorem,
                        'attempts': [
                                [disprove, success]
                                for disprove, success in stats.attempts
                        ],
                        'invalid': stats.invalid,
                }
                for theorem, stats in self.theorem_stats.items()
        ]
        with self.stats_path.open('w') as file:
            json.dump({'theorem_stats': records}, file, indent=2)
            file.write('\n')

    def compute_num_simulations(self, theorem: Theorem, stats: Stats) -> int:
        """Compute number of simulations to run for a theorem."""
        recent_attempts = stats.attempts[
            -self.config.mm_simulation_failure_window:
        ]
        failures = sum(not success for _, success in recent_attempts)
        num_simulations = int(
                self.config.num_simulations
                * self.config.mm_simulation_failure_multiplier ** failures
        )
        return min(num_simulations, self.config.mm_max_num_simulations)

    def get_start_position(self) -> Game:
        """Get a start position for a new game to be played."""
        # Get a theorem to be proved or disproved based on the per-theorem stats.
        # Prefer interesting theorems.
        weights = [
                stats.weight(self.config) for stats in self.theorem_stats.values()
        ]
        if sum(weights) == 0:
            raise ValueError('No valid theorems remain in the dataset.')
        [(theorem, stats)] = random.choices(
                list(self.theorem_stats.items()), weights, k=1
        )
        disprove = random.random() < self.config.mm_disprove_rate
        num_simulations = self.compute_num_simulations(theorem, stats)
        return Game(
                theorem=theorem, disprove=disprove, num_simulations=num_simulations
        )

    def send_game(self, game: Game):
        """Send completed game to matchmaker."""
        self.theorem_stats[game.theorem].update(game)
        self._save_theorem_stats()

    def reject_theorem(self, theorem: Theorem) -> None:
        """Permanently reject a theorem that Lean cannot initialize."""
        self.theorem_stats[theorem].invalid = True
        self._save_theorem_stats()
