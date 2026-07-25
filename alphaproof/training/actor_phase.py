from pathlib import Path

from alphaproof.core.actors import run_actor
from alphaproof.core.config import Config
from alphaproof.training.matchmaker import Matchmaker
from alphaproof.training.replay_buffer import ReplayBuffer
from alphaproof.training.run_logger import RunLogger
from alphaproof.training.shared_storage import SharedStorage
from alphaproof.training.validation import (
    load_validation_theorems,
    validate_theorems,
)


def run_validation_if_due(
    config: Config,
    storage: SharedStorage,
    logger: RunLogger,
    validation_theorems: list[str],
) -> None:
    """Run validation when the completed-game count reaches an interval."""
    game = logger.games_completed
    if (
        game > 0
        and game % config.theorem_validation_interval_games == 0
        and game not in logger.validation_games
    ):
        logger.log_validation(
            game,
            validate_theorems(config, storage, validation_theorems),
        )


def run_actor_phase(
    config: Config,
    run_dir: Path,
    resume: bool,
    storage: SharedStorage,
    replay_buffer: ReplayBuffer,
    matchmaker: Matchmaker,
    logger: RunLogger,
    game_target: int,
) -> None:
    """Run actors to a target game count and validate at fixed intervals."""
    validation_theorems = load_validation_theorems(config, run_dir, resume)
    run_validation_if_due(config, storage, logger, validation_theorems)

    while logger.games_completed < game_target:
        next_validation = (
            logger.games_completed
            // config.theorem_validation_interval_games
            + 1
        ) * config.theorem_validation_interval_games
        games_to_run = min(game_target, next_validation) - logger.games_completed
        run_actor(
            config,
            storage,
            replay_buffer,
            matchmaker,
            games_to_run,
            lambda game: logger.log_game(game, len(replay_buffer)),
            logger.log_game_start,
        )
        run_validation_if_due(config, storage, logger, validation_theorems)
