from pathlib import Path

from alphaproof.core.config import Config
from alphaproof.core.network import Network
from alphaproof.inference.parallel import ParallelSearchEngine, SearchRequest
from alphaproof.training.matchmaker import Matchmaker
from alphaproof.training.replay_buffer import ReplayBuffer
from alphaproof.training.run_logger import RunLogger
from alphaproof.training.validation import (
    load_validation_theorems,
    validate_theorems,
)


def run_validation_if_due(
    config: Config,
    network: Network,
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
            validate_theorems(config, network, validation_theorems),
        )


def run_actor_phase(
    config: Config,
    run_dir: Path,
    resume: bool,
    network: Network,
    replay_buffer: ReplayBuffer,
    matchmaker: Matchmaker,
    logger: RunLogger,
    game_target: int,
) -> None:
    """Run repeated-theorem search groups to a target game count."""
    validation_theorems = load_validation_theorems(config, run_dir, resume)
    run_validation_if_due(config, network, logger, validation_theorems)

    while logger.games_completed < game_target:
        next_validation = (
            logger.games_completed
            // config.theorem_validation_interval_games
            + 1
        ) * config.theorem_validation_interval_games
        boundary = min(game_target, next_validation)
        with ParallelSearchEngine(config, network) as engine:
            while logger.games_completed < boundary:
                assignment = matchmaker.get_start_position()
                group_size = min(
                    config.num_actors,
                    boundary - logger.games_completed,
                )
                logger.log_game_start(assignment)
                requests = [
                    SearchRequest(
                        request_id=(
                            f'train-{logger.games_completed + index + 1}'
                        ),
                        theorem=assignment.theorem,
                        disprove=assignment.disprove,
                        num_simulations=assignment.num_simulations,
                        stop_on_solution=True,
                    )
                    for index in range(group_size)
                ]
                results = engine.search(requests)
                if any(result.rejected for result in results):
                    matchmaker.reject_theorem(assignment.theorem)
                    print(
                        'Rejected theorem that Lean could not initialize or negate: '
                        f'{assignment.theorem}',
                        flush=True,
                    )
                    continue

                for result in results:
                    game = result.game
                    if game.root.is_optimal:
                        replay_buffer.save_game(game)
                    matchmaker.send_game(game)
                    logger.log_game(game, len(replay_buffer))

        logger.log_inference(engine.inference_stats)
        run_validation_if_due(config, network, logger, validation_theorems)
