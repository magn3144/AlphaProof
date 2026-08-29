from pathlib import Path

from alphaproof.core.actors import ObjectiveRejection
from alphaproof.core.config import Config
from alphaproof.inference.parallel import (
    ParallelSearchEngine,
    SearchRequest,
    SearchResult,
)
from alphaproof.training.matchmaker import Matchmaker
from alphaproof.training.replay_buffer import ReplayBuffer
from alphaproof.training.run_logger import RunLogger
from alphaproof.training.validation import (
    load_validation_theorems,
    validate_theorems,
)


def run_validation_if_due(
    config: Config,
    engine: ParallelSearchEngine,
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
            validate_theorems(config, engine, validation_theorems),
        )
        engine.take_inference_stats()


def make_actor_request(
    matchmaker: Matchmaker,
    logger: RunLogger,
) -> SearchRequest:
    """Assign and record one actor search."""
    assignment = matchmaker.get_start_position()
    request_id = logger.next_actor_request_id()
    logger.log_game_start(request_id, assignment)
    return SearchRequest(
        request_id=request_id,
        theorem=assignment.theorem,
        disprove=assignment.disprove,
        num_simulations=assignment.num_simulations,
        stop_on_solution=True,
    )


def process_actor_result(
    result: SearchResult,
    replay_buffer: ReplayBuffer,
    matchmaker: Matchmaker,
    logger: RunLogger,
) -> None:
    """Persist one usable actor result and reject unusable objectives."""
    game = result.game
    if result.rejection is ObjectiveRejection.THEOREM:
        matchmaker.reject_theorem(game.theorem)
        print(
            f'Rejected theorem that Lean could not initialize: {game.theorem}',
            flush=True,
        )
        return
    if result.rejection is ObjectiveRejection.DISPROOF:
        matchmaker.reject_disproof(game.theorem)
        print(
            f'Rejected disproof objective that Lean could not initialize: '
            f'{game.theorem}',
            flush=True,
        )
        return

    if game.root.is_optimal:
        replay_buffer.save_game(game)
    matchmaker.send_game(game)
    logger.log_game(game, len(replay_buffer))


def run_actor_phase(
    config: Config,
    run_dir: Path,
    resume: bool,
    engine: ParallelSearchEngine,
    validation_engine: ParallelSearchEngine,
    replay_buffer: ReplayBuffer,
    matchmaker: Matchmaker,
    logger: RunLogger,
    game_target: int,
) -> None:
    """Run independently assigned searches to a target game count."""
    validation_theorems = load_validation_theorems(config, run_dir, resume)
    run_validation_if_due(config, validation_engine, logger, validation_theorems)

    while logger.games_completed < game_target:
        next_validation = (
            logger.games_completed
            // config.theorem_validation_interval_games
            + 1
        ) * config.theorem_validation_interval_games
        boundary = min(game_target, next_validation)

        while engine.num_searches < engine.parallel_searches:
            engine.submit(make_actor_request(matchmaker, logger))

        while logger.games_completed < boundary:
            process_actor_result(
                engine.next_result(),
                replay_buffer,
                matchmaker,
                logger,
            )
            engine.submit(make_actor_request(matchmaker, logger))

        engine.pause()
        logger.log_inference(engine.take_inference_stats())
        run_validation_if_due(
            config,
            validation_engine,
            logger,
            validation_theorems,
        )
        if logger.games_completed < game_target:
            engine.resume()
