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
    transition_count: int,
) -> None:
    """Run validation after crossing a replay transition interval."""
    interval = config.theorem_validation_interval_transitions
    transition = transition_count // interval * interval
    if transition > 0 and transition not in logger.validation_transitions:
        logger.log_validation(
            transition,
            validate_theorems(config, engine, validation_theorems),
        )
        engine.take_inference_stats()


def make_actor_request(
    matchmaker: Matchmaker,
    logger: RunLogger,
    transition_target: int,
) -> SearchRequest:
    """Assign and record one actor search."""
    assignment = matchmaker.get_start_position()
    request_id = logger.next_actor_request_id()
    logger.log_game_start(request_id, assignment, transition_target)
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
    logger.log_game(
        game,
        len(replay_buffer),
        replay_buffer.transition_count,
    )


def run_actor_phase(
    config: Config,
    run_dir: Path,
    resume: bool,
    engine: ParallelSearchEngine,
    validation_engine: ParallelSearchEngine,
    replay_buffer: ReplayBuffer,
    matchmaker: Matchmaker,
    logger: RunLogger,
    transition_target: int,
) -> None:
    """Run independently assigned searches to a transition target."""
    logger.log_actor_start(transition_target)
    validation_theorems = load_validation_theorems(config, run_dir, resume)
    run_validation_if_due(
        config,
        validation_engine,
        logger,
        validation_theorems,
        replay_buffer.transition_count,
    )
    if replay_buffer.transition_count < transition_target:
        engine.resume()

    while replay_buffer.transition_count < transition_target:
        interval = config.theorem_validation_interval_transitions
        next_validation = (
            replay_buffer.transition_count // interval + 1
        ) * interval
        phase_target = min(transition_target, next_validation)

        while engine.num_searches < engine.parallel_searches:
            engine.submit(
                make_actor_request(matchmaker, logger, transition_target)
            )

        while replay_buffer.transition_count < phase_target:
            process_actor_result(
                engine.next_result(),
                replay_buffer,
                matchmaker,
                logger,
            )
            engine.submit(
                make_actor_request(matchmaker, logger, transition_target)
            )

        engine.pause()
        logger.log_inference(engine.take_inference_stats())
        run_validation_if_due(
            config,
            validation_engine,
            logger,
            validation_theorems,
            replay_buffer.transition_count,
        )
        if replay_buffer.transition_count < transition_target:
            engine.resume()
