import math
from collections.abc import Callable
from time import perf_counter
from typing import Dict, List

from alphaproof.core.config import Config
from alphaproof.core.environment import (
    Action,
    Environment,
    NodeType,
    TacticDeadlineExceeded,
)
from alphaproof.core.game import (
    Game,
    Node,
    ProofVerifier,
    action_to_tactic,
    compute_value_target,
    final_check,
)
from alphaproof.core.timing import GameTimings
from alphaproof.training.matchmaker import Matchmaker
from alphaproof.core.network import Network
from alphaproof.training.replay_buffer import ReplayBuffer
from alphaproof.training.shared_storage import SharedStorage
from leantree.repl_adapter.interaction import LeanInteractionException


# Each acting job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the learner by
# writing it to a shared replay buffer.
def run_actor(
        config: Config,
        storage: SharedStorage,
        replay_buffer: ReplayBuffer,
        matchmaker: Matchmaker,
        num_games: int,
        on_game: Callable[[Game], None] | None = None,
        on_game_start: Callable[[Game], None] | None = None,
):
    """Generate solved games from the latest checkpoint."""
    network = Network(config)
    games_completed = 0
    with ProofVerifier(config.final_check_timeout) as verifier:
        while games_completed < num_games:
            network.params = storage.latest_params()
            game_start = perf_counter()
            game = play_game(
                config,
                network,
                matchmaker,
                verifier,
                on_game_start,
            )
            if game is None:
                continue
            game.timings.total_seconds = perf_counter() - game_start
            if game.root.is_optimal:
                replay_buffer.save_game(game)
            matchmaker.send_game(game)
            games_completed += 1
            if on_game is not None:
                on_game(game)


# Each game is produced by starting from the initial Lean state, and executing
# Monte Carlo tree search to find a proof. If one is found, we extract from the
# search tree the state-tactic-value transitions in the proof, which are added
# to a replay buffer for training.
def play_game(
        config: Config,
        network: Network,
        matchmaker: Matchmaker,
        verifier: ProofVerifier,
        on_game_start: Callable[[Game], None] | None = None,
) -> Game | None:
    """Run one theorem episode and validate any discovered proof."""
    game = matchmaker.get_start_position()
    if on_game_start is not None:
        on_game_start(game)
    setup_start = perf_counter()
    with config.environment_ctor() as environment:
        try:
            state = environment.initial_state(game.theorem)
        except LeanInteractionException as error:
            matchmaker.reject_theorem(game.theorem)
            print(
                f'Rejected theorem that Lean could not initialize: {error}',
                flush=True,
            )
            return None
        if game.disprove:
            try:
                state = environment.step(state.id, 'disprove')
            except (LeanInteractionException, ValueError) as error:
                matchmaker.reject_theorem(game.theorem)
                print(
                    f'Rejected theorem that could not be negated: {error}',
                    flush=True,
                )
                return None
        game.timings.setup_seconds = perf_counter() - setup_start
        game.root = Node(
                action=None,
                observation=state.observation,
                prior=1.0,
                node_type=NodeType.OR,
                state_id=state.id,
                is_optimal=state.terminal,
                is_terminal=state.terminal,
                reward=state.reward,
        )
        assert game.root.node_type == NodeType.OR

        if config.debug:
            print(
                    f'\n{"=" * 80}\n'
                    'Game\n'
                    f'{"=" * 80}\n'
                    f'Mode: {"disprove" if game.disprove else "prove"}\n'
                    f'Theorem:\n{game.theorem}',
                    flush=True,
            )

        # Run Monte Carlo tree search to find a proof.
        try:
            run_mcts(config, game, network, environment)
        except TacticDeadlineExceeded as error:
            game.error = str(error)
            return game

    if game.root.is_optimal:
        # Perform final check to ensure the proof is valid.
        verification_start = perf_counter()
        game.root.is_optimal = final_check(
                game,
                config.final_check_timeout,
                verifier,
        )
        game.timings.final_verification_seconds = (
            perf_counter() - verification_start
        )
        game.timings.verifier_startup_seconds = verifier.last_startup_seconds
        game.timings.final_verification_success = game.root.is_optimal
        if game.root.is_optimal:
            # Compute value targets for the proof.
            compute_value_target(game.root)

    return game


# Core Monte Carlo tree search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(
        config: Config,
        game: Game,
        network: Network,
        environment: Environment,
):
    """Run MCTS simulations from the game root."""
    root = game.root
    for i in range(game.num_simulations):
        node = root
        search_path = [node]

        while node.expanded() and not progressive_sample(node, config):
            _, node = select_child(config, node)
            search_path.append(node)

        assert node.observation is not None
        generation_start = perf_counter()
        network_sample_output = network.sample(str(node.observation))
        game.timings.add_tactic_generation(
            simulation=i + 1,
            state_id=node.state_id,
            seconds=perf_counter() - generation_start,
            num_tactics=len(network_sample_output.action_logprobs),
        )
        if config.debug:
            actions = '\n'.join(
                    f'  {action}'
                    for action in network_sample_output.action_logprobs
            )
            print(
                    f'\n--- Rollout {i + 1}/{game.num_simulations} ---\n'
                    f'Leaf state:\n{node.observation}\n'
                    f'Actions sampled at leaf:\n{actions}',
                    flush=True,
            )
        expand_node(node, network_sample_output.action_logprobs,
                                environment, config.prior_temperature,
                                config.tactic_timeout, game.timings)
        backpropagate(
                search_path,
                network_sample_output.value,
                config,
        )
        if root.is_optimal:
            break


def progressive_sample(node: Node, config: Config) -> bool:
    """Whether to expand a node in the search tree again (progressive sampling)."""
    return (
            node.node_type == NodeType.OR
            and node.evaluations <= config.ps_c * node.visit_count**config.ps_alpha
    )


def select_child(config: Config, node: Node) -> tuple[Action, Node]:
    """Selects the child with the highest UCB score."""
    action, child = max(
            node.children.items(),
            key=lambda item: ucb_score(config, node, item[1]),
    )
    return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config: Config, parent: Node, child: Node) -> float:
    """Score a child with value and prior-based exploration."""
    pb_c = (
            math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base)
            + config.pb_c_init
    )
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
    if parent.node_type == NodeType.AND:
        pb_c *= config.c_and

    # Due to progressive sampling, we normalise priors here.
    prior_score = pb_c * child.prior / parent.prior_sum()
    if child.visit_count > 0:
        value = child.reward + child.value()
    else:
        value = parent.value() - config.unvisited_value_penalty
    value_score = config.value_discount ** (- 1 - value)

    if parent.node_type == NodeType.AND:
        # Invert value score for AND nodes.
        value_score = 1 - value_score
        if child.is_optimal:
            # Avoid re-selecting proven subgoals.
            value_score = -1e9
    return prior_score + value_score


# We expand a node using the value and sampled actions obtained from
# the neural network. Immediately attempt the actions in the environment.
def expand_node(
        node: Node,
        network_action_logprobs: Dict[Action, float],
        environment: Environment,
        temperature: float,
        tactic_timeout: float,
        timings: GameTimings,
):
    """Expand a node by applying sampled actions in the environment."""
    node.evaluations += 1
    policy = {
            a: math.exp(network_action_logprobs[a] / temperature)
            for a in network_action_logprobs
    }
    for action, p in policy.items():
        # Check if action is duplicate.
        if action in node.children:
            node.children[action].prior += p
            continue
        # Immediately apply the actions in the environment.
        try:
            tactic_start = perf_counter()
            successful = False
            try:
                state = environment.step(
                    node.state_id,
                    action,
                    tactic_timeout=tactic_timeout,
                )
                successful = True
            finally:
                timings.add_tactic_execution(
                    action_to_tactic(action),
                    perf_counter() - tactic_start,
                    successful,
                )
        except ValueError:
            # Invalid action encountered.
            continue
        else:
            node.children[action] = Node(
                    observation=state.observation,
                    action=action,
                    prior=p,
                    state_id=state.id,
                    node_type=NodeType.AND if state.num_goals > 1 else NodeType.OR,
                    is_optimal=state.terminal,
                    is_terminal=state.terminal,
                    reward=state.reward,
            )
            if node.node_type == NodeType.OR:
                node.is_optimal |= state.terminal
            if state.num_goals > 1:
                # For AND nodes, immediately add children with pseudo-actions to focus
                # on each goal.
                expand_node(
                    node.children[action],
                    {
                        f'focus_goal {i}': math.log(1./state.num_goals)
                        for i in range(state.num_goals)
                    },
                    environment,
                    temperature,
                    tactic_timeout,
                    timings,
                )


def backprop_value_towards_min(node):
    """Computes the value for an AND node by propagating the min value from children."""
    value = 1
    for child in node.children.values():
        if not child.is_optimal and child.visit_count > 0:
            value = min(value, child.value())
    return value


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(
        search_path: List[Node],
        value: float,
        config: Config,
):
    """Backpropagate a simulation value through the visited nodes."""
    if not search_path[-1].expanded():
        value = config.no_legal_actions_value
    is_optimal = False
    for ix, node in reversed(list(enumerate(search_path))):
        node.value_sum += value
        node.visit_count += 1
        if node.node_type == NodeType.AND:
            is_optimal = all(child.is_optimal for child in node.children.values())
        else:
            is_optimal |= node.is_optimal
        node.is_optimal = is_optimal
        if ix > 0 and search_path[ix - 1].node_type == NodeType.AND:
            value = backprop_value_towards_min(search_path[ix - 1])
        else:
            value = node.reward + value
