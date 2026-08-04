from time import perf_counter

from alphaproof.core.environment import Action, NodeType, Observation, Theorem
from alphaproof.core.helper import replace_sorry_proof, theorem_for_game, theorem_name
from alphaproof.core.paths import LEAN_PROJECT_DIR
from alphaproof.core.timing import GameTimings, is_internal_action
from leantree import LeanProject, LeanTactic
from leantree.repl_adapter.interaction import (
    LeanEnvironmentCheckpoint,
    LeanInteractionException,
    LeanProcess,
    LeanProcessException,
)


class Node:
    """Node in the search tree."""

    def __init__(
        self,
        action: Action | None,
        observation: Observation,
        prior: float,
        state_id: int,
        node_type: NodeType,
        reward: float,
        is_optimal: bool = False,
        is_terminal: bool = False,
    ):
        """Initialize a search node reached by an optional incoming action."""
        # Action that was taken to reach this node.
        self.action = action
        # Observation after the action has been applied.
        self.observation = observation
        # Environment state ID after the action has been applied.
        self.state_id = state_id
        # Whether the node is an OR or AND node.
        self.node_type = node_type
        # Whether the action closed the proof of the previous goal.
        self.is_terminal = is_terminal
        # Whether the node is part of an optimal path.
        self.is_optimal = is_optimal
        # Per-step reward obtained after applying the action.
        self.reward = reward
        # Prior probability of the node according to the policy.
        self.prior = prior

        self.visit_count: int = 0
        self.evaluations: int = 0
        self.value_sum: float = 0.0
        self.network_value: float | None = None
        self.children: dict[Action, Node] = {}
        self.expansion: int | None = None
        self.simulation: int | None = None
        self.expansion_seconds: float | None = None
        self.num_tactics: int | None = None

        # Not used in search, but used as a regression target in RL.
        self.value_target: float = 0.0

    def expanded(self) -> bool:
        """Return whether this node has children."""
        return len(self.children) > 0

    def value(self) -> float:
        """Return the average backed-up value."""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def prior_sum(self) -> float:
        """Return the sum of child policy priors."""
        return sum(child.prior for child in self.children.values())


class Game:
    """A single episode of interaction with the environment."""

    def __init__(self, theorem: Theorem, disprove: bool, num_simulations: int):
        """Create an episode around one theorem objective."""
        self.theorem = theorem
        self.error: str | None = None
        self.final_proof: str | None = None
        # Whether to try to prove or disprove the theorem.
        self.disprove = disprove
        # Number of simulations to run. Provided by the matchmaker.
        self.num_simulations = num_simulations
        self.timings = GameTimings()
        # Dummy node for the type checker.
        self.root = Node(
                action=None,
                observation=Observation([]),
                prior=1.0,
                state_id=0,
                node_type=NodeType.OR,
                reward=0.0,
        )


class ProofCheckProcessError(Exception):
    """Raised when the persistent Lean verifier process fails."""


class ProofVerifier:
    """Validate reconstructed proofs in an isolated persistent Lean process."""

    def __init__(self, timeout: float):
        self.timeout = timeout
        self.process: LeanProcess | None = None
        self.checkpoint: LeanEnvironmentCheckpoint | None = None
        self.last_startup_seconds = 0.0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self) -> None:
        """Close the verifier process if it has been started."""
        if self.process is not None:
            self.process.stop_safe()
            self.process = None
            self.checkpoint = None

    def verify(self, lean_code: str) -> None:
        """Raise if Lean rejects the provided declaration or times out."""
        process = None
        checkpoint = None
        self.last_startup_seconds = 0.0
        try:
            if self.process is None:
                start = perf_counter()
                try:
                    self._start()
                finally:
                    self.last_startup_seconds = perf_counter() - start
            assert self.process is not None
            assert self.checkpoint is not None

            process = self.process
            checkpoint = self.checkpoint
            response = process.send_command(lean_code, timeout=self.timeout)
            if any(
                    'sorryAx' in str(message.get('data', ''))
                    for message in response.get('messages', [])
            ):
                raise LeanInteractionException('Proof depends on sorryAx.')
        except LeanProcessException as error:
            self.close()
            raise ProofCheckProcessError(
                f'Final proof-check process failed: {error}'
            ) from error
        finally:
            if process is not None and self.process is process:
                assert checkpoint is not None
                process.rollback_to(checkpoint)

    def _start(self) -> None:
        """Start Lean and load the shared verifier environment once."""
        process = LeanProject(str(LEAN_PROJECT_DIR)).environment()
        process.__enter__()
        try:
            process.send_command('import Mathlib', timeout=self.timeout)
        except Exception:
            process.stop_safe()
            raise
        self.process = process
        self.checkpoint = process.checkpoint()


def compute_value_target(node: Node) -> float:
    """Computes the actual value for a node, to be used as a target in learning."""
    if node.is_terminal:
        node.value_target = 0
        return 0
    elif node.node_type == NodeType.OR:
        action = select_optimal_action(node)
        child_value = compute_value_target(node.children[action])
        value = -1 + child_value
        node.value_target = value
        return value
    elif node.node_type == NodeType.AND:
        value = min(compute_value_target(child) for child in node.children.values())
        node.value_target = value
        return value
    else:
        raise ValueError(f'Unknown to_play: {node.node_type}')


def extract_transitions(node: Node) -> list[tuple[Observation, Action, float]]:
    """Extracts transitions from the game."""
    if not node.is_optimal:
        return []
    assert node.node_type == NodeType.OR
    transitions = []
    while node.node_type == NodeType.OR and not node.is_terminal:
        action = select_optimal_action(node)
        transitions.append((node.observation, action, node.value_target))
        node = node.children[action]
    if node.node_type == NodeType.AND:
        for _, child in node.children.items():
            transitions.extend(extract_transitions(child))
    return transitions


def select_optimal_action(node: Node) -> Action:
    """Select the first proven action."""
    assert node.node_type == NodeType.OR
    for action, child in node.children.items():
        if child.is_optimal:
            return action
    raise ValueError('Node has no proven action.')


def final_check(
        game: Game,
        timeout: float,
        verifier: ProofVerifier | None = None,
) -> bool:
    """Checks that the proof found is actually valid."""
    game.error = None
    game.final_proof = None
    theorem = theorem_for_game(game.theorem, game.disprove)
    proof_lines = extract_proof_script(game.root)
    finished_proof = replace_sorry_proof(theorem, proof_lines)
    footer = ''
    name = theorem_name(finished_proof)
    if name is not None:
        footer = f'\n\n#print axioms {name}'
    lean_code = f'{finished_proof}{footer}'
    owns_verifier = verifier is None
    if verifier is None:
        verifier = ProofVerifier(timeout)
    try:
        verifier.verify(lean_code)
    except ProofCheckProcessError as error:
        game.error = str(error)
        output = ''
    except LeanInteractionException as error:
        game.error = 'Lean rejected the proof found by search.'
        output = str(error)
    else:
        game.final_proof = finished_proof
        return True
    finally:
        if owns_verifier:
            verifier.close()

    warning = (
            f'WARNING: {game.error}\n'
            f'Finished proof:\n{finished_proof}'
    )
    if output:
        warning += f'\nLean output:\n{output}'
    print(warning, flush=True)
    return False


def extract_proof_script(node: Node, indent: int = 2) -> list[str]:
    """Extract a Lean tactic script from the optimal proof tree."""
    if node.is_terminal:
        return []

    if node.node_type == NodeType.OR:
        action = select_optimal_action(node)
        tactic = action_to_tactic(action)
        child = node.children[action]

        if is_internal_action(tactic):
            return extract_proof_script(child, indent)
        return [
                _indent(tactic, indent),
                *extract_proof_script(child, indent),
        ]

    if node.node_type == NodeType.AND:
        lines = []
        for child in node.children.values():
            child_lines = extract_proof_script(child, indent + 2)
            lines.extend(_bullet_lines(child_lines, indent))
        return lines

    raise ValueError(f'Unknown node type: {node.node_type}')


def action_to_tactic(action: Action) -> str:
    """Convert an AlphaProof action into tactic text."""
    if isinstance(action, LeanTactic):
        return action.tactic
    return action


def _indent(text: str, spaces: int) -> str:
    prefix = ' ' * spaces
    return '\n'.join(prefix + line if line else line for line in text.splitlines())


def _bullet_lines(child_lines: list[str], indent: int) -> list[str]:
    bullet = ' ' * indent + '·'
    if not child_lines:
        return [bullet]

    first_line = child_lines[0]
    child_indent = ' ' * (indent + 2)
    if '\n' in first_line or not first_line.startswith(child_indent):
        return [bullet, *child_lines]

    return [
            f'{bullet} {first_line.removeprefix(child_indent)}',
            *child_lines[1:],
    ]
