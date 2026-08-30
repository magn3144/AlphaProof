from __future__ import annotations

import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from alphaproof.core.actors import run_mcts
from alphaproof.core.config import Config
from alphaproof.core.environment import Environment, NodeType
from alphaproof.core.game import Game, Node, final_check
from alphaproof.formalize.goedel_prover import GoedelProver
from alphaproof.core.helper import (
        negate_theorem,
        replace_sorry_proof,
        replace_goal_with_false,
        theorem_name,
)
from alphaproof.core.network import Network
from alphaproof.core.paths import LEAN_PROJECT_DIR
from leantree import LeanProject


THEOREM_NAME = 'generated_problem'


def run_lean(
        lean_code: str,
        prefix: str = 'AlphaProofCheck',
        timeout: float | None = 30,
) -> subprocess.CompletedProcess[str]:
    """Run Lean on generated code and return the completed process."""
    LEAN_PROJECT_DIR.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(
        'w',
        dir=LEAN_PROJECT_DIR,
        suffix='.lean',
        prefix=prefix,
        delete=False,
    ) as file:
        file.write(lean_code)
        file_path = Path(file.name)

    try:
        result = subprocess.run(
                ['lake', 'env', 'lean', file_path.name],
                cwd=LEAN_PROJECT_DIR,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
        )
    finally:
        file_path.unlink(missing_ok=True)

    return result


def run_lean_check(lean_code: str) -> bool:
    """Run Lean on generated code and reject sorry-backed proofs."""
    result = run_lean(lean_code, prefix='AlphaProofFinalCheck')
    return result.returncode == 0


def sample_auto_formalization(
        nl_problem: str,
        goedel_prover: GoedelProver,
) -> str:
    """Samples a Lean formalization using Goedel Prover."""
    prompt = f"""<task>
Translate the natural language math problem into one Lean 4 theorem statement.
</task>

<instructions>
Return only the Lean theorem statement.
Do not include imports, a proof, ":= by", "sorry", comments, markdown, or explanation.
</instructions>

<example>
<problem>
For every natural number n, n plus zero equals n.
</problem>
<output>
theorem problem_name (n : Nat) : n + 0 = n
</output>
</example>

<problem>
{nl_problem}
</problem>
"""
    return goedel_prover.sample(prompt, num_samples=1)[0]


def extract_lean_code(sample: str) -> str:
    """Extracts the Lean code from a sample."""
    statement = re.sub(
        r'^\s*theorem\s+\S+',
        f'theorem {THEOREM_NAME}',
        sample,
        count=1,
    )
    return statement + ' := by\n  sorry'


def lean_is_valid_syntax(
        lean_statement: str,
        environment: Environment | None = None,
) -> bool:
    """Validates Lean code for syntax and common linting errors."""
    if environment is not None:
        try:
            environment.initial_state(lean_statement)
            return True
        except Exception:
            return False

    with Environment(LeanProject(str(LEAN_PROJECT_DIR))) as environment:
        return lean_is_valid_syntax(lean_statement, environment)


def lean_is_complete_proof(lean_code: str) -> bool:
    """Checks if Lean accepts the code as a full proof."""
    footer = ''
    name = theorem_name(lean_code)
    if name is not None:
        footer = f'\n\n#print axioms {name}'

    check_code = (
            'import Mathlib\n'
            'set_option warningAsError true\n\n'
            f'{lean_code}{footer}\n'
    )
    return run_lean_check(check_code)


def lean_replace_goal_with_false(lean_code: str) -> str:
    """Creates a new statement where the goal is to prove a contradiction is among the hypotheses."""
    return replace_goal_with_false(lean_code)


def lean_negate_statement(lean_code: str) -> str:
    """Creates a new statement where the goal is to disprove the original statement."""
    return negate_theorem(lean_code)


def is_provable(
        lean_statement: str,
        config: Config,
        network: Network,
) -> bool:
    """Runs Alphaproof to check if the Lean statement is provable."""
    game = Game(
            theorem=lean_statement,
            disprove=False,
            num_simulations=config.num_simulations,
    )
    try:
        with config.environment_ctor() as environment:
            state = environment.initial_state(game.theorem)
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
            run_mcts(config, game, network, environment)
            if game.root.is_optimal:
                game.root.is_optimal = final_check(
                        game,
                        environment,
                        config.final_check_timeout,
                )
        return game.root.is_optimal
    except Exception:
        return False


def has_trivial_counterexample(lean_statement: str) -> bool:
    """Run a modified version of Lean's `plausible` tactic."""
    try:
        declaration = replace_sorry_proof(
                lean_statement,
                ['  plausible (config := { quiet := true })'],
        )
    except ValueError:
        return False

    lean_code = f'import Mathlib\nimport Plausible\n\n{declaration}\n'
    result = run_lean(lean_code, prefix='AlphaProofPlausible')
    output = result.stdout + result.stderr
    return 'Found a counter-example!' in output


def is_easily_provable(lean_statement: str) -> bool:
    """Checks if the statement can be easily decided by an ad-hoc set of simple tactics."""
    # try to prove the statement
    for tactic in [
            "simp",
            "norm_num",
            "abel",
            "nlinarith",
            "linarith",
            "ring",
            "aesop",
            "trivial",
    ]:
        if lean_is_complete_proof(lean_statement + " := by " + tactic):
            return True

    return False


def deformalize_lean(
        lean_statement: str,
        model: GoedelProver,
) -> str:
    """Deformalizes a Lean statement into a natural language statement."""
    prompt = f"""<task>
Translate the Lean theorem statement into a natural language math statement.
</task>

<instructions>
Return only the natural language statement.
Do not include a proof, commentary, markdown, or Lean code.
</instructions>

<lean>
{lean_statement}
</lean>
"""
    return model.sample(prompt, num_samples=1)[0].strip()


def check_cycle_consistency(
        original_statement: str,
        deformalized_statement: str,
        model: GoedelProver,
) -> bool:
    """Checks if the original and deformalized statements are equivalent."""
    prompt = f"""<task>
Decide whether two natural language math statements have the same mathematical meaning.
</task>

<instructions>
Answer only YES or NO.
Answer YES if the statements are mathematically equivalent.
Answer NO if either statement is stronger, weaker, or about different objects.
</instructions>

<statement_a>
{original_statement}
</statement_a>

<statement_b>
{deformalized_statement}
</statement_b>
"""
    answer = model.sample(prompt, num_samples=1)[0].strip().upper()
    return answer.startswith('YES')


def auto_formalize_problem(
        nl_problem: str,
        n_samples: int,
        goedel_prover: GoedelProver,
        config: Config,
        network: Network,
) -> str | None:
    """Translates for a natural language statement into a Lean statement."""

    samples = [
            sample_auto_formalization(nl_problem, goedel_prover)
            for _ in range(n_samples)
    ]
    lean_problems = [extract_lean_code(sample) for sample in samples]
    vote_counter = Counter(lean_problems)  # Deduplicate and count votes.

    problems_with_votes = [
            (votes, problem) for problem, votes in vote_counter.items()
    ]
    problems_with_votes.sort(reverse=True)  # Order by votes (most to least).

    with Environment(LeanProject(str(LEAN_PROJECT_DIR))) as environment:
        # Find the most-voted candidate that passes sanity checking.
        for _, lean_problem in problems_with_votes:
            # Remove samples that do not have a valid Lean syntax.
            if not lean_is_valid_syntax(lean_problem, environment):
                continue

            # Create two new Lean statements: one where the goal is to disprove the
            # original statement, and one where the goal is to prove the hypotheses
            # are contradictory.
            lean_negated = lean_negate_statement(lean_problem)
            lean_exfalso = lean_replace_goal_with_false(lean_problem)

            # Discard statements that have a single-tactic proof.
            if (
                    is_easily_provable(lean_problem)
                    or is_easily_provable(lean_negated)
                    or is_easily_provable(lean_exfalso)
            ):
                continue

            # Discard statements that have a trivial counterexamples.
            if has_trivial_counterexample(lean_problem):
                continue

            # Check cycle consistency: ask a model to deformalize a statement,
            # then ask if the original and deformalized statements are equivalent.
            deformalized_stmt = deformalize_lean(lean_problem, goedel_prover)
            if not check_cycle_consistency(
                    nl_problem,
                    deformalized_stmt,
                    goedel_prover,
            ):
                continue

            # Use small-budget Alphaproof to check if the statement is disprovable or
            # the hypotheses are contradictory.
            if is_provable(
                    lean_negated,
                    config,
                    network,
            ) or is_provable(
                    lean_exfalso,
                    config,
                    network,
            ):
                continue

            return lean_problem

    # All samples failed.
    return None
