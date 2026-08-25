from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alphaproof.core.config import Config


_PROOF_SEPARATOR = ':= by sorry'
_DECLARATION_PATTERN = re.compile(
    r'^[ \t]*(?:(?:theorem|lemma)[ \t]+([^\s(:]+)|example\b)',
    re.MULTILINE,
)


def theorem_for_game(theorem: str, disprove: bool) -> str:
    """Return the Lean theorem declaration used for this game objective."""
    if disprove:
        return negate_theorem(theorem)
    return theorem


def negate_theorem(theorem: str) -> str:
    """Create a theorem negating the fully quantified original statement."""
    before_proof, proof = _split_sorry_proof(theorem)
    declaration = _find_declaration(before_proof)
    goal_start = _find_goal_colon(before_proof)
    header = _rename_decl(before_proof[:declaration.end()], '_disproof')
    binders = before_proof[declaration.end():goal_start].strip()
    goal = before_proof[goal_start + 1:].strip()
    if binders:
        goal = f'∀ {binders}, {goal}'
    return f'{header} : ¬ ({goal}) {proof}'


def replace_goal_with_false(theorem: str) -> str:
    """Create a theorem whose hypotheses imply False."""
    before_proof, proof = _split_sorry_proof(theorem)
    goal_start = _find_goal_colon(before_proof)
    header = before_proof[:goal_start + 1]
    header = _rename_decl(header, '_exfalso')
    return f'{header} False {proof}'


def theorem_name(theorem: str) -> str | None:
    """Extract the declaration name if the theorem is named."""
    match = _find_declaration(theorem)
    if match.group(1) is None:
        return None
    return match.group(1)


def replace_sorry_proof(theorem: str, proof_lines: list[str]) -> str:
    """Replace the theorem's single sorry with a generated tactic proof."""
    before_proof, _ = _split_sorry_proof(theorem)
    proof = '\n'.join(proof_lines)
    return f'{before_proof} := by\n{proof}'


def _split_sorry_proof(theorem: str) -> tuple[str, str]:
    theorem = theorem.rstrip()
    separator_index = theorem.rfind(_PROOF_SEPARATOR)
    if separator_index + len(_PROOF_SEPARATOR) != len(theorem):
        raise ValueError(f'Expected theorem to end with `{_PROOF_SEPARATOR}`.')
    if theorem.find(_PROOF_SEPARATOR) != separator_index:
        raise ValueError(
                f'Expected exactly one `{_PROOF_SEPARATOR}` separator.'
        )
    return (
            theorem[:separator_index].rstrip(),
            theorem[separator_index:],
    )


def _find_goal_colon(header: str) -> int:
    declaration = _find_declaration(header)

    depth = 0
    opening = '([{'
    closing = ')]}'

    for index in range(declaration.end(), len(header)):
        char = header[index]
        if char in opening:
            depth += 1
        elif char in closing:
            depth -= 1
        elif char == ':' and depth == 0:
            return index

    raise ValueError('Could not find theorem goal.')


def _rename_decl(header: str, suffix: str) -> str:
    match = _find_declaration(header)
    if match.group(1) is None:
        return header
    name_start, name_end = match.span(1)
    return f'{header[:name_start]}{match.group(1)}{suffix}{header[name_end:]}'


def _find_declaration(theorem: str) -> re.Match[str]:
    """Find the final theorem-like declaration in a dataset record."""
    declarations = list(_DECLARATION_PATTERN.finditer(theorem))
    if not declarations:
        raise ValueError('Could not find theorem declaration.')
    return declarations[-1]


def make_config() -> 'Config':
    """Create the default pseudocode training configuration."""
    from alphaproof.core.config import Config

    return Config(
            num_simulations=800,
            batch_size=2048,
            num_actors=3000,
            num_games_per_actor=1,
            lr=1.0,
    )
