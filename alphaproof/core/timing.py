from typing import Any


class TacticTiming:
    """Aggregate execution times for one tactic."""

    def __init__(self):
        self.count = 0
        self.successful_count = 0
        self.total_seconds = 0.0
        self.min_seconds: float | None = None
        self.max_seconds: float | None = None

    def add(self, seconds: float, successful: bool) -> None:
        """Add one tactic execution."""
        self.count += 1
        self.successful_count += int(successful)
        self.total_seconds += seconds
        self.min_seconds = (
            seconds if self.min_seconds is None else min(self.min_seconds, seconds)
        )
        self.max_seconds = (
            seconds if self.max_seconds is None else max(self.max_seconds, seconds)
        )

    def record(self, tactic: str) -> dict[str, Any]:
        """Return a JSON-compatible timing record."""
        return {
            'tactic': tactic,
            'count': self.count,
            'successful_count': self.successful_count,
            'total_seconds': self.total_seconds,
            'min_seconds': self.min_seconds,
            'max_seconds': self.max_seconds,
        }


class GameTimings:
    """Collect profiling data for one actor game."""

    def __init__(self):
        self.total_seconds = 0.0
        self.setup_seconds = 0.0
        self.tactic_generations: list[dict[str, int | float]] = []
        self.tactics: dict[str, TacticTiming] = {}
        self.internal_action_count = 0
        self.internal_action_seconds = 0.0
        self.final_verification_seconds: float | None = None
        self.final_verification_success = False

    @property
    def tactic_generation_seconds(self) -> float:
        """Return total time spent generating tactics."""
        return sum(
            float(expansion['seconds']) for expansion in self.tactic_generations
        )

    @property
    def tactic_execution_seconds(self) -> float:
        """Return total time spent executing Lean tactics."""
        return sum(timing.total_seconds for timing in self.tactics.values())

    def add_tactic_generation(
        self,
        simulation: int,
        state_id: int,
        seconds: float,
        num_tactics: int,
    ) -> None:
        """Record tactic generation for one node expansion."""
        self.tactic_generations.append({
            'expansion': len(self.tactic_generations) + 1,
            'simulation': simulation,
            'state_id': state_id,
            'seconds': seconds,
            'num_tactics': num_tactics,
        })

    def add_tactic_execution(
        self,
        tactic: str,
        seconds: float,
        successful: bool,
    ) -> None:
        """Record one tactic or internal action execution."""
        if is_internal_action(tactic):
            self.internal_action_count += 1
            self.internal_action_seconds += seconds
            return
        timing = self.tactics.setdefault(tactic, TacticTiming())
        timing.add(seconds, successful)

    def record(self) -> dict[str, Any]:
        """Return JSON-compatible timings for this game."""
        final_verification = None
        if self.final_verification_seconds is not None:
            final_verification = {
                'seconds': self.final_verification_seconds,
                'success': self.final_verification_success,
            }
        return {
            'total_seconds': self.total_seconds,
            'setup_seconds': self.setup_seconds,
            'tactic_generation': {
                'count': len(self.tactic_generations),
                'total_seconds': self.tactic_generation_seconds,
                'expansions': self.tactic_generations,
            },
            'tactic_execution': {
                'count': sum(timing.count for timing in self.tactics.values()),
                'total_seconds': self.tactic_execution_seconds,
                'tactics': [
                    timing.record(tactic)
                    for tactic, timing in sorted(self.tactics.items())
                ],
            },
            'internal_actions': {
                'count': self.internal_action_count,
                'total_seconds': self.internal_action_seconds,
            },
            'final_verification': final_verification,
        }


def is_internal_action(tactic: str) -> bool:
    """Return whether an action is search bookkeeping rather than a tactic."""
    return tactic == 'disprove' or tactic.startswith('focus_goal ')
