import faulthandler
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

from alphaproof.core.game import Game


CRASH_FILE = 'crash.log'
STATUS_FILE = 'status.json'


class RunDiagnostics:
    """Persist the active training operation and fatal errors."""

    def __init__(self, run_dir: Path):
        self.status_path = run_dir / STATUS_FILE
        self.crash_file = (run_dir / CRASH_FILE).open('a', encoding='utf-8')
        self.status: dict[str, object] = {}
        faulthandler.enable(file=self.crash_file, all_threads=True)

    def game_started(
        self,
        request_id: str,
        game: Game,
        transition_target: int,
    ) -> None:
        """Record a game before Lean starts processing it."""
        self._write_status(
            {
                'state': 'running',
                'phase': 'actor',
                'transition_target': transition_target,
                'request_id': request_id,
                'theorem': game.theorem,
            }
        )

    def actor_started(self, transition_target: int) -> None:
        """Record the replay transition target being collected."""
        self._write_status(
            {
                'state': 'running',
                'phase': 'actor',
                'transition_target': transition_target,
            }
        )

    def learner_started(
        self,
        start_step: int,
        target_step: int,
        transition_target: int,
    ) -> None:
        """Record the learner step range currently being processed."""
        self._write_status(
            {
                'state': 'running',
                'phase': 'learner',
                'transition_target': transition_target,
                'start_step': start_step,
                'target_step': target_step,
            }
        )

    def crash(self, error: BaseException) -> None:
        """Persist an exception before external logger shutdown."""
        formatted = ''.join(traceback.format_exception(error))
        self.crash_file.write(
            f'\n{self._timestamp()} pid={os.getpid()} '
            f'context={json.dumps(self.status)}\n{formatted}'
        )
        self.crash_file.flush()
        print(formatted, file=sys.stderr, end='', flush=True)
        self._write_status(
            {
                **self.status,
                'state': 'failed',
                'error_type': type(error).__name__,
                'error': str(error),
            }
        )

    def wandb_finish_failed(self, error: Exception) -> None:
        """Append a W&B shutdown failure without replacing the run failure."""
        formatted = ''.join(traceback.format_exception(error))
        self.crash_file.write(
            f'\n{self._timestamp()} W&B shutdown failed\n{formatted}'
        )
        self.crash_file.flush()
        print(formatted, file=sys.stderr, end='', flush=True)

    def complete(self) -> None:
        """Mark the run as completed."""
        self._write_status({'state': 'completed'})

    def close(self) -> None:
        """Close the native-fault output file."""
        faulthandler.disable()
        self.crash_file.close()

    def _write_status(self, status: dict[str, object]) -> None:
        status['timestamp'] = self._timestamp()
        status['pid'] = os.getpid()
        self.status = status
        temporary_path = self.status_path.with_suffix('.tmp')
        with temporary_path.open('w', encoding='utf-8') as status_file:
            json.dump(status, status_file, indent=2)
            status_file.write('\n')
        temporary_path.replace(self.status_path)

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().astimezone().isoformat()
