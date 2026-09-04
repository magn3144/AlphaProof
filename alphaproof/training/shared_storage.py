import random
import typing
from pathlib import Path

import torch

from alphaproof.core.network import Network, Params


class SharedStorage:
    """Latest actor parameters and persistent learner checkpoints."""

    def __init__(self, run_dir: Path):
        """Initialize storage under one training run."""
        self.checkpoints_dir = run_dir / 'checkpoints'
        self.checkpoints_dir.mkdir(exist_ok=True)

    def save_checkpoint(self, step: int, network: Network) -> Path:
        """Atomically save learner and optimizer state."""
        checkpoint_path = self.checkpoints_dir / f'step_{step:07d}.pt'
        return self._save_checkpoint(checkpoint_path, step, network)

    def save_latest_checkpoint(self, step: int, network: Network) -> Path:
        """Atomically replace the checkpoint for the latest training round."""
        checkpoint_path = self.checkpoints_dir / 'latest.pt'
        return self._save_checkpoint(checkpoint_path, step, network)

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        step: int,
        network: Network,
    ) -> Path:
        """Save a checkpoint atomically at the requested path."""
        temporary_path = checkpoint_path.with_suffix('.tmp')
        torch.save(
            {
                'step': step,
                'network_params': network.params,
                'optimizer_state_dict': network.optimizer.state_dict(),
                'python_random_state': random.getstate(),
                'torch_random_state': torch.get_rng_state(),
                'cuda_random_states': (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else []
                ),
            },
            temporary_path,
        )
        temporary_path.replace(checkpoint_path)
        return checkpoint_path

    def load_latest_checkpoint(self, network: Network) -> int:
        """Restore the latest learner checkpoint and return its step."""
        checkpoints = sorted(self.checkpoints_dir.glob('step_*.pt'))
        latest_path = self.checkpoints_dir / 'latest.pt'
        if latest_path.is_file():
            checkpoints.append(latest_path)
        if not checkpoints:
            raise FileNotFoundError(
                f'No checkpoints found under {self.checkpoints_dir}.'
            )
        loaded_checkpoints = [
            typing.cast(
                dict[str, typing.Any],
                torch.load(
                    path,
                    map_location=network.device,
                    weights_only=True,
                ),
            )
            for path in checkpoints[-2:]
        ]
        checkpoint = max(loaded_checkpoints, key=lambda item: int(item['step']))
        network.params = typing.cast(Params, checkpoint['network_params'])
        network.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        random.setstate(checkpoint['python_random_state'])
        torch.set_rng_state(checkpoint['torch_random_state'].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint['cuda_random_states']]
            )
        return int(checkpoint['step'])
