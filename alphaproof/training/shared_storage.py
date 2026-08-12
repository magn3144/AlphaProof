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
        params = network.params
        checkpoint_path = self.checkpoints_dir / f'step_{step:07d}.pt'
        temporary_path = checkpoint_path.with_suffix('.tmp')
        torch.save(
            {
                'step': step,
                'network_params': params,
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
        if not checkpoints:
            raise FileNotFoundError(
                f'No checkpoints found under {self.checkpoints_dir}.'
            )
        checkpoint = torch.load(
            checkpoints[-1],
            map_location=network.device,
            weights_only=True,
        )
        checkpoint = typing.cast(dict[str, typing.Any], checkpoint)
        network.params = typing.cast(Params, checkpoint['network_params'])
        network.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        random.setstate(checkpoint['python_random_state'])
        torch.set_rng_state(checkpoint['torch_random_state'].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint['cuda_random_states']]
            )
        return int(checkpoint['step'])
