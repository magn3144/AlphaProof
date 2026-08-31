"""Weights & Biases logging for supervised fine-tuning."""

from pathlib import Path
from typing import Any

import wandb

from alphaproof.core.config import SFTConfig, serializable_config
from alphaproof.training.resource_monitor import (
    ResourceMonitor,
    start_resource_monitor,
)


class SFTLogger:
    """Log step-based SFT metrics and manage the W&B run lifecycle."""

    def __init__(
        self,
        run_name: str,
        run_dir: Path,
        resume: bool,
        wandb_run_id: str,
        config: SFTConfig,
    ):
        self.run: Any = wandb.init(
            project=config.wandb_project,
            dir=run_dir,
            name=config.wandb_name or run_name,
            id=wandb_run_id,
            mode=config.wandb_mode,
            resume='allow' if resume else 'never',
            config=serializable_config(config),
            settings=wandb.Settings(
                finish_timeout=60.0,
                finish_timeout_raises=True,
            ),
        )
        self.run.define_metric('sft/step')
        self.run.define_metric('train/*', step_metric='sft/step')
        self.run.define_metric('validation/*', step_metric='sft/step')
        self.run.define_metric('resources/*')
        self.resource_monitor: ResourceMonitor | None = start_resource_monitor(
            self.run
        )

    def log_training(
        self,
        step: int,
        loss: float,
        policy_loss: float,
        value_loss: float,
        learning_rate: float,
        examples_seen: int,
    ) -> None:
        """Log one optimizer update."""
        self.run.log(
            {
                'sft/step': step,
                'train/loss': loss,
                'train/policy_loss': policy_loss,
                'train/value_loss': value_loss,
                'train/learning_rate': learning_rate,
                'train/examples_seen': examples_seen,
            }
        )

    def log_validation(
        self,
        step: int,
        metrics: dict[str, float],
        samples: int,
    ) -> None:
        """Log a validation pass."""
        self.run.log(
            {
                'sft/step': step,
                **{
                    f'validation/{name}': value
                    for name, value in metrics.items()
                },
                'validation/samples': samples,
            }
        )

    def log_epoch(
        self,
        step: int,
        epoch: int,
        training_metrics: dict[str, float],
        validation_metrics: dict[str, float],
        validation_samples: int,
    ) -> None:
        """Log epoch aggregates and the full validation pass."""
        self.run.log(
            {
                'sft/step': step,
                'train/epoch': epoch,
                **{
                    f'train/epoch_{name}': value
                    for name, value in training_metrics.items()
                },
                **{
                    f'validation/{name}': value
                    for name, value in validation_metrics.items()
                },
                'validation/samples': validation_samples,
            }
        )

    def finish(self, exit_code: int) -> None:
        """Close the W&B run."""
        if self.resource_monitor is not None:
            self.resource_monitor.stop()
        self.run.finish(exit_code=exit_code)
