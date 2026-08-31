import json
from pathlib import Path
from typing import Any

from alphaproof.core.config import Config, SFTConfig, serializable_config


CONFIG_FILE = 'config.json'


def save_run_config(
    run_dir: Path,
    config: Config | SFTConfig,
    wandb_run_id: str,
) -> None:
    """Save the resolved experiment section and runtime metadata."""
    with (run_dir / CONFIG_FILE).open('w', encoding='utf-8') as config_file:
        json.dump(
            {
                'config': serializable_config(config),
                'wandb_run_id': wandb_run_id,
            },
            config_file,
            indent=2,
        )
        config_file.write('\n')


def load_run_config(run_dir: Path) -> dict[str, Any]:
    """Load a run's resolved configuration and runtime metadata."""
    config_path = run_dir / CONFIG_FILE
    if not config_path.is_file():
        raise FileNotFoundError(f'Run configuration does not exist: {config_path}')
    with config_path.open(encoding='utf-8') as config_file:
        return json.load(config_file)
