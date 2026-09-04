import json
import re
from pathlib import Path
from typing import Any

from alphaproof.core.config import Config, SFTConfig, serializable_config


CONFIG_PATTERN = re.compile(r'config_(\d+)\.json')


def _run_configs(run_dir: Path) -> list[tuple[int, Path]]:
    configs = []
    for path in run_dir.glob('config_*.json'):
        match = CONFIG_PATTERN.fullmatch(path.name)
        if match is not None:
            configs.append((int(match.group(1)), path))
    return configs


def has_run_config(run_dir: Path) -> bool:
    """Return whether a run has at least one versioned configuration."""
    return bool(_run_configs(run_dir))


def save_run_config(
    run_dir: Path,
    config: Config | SFTConfig,
    wandb_run_id: str,
) -> None:
    """Save the resolved experiment section and runtime metadata."""
    configs = _run_configs(run_dir)
    version = max((version for version, _ in configs), default=0) + 1
    config_path = run_dir / f'config_{version:02d}.json'
    with config_path.open('x', encoding='utf-8') as config_file:
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
    """Load a run's latest resolved configuration and runtime metadata."""
    configs = _run_configs(run_dir)
    if not configs:
        raise FileNotFoundError(
            f'Run configuration does not exist under: {run_dir}'
        )
    _, config_path = max(configs)
    with config_path.open(encoding='utf-8') as config_file:
        return json.load(config_file)


def changed_config_fields(
    old_config: Config | SFTConfig,
    new_config: Config | SFTConfig,
) -> list[str]:
    """Return the names of persisted configuration fields that changed."""
    old_values = serializable_config(old_config)
    new_values = serializable_config(new_config)
    return sorted(
        name
        for name in old_values.keys() | new_values.keys()
        if old_values.get(name) != new_values.get(name)
    )
