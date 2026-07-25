import argparse
import json
from pathlib import Path
from typing import Any

from alphaproof.core.config import Config


CONFIG_FILE = 'config.json'


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    """Convert CLI paths to JSON-compatible strings."""
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }


def serializable_config(config: Config) -> dict[str, Any]:
    """Convert the full AlphaProof configuration to JSON values."""
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(config).items()
        if name != 'environment_ctor'
    }


def save_run_config(
    run_dir: Path,
    args: argparse.Namespace,
    config: Config,
) -> None:
    """Save CLI and complete algorithm configuration for a new run."""
    with (run_dir / CONFIG_FILE).open('w', encoding='utf-8') as config_file:
        json.dump(
            {
                'args': serializable_args(args),
                'config': serializable_config(config),
            },
            config_file,
            indent=2,
        )
        config_file.write('\n')
