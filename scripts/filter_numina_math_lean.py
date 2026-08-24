"""Keep only NanoProof-whitelisted NuminaMath-LEAN theorems."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from alphaproof.core.paths import DATASET_DIR


NANOPROOF_COMMIT = '4f410c6e7a820557501848f954402514201322a5'
WHITELIST_URL = (
    'https://raw.githubusercontent.com/Kripner/nanoproof/'
    f'{NANOPROOF_COMMIT}/data/whitelists/'
    'numinamath.parquet.whitelist.v4.27.0.json'
)
DEFAULT_INPUT_DIR = DATASET_DIR / 'numina_math_lean_cleaned'
DEFAULT_OUTPUT_DIR = DATASET_DIR / 'numina_math_lean_passing'
DEFAULT_WHITELIST_PATH = (
    DATASET_DIR / 'numinamath.parquet.whitelist.v4.27.0.json'
)
SPLITS = ('train', 'validation', 'test')


def theorem_hash(theorem: str) -> str:
    """Return the content hash used by NanoProof's whitelist."""
    return hashlib.sha256(theorem.encode('utf-8')).hexdigest()[:16]


def download_whitelist(path: Path) -> None:
    """Download NanoProof's pinned Lean v4.27.0 whitelist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    urllib.request.urlretrieve(WHITELIST_URL, temporary_path)
    temporary_path.replace(path)


def load_passing_hashes(path: Path) -> set[str]:
    """Load hashes of theorems that initialize successfully."""
    with path.open(encoding='utf-8') as whitelist_file:
        whitelist = json.load(whitelist_file)
    if whitelist['lean_version'] != 'v4.27.0':
        raise ValueError('Expected a Lean v4.27.0 whitelist.')
    return set(whitelist['passing'])


def filter_split(input_path: Path, output_path: Path, passing: set[str]) -> int:
    """Write only known-passing theorem records and return their count."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + '.tmp')
    kept = 0
    with input_path.open(encoding='utf-8') as input_file, temporary_path.open(
        'w', encoding='utf-8'
    ) as output_file:
        for line in input_file:
            record = json.loads(line)
            if theorem_hash(record['theorem']) in passing:
                output_file.write(json.dumps(record) + '\n')
                kept += 1
    temporary_path.replace(output_path)
    return kept


def main() -> None:
    """Download the whitelist and create passing-only dataset splits."""
    parser = argparse.ArgumentParser(
        description='Filter NuminaMath-LEAN with NanoProof\'s whitelist.'
    )
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        '--whitelist-path', type=Path, default=DEFAULT_WHITELIST_PATH
    )
    args = parser.parse_args()

    download_whitelist(args.whitelist_path)
    passing = load_passing_hashes(args.whitelist_path)
    for split in SPLITS:
        input_path = args.input_dir / f'{split}.jsonl'
        output_path = args.output_dir / f'{split}.jsonl'
        kept = filter_split(input_path, output_path, passing)
        print(f'Wrote {kept:,} passing theorems to {output_path}')


if __name__ == '__main__':
    main()
