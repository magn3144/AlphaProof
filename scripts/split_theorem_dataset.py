import argparse
import random
from pathlib import Path


SPLITS = ('train', 'validation', 'test')


def split_dataset(input_path: Path, output_dir: Path, seed: int) -> None:
    """Split a theorem JSONL dataset into deterministic 60:20:20 files."""
    with input_path.open(encoding='utf-8') as input_file:
        lines = [line.rstrip('\n') for line in input_file if line.strip()]

    random.Random(seed).shuffle(lines)
    train_end = int(len(lines) * 0.6)
    validation_end = train_end + int(len(lines) * 0.2)
    split_lines = (
        lines[:train_end],
        lines[train_end:validation_end],
        lines[validation_end:],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in zip(SPLITS, split_lines, strict=True):
        path = output_dir / f'{split}.jsonl'
        with path.open('w', encoding='utf-8') as output_file:
            for record in records:
                output_file.write(record + '\n')
        print(f'Wrote {len(records)} records to {path}')


def main() -> None:
    """Parse dataset splitting arguments."""
    parser = argparse.ArgumentParser(
        description='Split a theorem JSONL dataset into train/validation/test.'
    )
    parser.add_argument('input_path', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    split_dataset(args.input_path, args.output_dir, args.seed)


if __name__ == '__main__':
    main()
