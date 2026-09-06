"""Create reproducible fractional subsets of JSONL dataset splits.

Input filenames must end in train.jsonl, validation.jsonl, or test.jsonl.
"""

import argparse
import random
from pathlib import Path


def subset_dataset(
    input_paths: list[Path], output_prefix: Path, denominators: list[int], seed: int
) -> None:
    for denominator in denominators:
        output_dir = output_prefix.with_name(f'{output_prefix.name}_1_{denominator}')
        output_dir.mkdir()
        for input_path in input_paths:
            split = input_path.stem.rsplit('.', 1)[-1]
            lines = input_path.read_text(encoding='utf-8').splitlines(keepends=True)
            records = random.Random(seed).sample(lines, len(lines) // denominator)
            path = output_dir / f'{split}.jsonl'
            path.write_text(''.join(records), encoding='utf-8')
            print(f'Wrote {len(records)} records to {path}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_prefix', type=Path)
    parser.add_argument('--input-paths', type=Path, nargs='+', required=True)
    parser.add_argument('--denominators', type=int, nargs='+', required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    subset_dataset(args.input_paths, args.output_prefix, args.denominators, args.seed)


if __name__ == '__main__':
    main()
