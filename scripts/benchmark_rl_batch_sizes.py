import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch

from alphaproof.core.config import Config, RL_PRECISIONS
from alphaproof.core.network import Network


GiB = 1024 ** 3


def memory_gib(value: int) -> float:
    """Convert bytes to GiB."""
    return value / GiB


def cuda_memory() -> dict[str, float]:
    """Return current and peak CUDA memory measurements."""
    return {
        'allocated_gib': memory_gib(torch.cuda.memory_allocated()),
        'reserved_gib': memory_gib(torch.cuda.memory_reserved()),
        'peak_allocated_gib': memory_gib(torch.cuda.max_memory_allocated()),
        'peak_reserved_gib': memory_gib(torch.cuda.max_memory_reserved()),
    }


def is_cuda_oom(error: BaseException) -> bool:
    """Return whether an exception is a CUDA out-of-memory failure."""
    return isinstance(error, torch.OutOfMemoryError) or (
        isinstance(error, RuntimeError)
        and 'out of memory' in str(error).lower()
    )


def run_candidate(
    batch_size: int,
    repeats: int,
    operation: Callable[[int], Any],
) -> dict[str, Any]:
    """Run one steady-state batch-size candidate and record its peak memory."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        for _ in range(repeats):
            result = operation(batch_size)
            del result
            torch.cuda.synchronize()
    except BaseException as error:
        if not is_cuda_oom(error):
            raise
        return {
            'batch_size': batch_size,
            'success': False,
            'seconds': time.perf_counter() - started,
            'error': str(error),
            **cuda_memory(),
        }
    return {
        'batch_size': batch_size,
        'success': True,
        'seconds': time.perf_counter() - started,
        **cuda_memory(),
    }


def find_largest_batch_size(
    repeats: int,
    maximum: int,
    operation: Callable[[int], Any],
) -> tuple[int, list[dict[str, Any]]]:
    """Test consecutive sizes until the first OOM."""
    results = []
    for batch_size in range(1, maximum + 1):
        result = run_candidate(batch_size, repeats, operation)
        results.append(result)
        print(json.dumps(result), flush=True)
        if not result['success']:
            return batch_size - 1, results
    raise RuntimeError(
        f'No OOM through batch size {maximum}; increase --maximum-batch-size.'
    )


def maximum_length_state(network: Network) -> str:
    """Construct text that fills the complete configured encoder length."""
    state = 'x ' * (network.max_state_length * 4)
    encoded = network.tokenizer(
        state,
        max_length=network.max_state_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    if int(encoded.attention_mask.sum()) != network.max_state_length:
        raise RuntimeError('Synthetic state did not fill the encoder sequence.')
    return state


def training_batch(
    network: Network,
    batch_size: int,
) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
    """Create a full-length synthetic learner batch."""
    token_id = int(network.model.config.eos_token_id)
    observation = torch.full(
        (network.max_state_length,), token_id, dtype=torch.long
    )
    action = torch.full(
        (network.max_action_length,), token_id, dtype=torch.long
    )
    return [(observation, action, -1.0)] * batch_size


def parse_args() -> argparse.Namespace:
    """Parse benchmark controls."""
    parser = argparse.ArgumentParser(
        description='Find worst-case 770M RL learner and solver batch sizes.'
    )
    parser.add_argument('--dtype', choices=RL_PRECISIONS, default='float32')
    parser.add_argument('--maximum-batch-size', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run worst-case learner and solver batch-size searches."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('This benchmark requires CUDA.')

    config = Config(batch_size=1, dtype=args.dtype)
    network = Network(config)
    parameter_dtype = (
        torch.bfloat16 if args.dtype == 'bfloat16' else torch.float32
    )
    if network.model.dtype != parameter_dtype:
        raise RuntimeError(
            f'Expected {parameter_dtype} model, got {network.model.dtype}.'
        )

    state = maximum_length_state(network)
    network.model.generation_config.eos_token_id = None

    print('Training batch sizes', flush=True)
    largest_training, training_results = find_largest_batch_size(
        args.repeats,
        args.maximum_batch_size,
        lambda size: network.update(training_batch(network, size)),
    )

    network.optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    network.update(training_batch(network, 1))

    print('Solving batch sizes', flush=True)
    largest_solving, solving_results = find_largest_batch_size(
        args.repeats,
        args.maximum_batch_size,
        lambda size: network.sample_batch([state] * size),
    )

    report = {
        'gpu': torch.cuda.get_device_name(),
        'total_memory_gib': memory_gib(
            torch.cuda.get_device_properties(0).total_memory
        ),
        'precision': args.dtype,
        'parameter_dtype': str(network.model.dtype),
        'autocast_dtype': 'torch.bfloat16' if network.mixed_precision else None,
        'max_state_length': network.max_state_length,
        'max_training_action_length': network.max_action_length,
        'max_generated_action_length': network.rollout_max_action_length,
        'num_sampled_actions': network.num_sampled_actions,
        'repeats': args.repeats,
        'largest_training_batch_size': largest_training,
        'largest_solving_batch_size': largest_solving,
        'training': training_results,
        'solving': solving_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
