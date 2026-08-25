import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from alphaproof.core.environment import Environment
from alphaproof.core.paths import (
    DATASET_DIR,
    DEFAULT_THEOREMS_DIR,
    LEAN_PROJECT_DIR,
    MODELS_DIR,
    RUNS_DIR,
)
from leantree import LeanProject


@dataclass(frozen=True)
class SFTConfig:
    """Default supervised fine-tuning hyperparameters."""

    train_input: Path = (
        DATASET_DIR / 'leantree_mathlib_state_action_pairs.train.jsonl'
    )
    validation_input: Path = (
        DATASET_DIR / 'leantree_mathlib_state_action_pairs.validation.jsonl'
    )
    model: Path = MODELS_DIR / 'Salesforce--codet5p-770m'
    epochs: int = 1
    checkpoints_per_epoch: int = 1
    num_pairs: int | None = None
    num_validation_pairs: int | None = None
    batch_size: int = 8
    learning_rate: float = 5e-5
    value_weight: float = 0.001
    max_state_length: int = 640
    max_action_length: int = 128
    max_grad_norm: float = 1.0
    log_every: int = 100
    validation_interval: int = 500
    validation_samples: int = 512
    wandb_name: str | None = None
    wandb_mode: str = 'disabled'
    seed: int = 0
    device: str = 'auto'
    dtype: str = 'bfloat16'


class Config:
    """Hyperparameters and constructors used by the pseudocode pipeline."""

    def __init__(
        self,
        num_simulations: int = 64,
        batch_size: int = 20,
        num_actors: int = 1,
        num_games: int = 50_000,
        inference_batch_size: int = 4,
        inference_batch_timeout: float = 0.05,
        num_sampled_actions: int = 6,
        tactic_timeout: float = 1.0,
        final_check_timeout: float = 300.0,
        seed: int | None = None,
        debug: bool = False,
        lr: float = 1e-5,
        environment_ctor: Callable[[], Environment] = (
            lambda: Environment(LeanProject(str(LEAN_PROJECT_DIR)))
        ),
        tokenizer_model: str = str(MODELS_DIR / 'Salesforce--codet5p-770m'),
        dataset_dir: str | Path = DEFAULT_THEOREMS_DIR,
        sft_dataset_path: str | Path = (
            DATASET_DIR / 'leantree_mathlib_state_action_pairs.train.jsonl'
        ),
        sft_fraction: float = 0.1,
        disprove_rate: float = 0.0,
        run_id: int | str = 0,
        sft_run_dir: str | Path | None = (
            RUNS_DIR / 'sft_codet5p_770m_v100_32gb'
        ),
        max_state_length: int = 640,
        max_action_length: int = 128,
        rollout_max_action_length: int = 32,
        training_steps: int = 5_000,
        training_iterations: int = 1_000,
        checkpoint_interval: int = 250,
        window_size: int = 250_000,
        value_weight: float = 0.01,
        validation_fraction: float = 0.05,
        validation_batch_size: int = 64,
        validation_interval: int = 100,
        theorem_validation_interval_games: int = 2_500,
        theorem_validation_num_theorems: int = 20,
        log_interval: int = 10,
        reward_window: int = 100,
        wandb_project: str = 'alphaproof',
        wandb_entity: str | None = None,
        wandb_tags: tuple[str, ...] = (),
    ):
        """Populate acting, search, training, and matchmaker settings."""
        ### Acting
        self.environment_ctor = environment_ctor
        self.dataset_dir = Path(dataset_dir)
        self.train_dataset_path = self.dataset_dir / 'train.jsonl'
        self.validation_dataset_path = self.dataset_dir / 'validation.jsonl'
        self.test_dataset_path = self.dataset_dir / 'test.jsonl'
        self.sft_dataset_path = Path(sft_dataset_path)
        self.sft_run_dir = Path(sft_run_dir) if sft_run_dir is not None else None
        if self.sft_run_dir is None:
            self.tokenizer_model = tokenizer_model
            self.initial_params_path = None
        else:
            self.tokenizer_model = str(self.sft_run_dir / 'model_source')
            self.initial_params_path = self.sft_run_dir / 'network_params.pt'
        self.num_actors = num_actors
        self.num_games = num_games
        self.inference_batch_size = inference_batch_size
        self.inference_batch_timeout = inference_batch_timeout
        self.num_simulations = num_simulations
        self.num_sampled_actions = num_sampled_actions
        self.seed = secrets.randbits(63) if seed is None else seed
        self.debug = debug
        self.tactic_timeout = tactic_timeout
        self.final_check_timeout = final_check_timeout

        # UCB formula
        self.pb_c_base = 200
        self.pb_c_init = 0.001
        self.value_discount = 0.98
        self.prior_temperature = 200
        self.c_and = 64
        self.unvisited_value_penalty = 16

        # Other MCTS parameters
        self.no_legal_actions_value = -5

        # Progressive sampling parameters
        self.ps_c = 0.1
        self.ps_alpha = 0.6

        # Value predictions
        self.num_value_bins = 64

        ### Training
        self.training_steps = training_steps
        self.training_iterations = training_iterations
        self.checkpoint_interval = checkpoint_interval
        self.window_size = window_size
        self.batch_size = batch_size
        self.sft_fraction = sft_fraction
        self.max_state_length = max_state_length
        self.max_action_length = max_action_length
        self.lr = lr
        self.rollout_max_action_length = rollout_max_action_length
        self.value_weight = value_weight
        self.validation_fraction = validation_fraction
        self.validation_batch_size = validation_batch_size
        self.validation_interval = validation_interval
        self.theorem_validation_interval_games = (
            theorem_validation_interval_games
        )
        self.theorem_validation_num_theorems = theorem_validation_num_theorems
        self.log_interval = log_interval
        self.reward_window = reward_window

        ### Logging
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_tags = wandb_tags

        # Matchmaker
        self.mm_disprove_rate = disprove_rate
        self.mm_trust_count = 4
        self.mm_fully_decided_trust_count = 6
        self.mm_proved_weight = 1e-3
        self.mm_undecided_weight = 0.1
        self.mm_simulation_failure_window = 4
        self.mm_simulation_failure_multiplier = 1.5
        self.mm_max_num_simulations = 1_024

        self.run_id = run_id
