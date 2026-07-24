import secrets
from pathlib import Path
from typing import Callable

from alphaproof.core.environment import Environment
from alphaproof.core.paths import (
    DATASET_DIR,
    DEFAULT_THEOREMS_PATH,
    LEAN_PROJECT_DIR,
    MODELS_DIR,
    RUNS_DIR,
)
from leantree import LeanProject


class Config:
    """Hyperparameters and constructors used by the pseudocode pipeline."""

    def __init__(
        self,
        num_simulations: int = 250,
        batch_size: int = 10,
        num_actors: int = 1,
        num_games: int = 32,
        seed: int | None = None,
        debug: bool = False,
        lr: float = 1e-5,
        environment_ctor: Callable[[], Environment] = (
            lambda: Environment(LeanProject(str(LEAN_PROJECT_DIR)))
        ),
        tokenizer_model: str = str(MODELS_DIR / 'Salesforce--codet5p-220m'),
        dataset_path: str | Path = DEFAULT_THEOREMS_PATH,
        sft_dataset_path: str | Path = (
            DATASET_DIR / 'leantree_mathlib_state_action_pairs.train.jsonl'
        ),
        sft_fraction: float = 0.1,
        disprove_rate: float = 0.0,
        run_id: int | str = 0,
        sft_run_dir: str | Path | None = (
            RUNS_DIR / 'sft_codet5p_220m_v100_32gb'
        ),
        max_state_length: int = 640,
        max_action_length: int = 128,
        training_steps: int = 10_000,
        training_iterations: int = 8,
        checkpoint_interval: int = int(1e3),
        window_size: int = int(1e6),
        value_weight: float = 0.001,
        validation_fraction: float = 0.05,
        validation_batch_size: int = 64,
        validation_interval: int = 100,
        log_interval: int = 10,
        reward_window: int = 100,
        wandb_project: str = 'alphaproof',
        wandb_entity: str | None = None,
        wandb_tags: tuple[str, ...] = (),
    ):
        """Populate acting, search, training, and matchmaker settings."""
        ### Acting
        self.environment_ctor = environment_ctor
        self.dataset_path = Path(dataset_path)
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
        self.num_simulations = num_simulations
        self.seed = secrets.randbits(63) if seed is None else seed
        self.debug = debug
        self.tactic_timeout = 1.0
        self.final_check_timeout = 300.0

        # UCB formula
        self.pb_c_base = 3200
        self.pb_c_init = 0.001
        self.value_discount = 0.99
        self.prior_temperature = 200
        self.c_and = 64
        self.unvisited_value_penalty = 32

        # Other MCTS parameters
        self.num_sampled_actions = 6
        self.no_legal_actions_value = -40

        # Progressive sampling parameters
        self.ps_c = 0.01
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
        self.value_weight = value_weight
        self.validation_fraction = validation_fraction
        self.validation_batch_size = validation_batch_size
        self.validation_interval = validation_interval
        self.log_interval = log_interval
        self.reward_window = reward_window

        ### Logging
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_tags = wandb_tags

        # Matchmaker
        self.mm_disprove_rate = disprove_rate
        self.mm_trust_count = 8
        self.mm_fully_decided_trust_count = 12
        self.mm_proved_weight = 1e-3
        self.mm_undecided_weight = 0.1
        self.mm_simulation_failure_window = 32
        self.mm_simulation_failure_multiplier = 1.17
        self.mm_max_num_simulations = 16_000

        self.run_id = run_id
