# AlphaProof Training Project (unofficial)
AlphaProof is a state-of-the-art reinforcement learning system developed by Google DeepMind, designed to formalize and solve complex, IMO-level mathematical problems. As the full source code has not been publicly released, I am implementing the training and inference pipelines for AlphaProof as part of my master's thesis.

This implementation is based on the official pseudocode provided by the DeepMind team. Notably, I have chosen to omit TTRL as it falls outside the scope of my research and requires computational resources beyond the scale of this project.

Disclaimers:
 - **Work in Progress**: This repository is currently under active development.
 - **Implementation Variance**: This version may differ from the original DeepMind implementation and is not guaranteed to achieve the same performance benchmarks.
 - **Academic Use**: This project is intended strictly for academic and research purposes.

## Training

Download and cheaply clean the already-formalized NuminaMath-LEAN statements:

```bash
python -m alphaproof.data.numina_math_lean download
python -m alphaproof.data.numina_math_lean clean
```

This writes the raw parquet and cleaned theorem JSONL under `data/dataset/`.
Split the cleaned theorem file into the directory format used by training:

```bash
python scripts/split_theorem_dataset.py \
  data/dataset/numina_math_lean_cleaned.jsonl \
  data/dataset/numina_math_lean_cleaned
```

Every theorem dataset passed to RL training must contain exactly the split
files `train.jsonl`, `validation.jsonl`, and `test.jsonl`.
Lean compatibility is checked lazily when training initializes each theorem;
statements that do not elaborate are recorded as invalid and not sampled again.

Start a reinforcement-learning run from the default SFT checkpoint:

```bash
python -m alphaproof.training.train rl_run --wandb-mode online
```

The default single-actor profile runs 32 games starting at 250 simulations per
attempt, with the matchmaker increasing difficult attempts up to 16,000
simulations. It performs 10,000 learner updates across eight alternating
actor-learner iterations. Wall time depends on theorem difficulty and hardware.
Use `--wandb-mode disabled` to train without W&B.

Each RL run writes per-game profiling data to `timings.jsonl`. Summarize total
game time, node tactic generation, tactic execution, and final verification with:

```bash
python scripts/summarize_rl_timings.py data/runs/<run_name>
```

### Debugging RL training on DTU voltash

Install the VS Code Python debugger in the project environment once:

```bash
uv pip install --python .venv/bin/python debugpy gnureadline
```

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Attach to voltash training",
      "type": "debugpy",
      "request": "attach",
      "connect": {
        "host": "n-62-20-1",
        "port": 5678
      },
      "justMyCode": true
    }
  ]
}
```

For each debugging session, open the repository in VS Code through Remote SSH,
then enter a V100 node and print its hostname:

```bash
voltash
cd /work3/s204164/mini-alphaproof
hostname
```

Update `host` in `.vscode/launch.json` if the printed hostname differs, then
start training and wait for the debugger:

```bash
.venv/bin/python -m debugpy \
  --listen 0.0.0.0:5678 \
  --wait-for-client \
  -m alphaproof.training.train \
  rl_debug_01 \
  --dataset-dir data/dataset/codex_theorems \
  --disprove-rate 0 \
  --num-games 4 \
  --num-simulations 16 \
  --training-iterations 1 \
  --training-steps 2 \
  --batch-size 1
```

Use a new run name for a fresh run. In VS Code, set breakpoints, open **Run and
Debug** (`Ctrl+Shift+D`), select **Attach to voltash training**, and press `F5`.
Keep the voltash terminal open throughout the session. The main debugger keys
are `F10` to step over, `F11` to step into, `Shift+F11` to step out, `F5` to
continue, and `Shift+F5` to stop.

To continue a stopped run instead of creating a fresh one:

```bash
.venv/bin/python -m debugpy \
  --listen 0.0.0.0:5678 \
  --wait-for-client \
  -m alphaproof.training.train \
  rl_debug_01 \
  --resume
```

## Inference

Inference reads one JSONL record per theorem:

```json
{"request_id":"example","theorem":"theorem example : True := by sorry","seed":0}
```

Run the batch from either an SFT or RL checkpoint:

```bash
python -m alphaproof.inference.infer \
  --input requests.jsonl \
  --output results.jsonl \
  --num-simulations 250 \
  --no-stop-on-solution
```

The network is loaded once and each input theorem receives one tree search.
Each output record contains the status, tactic proof, duration, and complete
AND-OR search tree. A single theorem is represented by a one-record input file.
By default inference stops when it finds a proof; `--no-stop-on-solution` runs
the complete simulation budget.

## Interactive environment

Start the backend with the model used by the Agent tab:

```bash
.venv/bin/python -m alphaproof.inference.interactive_env \
  --num-simulations 64 \
  --num-sampled-actions 3
```

Then start the React client in another terminal:

```bash
cd frontend
npm run dev
```

The Manual tab supports saved tactic branches and full AND-OR proof trees. The
Agent tab runs the normal MCTS search and polls the same tree view while the
search is active.
