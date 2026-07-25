# Notes for CODEX

- Write implementations as simple as possible, reusing exsisting functionality and avoiding ovverly complicated structures.
- The actual project is in "alphaproof/".
- "pseudocode.py" is not part of the actual project. It is used as a guide for how to structure the code. The code in "alphaproof/" should closely follow the structure of "pseudocode.py".
- When you want to use py_compile, dont place the cache files in this repo.
- Always place imports at the top of files. Not inside functions.
- Backward compatability doesnt matter, it just complicates the code unnecessarily.
- Dont make fallbacks. Its better to assume that the input to a function always is in the correct format, and just let the code fail if thats not the case.
- Write code as simple as possible, without complicated structures. If you can remove more code than you write as you complete a task, thats great.
- After writing code fix any potential Pylance errors.
- Scripts that should only be executed once, like downloading a dataset, should be in the scripts folder.
- Dont write tests. Its not necessary for this project.
- LeanTree is used as a submodule. Dont edit this, just see it as a library.
- I want the main AlphaProof algorithm files (like train.py and actors.py) to be as minimal as possible, focusing on just implementing the core algorithm. Less important code like logging should be placed in separate files.


# Design choices

These are the design choices we have made so far.
They might differ from the pseudocde, which is ok.

 - Used LeanTree for interacting with Lean 4.
 - SFT dataset (state, action, proof_depth) was generated in a similar fashion as in NanoProof.
 - SFT code might be similar to NanoProof SFT code.
 - Replay buffer samples uniformly.
 - Computes tactic prior by summing token logprobs. This is used as the prior in PUCT.
 - Value head uses mean pooled encoder output.
 - Value head is currently linear layer.
 - TODO: Data for each run should be stored like this, so runs can be resumed:
 runs/
   0/
    config.json
    matchmaker_stats.json
    results.jsonl
    replay_buffer.jsonl
    checkpoints/
      step_0001000.pt
      step_0002000.pt
      step_0003000.pt
 - Actors are run sequentially *for now*, for a specific amount of iterations each.
 - Encoder called again every time a node is expanded.
 - The autoformalizer only generates one lean problem per natural language problem.
 - The dataset used for training is numina_math_lean_cleaned.jsonl for now.
 - Models used:
  - Data cleaning: Qwen3.6-27B
  - Autoformalization: Goedel-Prover-V2-32B
  - Prover: Salesforce--codet5p-220m
