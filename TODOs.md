## AlphaProof implementation
- [X] Benchmark execution times of different tasks in the code.
- [ ] Make rollout agents run in parallel.
- [X] CLI for training.
- [X] CLI for inference.
- [X] Fix discrepancies between implementation and original AlphaProof paper.

## Data cleaning / formalization
- [X] Fix errors in data cleaning script.
- [X] Test data cleaning script with timer and LLM otputs printed to see why it takes so long to execute.
- [X] Organize code in data_cleaning.py
- [X] Understand how data cleaning script batches problems of the same type.
- [X] Add splitter for proof questions.
- [X] Add JSON schema to model in data cleaning pipeline.
- [X] Run a cleaning run.
- [X] Give qwen a max batch size parameter to avoid memory problems.
- [X] Make cleaning pipeline run on two A100s to achieve a much larger batch size.
- [X] Check after the existence theorem has been created that it is not trivial.
- [X] Generate SFT dataset from finished lean proofs.
- [X] Make formalization pipeline run on two A100 80GB GPUs.
- [ ] Run formalization pipeline.
- [X] Figure out which licences the datasets / models have.

## Training
- [X] Train on SFT dataset.
- [X] Train on formalized theorems using AlphaProof.

## Interactive Environment
- [X] Visualize agent AND-OR search tree.

## Develop difficulty metric
- [X] Read STP paper again