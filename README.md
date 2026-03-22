# LetterCountingReasoner

> Fine-tunes `Qwen2.5-3B-Instruct` with LoRA and GRPO to improve step-by-step letter counting while preserving general knowledge behavior.

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Transformers%20%7C%20TRL-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)

## Overview

Large language models are often fluent but unreliable on small procedural reasoning tasks such as counting how many times a letter appears in a word. This project focuses on a narrow but revealing failure mode: teaching an instruction-tuned LLM to spell a word letter by letter, maintain a running total, and return the final count in a structured format.

The repository contains a completed notebook that builds a synthetic letter-counting dataset, defines reward functions for intermediate reasoning quality, and fine-tunes `Qwen2.5-3B-Instruct` with LoRA and GRPO. The saved notebook outputs show a qualitative improvement on the target task and a simple retention check showing that the tuned adapter still answers a general knowledge question correctly.

## Project Highlights

- Fine-tunes `Qwen2.5-3B-Instruct` in 4-bit mode with Unsloth and LoRA
- Uses a synthetic 401-example dataset generated from 62 curated words
- Shapes model behavior with five reward functions: numbering, spelling, counting, XML format, and final correctness
- Runs both a short 5-step sanity check and a longer 250-step GRPO training pass
- Compares the pretrained model and the LoRA-adapted model on the same letter-counting prompt
- Includes a catastrophic-forgetting check with a general knowledge question

## Problem Statement

Counting letters in a word is easy for people and surprisingly brittle for LLMs. The model must process the word sequentially, update a running count only when the target letter appears, and keep the reasoning trace consistent with the final answer. This project uses reinforcement learning with structured rewards to push the model toward that behavior instead of relying on prompting alone.

## Methodology

1. Start with a prompt-only baseline using a chain-of-thought style system prompt and XML-style `<reasoning>` / `<answer>` output.
2. Generate a synthetic dataset of `(word, target letter, count)` records, including both positive matches and letters that do not appear in the word.
3. Format the dataset as chat prompts for `Qwen2.5-3B-Instruct`.
4. Define reward functions that score intermediate reasoning quality and final answer correctness.
5. Fine-tune the model with GRPO using a LoRA rank of `64` across the main attention and MLP projection layers.
6. Compare the untuned and tuned models on the task, then run a simple general-knowledge retention check.

## Results And Observations

The notebook stores qualitative evidence rather than a formal benchmark table. The strongest supported claims are the sample comparison output, the executed training runs, and the preserved answer on a non-letter-counting question.

| Check | Observed notebook outcome |
| --- | --- |
| Prompt-only baseline | The untuned model can produce an inconsistent running count on the target task |
| Fine-tuned comparison | On the `idea` / `a` example, the tuned model answers `1` while the old model answers `3` |
| Training runs | A 5-step validation run and a 250-step training run were executed and plotted in the notebook |
| Retention check | Both old and new models answer `What is the capital of France?` with `Paris` |

Example comparison captured in the notebook:

| Prompt | Old model | Tuned model |
| --- | --- | --- |
| Count `a` in `idea` | Incorrect final answer: `3` | Correct final answer: `1` |

## Tech Stack

**Core libraries**

- Python
- Jupyter Notebook
- PyTorch
- Hugging Face `datasets`
- Hugging Face `transformers`
- `trl`
- Unsloth
- vLLM
- pandas
- matplotlib

**Modeling setup**

- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Adaptation method: LoRA
- RL algorithm: GRPO
- Quantization: 4-bit loading in the notebook workflow

## Repository Structure

```text
LetterCountingReasoner/
├── README.md
└── gen_ai_fundamentals_project_starter.ipynb
```

## File Guide

| File | Purpose |
| --- | --- |
| `gen_ai_fundamentals_project_starter.ipynb` | End-to-end project notebook covering prompt engineering, synthetic dataset generation, reward design, GRPO training, plotting, and model comparison |
| `README.md` | Public project summary and usage notes |

## Dataset

- Dataset type: synthetic dataset generated inside the notebook
- Source vocabulary: 62 curated words of varying lengths
- Final dataset size: 401 prompt-answer records
- Labels: exact count of the target letter in each word
- Negative cases: includes letters that do not appear in the word, with count `0`

The dataset is created programmatically with `Dataset.from_generator`, so no separate dataset download is required for the published repo.

## How To Run

### Option 1: Vocareum GPU environment

This is the validated workflow reflected by the stored notebook outputs.

1. Start the cloud lab in Vocareum and open the VS Code cloud console.
2. Open [`gen_ai_fundamentals_project_starter.ipynb`](./gen_ai_fundamentals_project_starter.ipynb).
3. Run the notebook cells in order.
4. Review the quick training run, full training run, comparison outputs, and reward plots directly in the notebook.

```bash
git clone https://github.com/gour6380/LetterCountingReasoner.git
cd LetterCountingReasoner
```

### Option 2: Local macOS workflow

The committed notebook was validated in a pre-provisioned GPU lab environment. It uses `unsloth` and `vllm`, which are typically CUDA-oriented dependencies, so the exact training path in this notebook is not guaranteed to run unchanged on Apple Silicon.

Practical local usage:

- Open the notebook locally to inspect the full implementation and saved outputs.
- Use a remote GPU environment if you want to reproduce the full GRPO fine-tuning path exactly as written.
- If you want to adapt the project for Apple Silicon, expect to swap or reconfigure the CUDA-specific pieces rather than running the notebook unchanged.

## Expected Outputs

After running the notebook in a compatible environment, you should expect:

- Printed examples of baseline and post-training model behavior
- Reward validation assertions for the custom reward functions
- Logged GRPO training output for both the quick and longer runs
- Reward plots for `reward` and `rewards/correct_answer_reward_func/mean`
- A saved LoRA adapter directory created by `model.save_lora("grpo_saved_lora")`
- A side-by-side old-vs-new comparison on a dataset prompt
- A simple knowledge-retention comparison on a general question

## Limitations

- The repository does not include a standalone training script; the workflow is notebook-centric.
- Results are qualitative and sample-based in the stored outputs, not a formal held-out benchmark.
- The validated execution environment is a cloud GPU lab rather than local Apple Silicon.
- The saved adapter artifacts are generated during notebook execution and are not committed in this repository.
