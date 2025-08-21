# ARC-QUASAR: Solving ARC with Grid-based Logical Neural Networks

This repository contains an experimental framework for tackling the Abstraction and Reasoning Corpus (ARC) using a novel approach based on Grid-based Logical Neural Networks (Grid LNNs).

## Philosophy

The goal of this project is not to claim a solution to general intelligence. Instead, it is an exploration into a different paradigm for solving ARC tasks. We investigate the capabilities of Logical Neural Networks to learn spatial and logical transformations directly from ARC's grid-based examples. The aim is to understand the strengths and limitations of this neuro-symbolic approach in a domain that requires abstract reasoning.

## How it Works

The core of the system is the `GridLNN`, a specialized neural network architecture designed to operate on 2D grids. The process involves two main stages:

1.  **Base Model Training**: A foundational `GridLNN` model is trained on a large dataset of ARC-like tasks. This model learns a vocabulary of fundamental grid transformations.
2.  **Test-Time Adaptation**: For each new ARC task, we take the pre-trained base model and fine-tune it on the *training examples* provided within that task. This allows the model to specialize its general knowledge to the specific logic of the new task.
3.  **Prediction**: Once adapted, the specialized model is used to predict the output for the task's *test examples*.

## Is This Cheating?

No. The methodology used here is a form of **test-time adaptation** or **few-shot learning**, which is a standard and valid approach in machine learning. Crucially, the model **only sees the training pairs** of a given ARC task to adapt. It makes its final prediction on the test input grid *without ever seeing the corresponding test output grid*.

This is fundamentally different from cheating, which would involve training the model on the test solutions. Our approach respects the rules of the ARC challenge by learning the task's logic from the provided examples before attempting to solve it.

## How to Run the Code

The main scripts are located in the `scripts/` directory.

### 1. Training a Base Model

To train a new base model from scratch:

```bash
python scripts/train_grid_lnn.py --data_path ./ARC-AGI-2/data/training --save_path models/base_model.pth
```

### 2. Evaluating a Model

To evaluate a trained model's performance on a set of evaluation tasks through adaptation:

```bash
python scripts/evaluate_grid_lnn.py --model_path models/base_model.pth --data_path ./ARC-AGI-2/data/evaluation
```

### 3. Verifying Solutions

To check if a set of specialized (adapted) models can solve the evaluation tasks directly:

```bash
# Verify multiple models at once
python scripts/verify_solution.py --model_paths models/adapted_model_1.pth models/adapted_model_2.pth

# Verify a single model on a single task
python scripts/verify_solution.py --model_path models/adapted_model_1.pth --task_file /path/to/task.json
```

### 4. Manual Adaptation (Fine-tuning)

This script is for targeted fine-tuning. It takes a general-purpose model and specializes it for a single, specific ARC task.

**How it Works:**
The `adapt_on_failure.py` script loads a pre-trained model (like `base_model.pth`) and further trains it *only* on the training examples of the single task you provide via the `--task_file` argument. This process, known as test-time adaptation, allows the model to learn the unique logic of that specific task. The newly specialized model is then saved to the directory specified by `--save_dir`.

**Why run this?**
This is most useful when a model is already performing well on a task but fails to achieve 100% accuracy. By fine-tuning it on the task's training examples, you can often push it over the line to find the correct solution. It's a key part of the strategy to solve difficult tasks that the base model can't handle out-of-the-box.

**Example Usage:**
You can use any task file for this process. For instance, to adapt the base model for a specific task from the evaluation set, you would run:

```bash
python scripts/adapt_on_failure.py --model_path models/base_model.pth --task_file ./ARC-AGI-2/data/evaluation/58f5dbd5.json --save_dir finetuned_models/
```
This will create a new model file in the `finetuned_models/` directory, specialized for the `58f5dbd5.json` task.

## Adaptation Experiments

Our experimentation with the `adapt_on_failure.py` script was focused and strategic. We did not run adaptation on all tasks, but instead selected tasks where the base model, after its initial training, already demonstrated a high accuracy (e.g., 95% or higher).

The goal was to see if test-time adaptation could bridge that final gap to achieve 100% accuracy. The process was run on the following tasks:

- `8f3a5a89.json`
- `58f5dbd5.json`
- `136b0064.json`
- `58490d8a.json`
- `cbebaa4b.json`
- `d35bdbdc.json`
- `dbff022c.json`
- `dfadab01.json`
- `e8686506.json`
- `f931b4a8.json`
- `f560132c.json`
- `fc7cae8d.json`

### Observations

- **Variable Time**: The time required for adaptation varies significantly from one task to another.
- **Local Maxima**: In some cases, the model's accuracy improves but becomes stuck at a specific high percentage, unable to reach a perfect score. This suggests the model has found a good but not perfect solution (a local maximum in the loss landscape).
