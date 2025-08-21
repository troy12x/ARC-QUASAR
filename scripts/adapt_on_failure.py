import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import os
import json
import copy
from tqdm import tqdm

from quasar.grid_lnn import GridLNN, GridLNNConfig
from scripts.train_grid_lnn import pad_grid

def adapt_with_confidence_penalty(args):
    # --- 1. Setup ---
    config = GridLNNConfig(
        grid_size=(args.grid_size, args.grid_size),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        evolution_steps=args.evolution_steps
    )
    device = torch.device(args.device)
    criterion = nn.CrossEntropyLoss(reduction='none') # Use 'none' for per-pixel loss

    # --- 2. Load Data ---
    with open(args.task_file, 'r') as f:
        task_data = json.load(f)
    test_cases = task_data['test']
    if not test_cases:
        print("No test cases found in task file.")
        return
    print(f"Found {len(test_cases)} test cases in {os.path.basename(args.task_file)}.")

    input_grid = torch.stack([torch.LongTensor(pad_grid(case['input'], (args.grid_size, args.grid_size))) for case in test_cases]).to(device)
    target_grid = torch.stack([torch.LongTensor(pad_grid(case['output'], (args.grid_size, args.grid_size))) for case in test_cases]).to(device)

    # --- 3. Load Model ---
    model = GridLNN(config).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.adaptation_lr)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=args.patience, verbose=True)

    # --- 4. Initial Evaluation ---
    model.eval()
    with torch.no_grad():
        initial_logits = model(input_grid)
        initial_prediction = torch.argmax(initial_logits, dim=-1)
        initial_accuracy = (initial_prediction == target_grid).float().mean().item()
        correct_mask = initial_prediction == target_grid
        incorrect_mask = ~correct_mask

    print(f"Focusing on model: {os.path.basename(args.model_path)}")
    print(f"Initial Accuracy: {initial_accuracy:.2%}")
    if initial_accuracy == 1.0:
        print("Model already perfect.")
        return
    print(f"Targeting {incorrect_mask.sum()} incorrect pixels for exploration.")

    # --- 5. Confidence-Penalized Exploration Loop ---
    best_accuracy = initial_accuracy
    model.train()
    print("Starting Confidence-Penalized Exploration (runs until 100% accuracy is reached)...")
    step = 0
    while True:
        optimizer.zero_grad()

        logits = model(input_grid)
        dist = Categorical(logits=logits)

        # --- Loss Part 1: Distillation (for CORRECT pixels) ---
        # We want to keep the correct predictions stable.
        # Loss is calculated against the model's own initial (correct) predictions.
        distillation_loss = criterion(logits.permute(0, 3, 1, 2), initial_prediction)
        # Apply loss ONLY to the pixels that were already correct.
        distillation_loss = (distillation_loss * correct_mask).mean()

        # --- Loss Part 2: Targeted Correction (for INCORRECT pixels) ---
        # For the pixels the model got wrong, we provide the ground truth as a direct learning signal.
        # This is applied ONLY to the incorrect pixels, avoiding "cheating" on the correct ones.
        correction_loss = criterion(logits.permute(0, 3, 1, 2), target_grid)
        correction_loss = (correction_loss * incorrect_mask).mean()

        # --- Combine Losses ---
        loss = distillation_loss + (correction_loss * args.correction_weight)

        loss.backward()
        optimizer.step()
        step += 1

        if (step + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                current_logits = model(input_grid)
                current_prediction = torch.argmax(current_logits, dim=-1)
                current_accuracy = (current_prediction == target_grid).float().mean().item()
            
            tqdm.write(f"  Step {step+1}, Loss: {loss.item():.6f}, Accuracy: {current_accuracy:.2%} (Best: {best_accuracy:.2%})")

            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                if args.save_path:
                    tqdm.write(f"  New best accuracy! Saving model to {args.save_path}")
                    torch.save(model.state_dict(), args.save_path)

            # Adjust learning rate if accuracy plateaus
            scheduler.step(current_accuracy)

            model.train()
            if current_accuracy == 1.0:
                print("Reached 100% accuracy!")
                break

    # --- 6. Final Evaluation ---
    model.eval()
    with torch.no_grad():
        final_logits = model(input_grid)
        final_prediction = torch.argmax(final_logits, dim=-1)
        final_accuracy = (final_prediction == target_grid).float().mean().item()
    print(f"Adaptation finished. Final Accuracy: {final_accuracy:.2%}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Adapt a single high-accuracy model using Confidence-Penalized Exploration.")
    parser.add_argument('--task_file', type=str, default=None, help='Path to a single .json task file (overrides data_path).')
    parser.add_argument('--task_files', type=str, nargs='+', default=None, help='List of task file names (without .json extension) to process.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the specific .pth model file to adapt.')
    parser.add_argument('--grid_size', type=int, default=30)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--evolution_steps', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--adaptation_lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10, help='Patience for LR scheduler (in epochs of 50 steps).')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the best performing model.')
    parser.add_argument('--correction_weight', type=float, default=1.0, help='Weight to amplify the loss on incorrect pixels.')

    args = parser.parse_args()
    adapt_with_confidence_penalty(args)

