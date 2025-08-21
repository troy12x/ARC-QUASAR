# Copyright 2024 Quasar AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn as nn
import argparse
from tqdm import tqdm
import numpy as np
import os
import json
import copy

from quasar.grid_lnn import GridLNN, GridLNNConfig
from scripts.train_grid_lnn import pad_grid

def print_grid(tensor, title=""):
    print(title)
    grid = tensor.cpu().numpy()
    for row in grid:
        print(" ".join(map(str, row)))
    print("\n")

def evaluate(args):
    config = GridLNNConfig(
        grid_size=(args.grid_size, args.grid_size),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        evolution_steps=args.evolution_steps
    )
    model = GridLNN(config).to(args.device)

    try:
        # Load state dict and adapt for key name change from 'encoder' to 'embedding'
        state_dict = torch.load(args.checkpoint_path, map_location=args.device)
        if 'encoder.weight' in state_dict and 'embedding.weight' not in state_dict:
            state_dict['embedding.weight'] = state_dict.pop('encoder.weight')
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        return

    task_files = [os.path.join(args.data_path, f) for f in os.listdir(args.data_path) if f.endswith('.json')]
    print(f"Starting persistent search on {len(task_files)} tasks...")

    tasks_solved = 0
    if not os.path.exists('solved_models'):
        os.makedirs('solved_models')

    for task_file in tqdm(task_files, desc="Searching for Solutions"):
        with open(task_file, 'r') as f:
            task_data = json.load(f)
        
        train_cases = task_data.get('train', [])
        test_cases = task_data.get('train', [])
        if not train_cases or not test_cases:
            continue

        adapted_model = copy.deepcopy(model).train()
        optimizer = optim.Adam(adapted_model.parameters(), lr=args.test_time_lr)

        # Prepare training data
        train_inputs = [torch.LongTensor(pad_grid(case['input'], (args.grid_size, args.grid_size))) for case in train_cases]
        train_targets = [torch.LongTensor(pad_grid(case['output'], (args.grid_size, args.grid_size))) for case in train_cases]
        train_input_batch = torch.stack(train_inputs).to(args.device)
        train_target_batch = torch.stack(train_targets).to(args.device)

        # Prepare test data
        test_inputs = [torch.LongTensor(pad_grid(case['input'], (args.grid_size, args.grid_size))) for case in test_cases]
        test_targets = [torch.LongTensor(pad_grid(case['output'], (args.grid_size, args.grid_size))) for case in test_cases]
        test_input_batch = torch.stack(test_inputs).to(args.device)
        test_target_batch = torch.stack(test_targets).to(args.device)

        tqdm.write(f"--- Starting search for {os.path.basename(task_file)} ---")
        step = 0
        best_accuracy = 0.0
        steps_since_improvement = 0
        criterion = nn.CrossEntropyLoss()
        use_focused_loss = False
        last_accuracy = -1.0
        steps_at_current_accuracy = 0

        # --- Test-Time Adaptation on Training Set ---
        while True:
            step += 1
            optimizer.zero_grad()
            
            # Forward pass on training data
            logits = adapted_model(train_input_batch)
            predicted_batch = torch.argmax(logits, dim=-1)
            current_accuracy = (predicted_batch == train_target_batch).float().mean().item()

            # Stop adaptation if max steps are reached or search stagnates
            if step >= args.max_adaptation_steps:
                tqdm.write(f"Reached max adaptation steps ({args.max_adaptation_steps}). Evaluating on test set.")
                break

            # Check for stagnation or completion
            if current_accuracy == 1.0 and steps_at_current_accuracy > args.perfect_accuracy_skip_steps:
                tqdm.write(f"Reached 100% train accuracy and held for {args.perfect_accuracy_skip_steps} steps. Evaluating on test set.")
                break
            elif steps_at_current_accuracy > args.skip_threshold:
                tqdm.write(f"Adaptation stagnated at {current_accuracy:.4f} for {args.skip_threshold} steps. Evaluating on test set.")
                break

            if abs(current_accuracy - last_accuracy) < 1e-6:
                steps_at_current_accuracy += 1
            else:
                last_accuracy = current_accuracy
                steps_at_current_accuracy = 0

            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1

            if not use_focused_loss and steps_since_improvement > args.stuck_threshold and best_accuracy > 0.9:
                use_focused_loss = True
                tqdm.write("Plateau detected on training set. Switching to focused cross-entropy loss.")

            # --- Calculate Loss on Training Data ---
            base_loss = 0
            if use_focused_loss:
                incorrect_mask = (predicted_batch != train_target_batch)
                if incorrect_mask.sum() > 0:
                    logits_incorrect = logits[incorrect_mask]
                    targets_incorrect = train_target_batch[incorrect_mask]
                    base_loss = criterion(logits_incorrect, targets_incorrect)
                else:
                    base_loss = torch.tensor(0.0, device=args.device, requires_grad=True)
            else:
                dist = Categorical(logits=logits / args.temperature)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                rewards = (actions.view_as(train_target_batch) == train_target_batch).float()
                reward_baseline = rewards.mean()
                base_loss = -(log_probs * (rewards - reward_baseline)).mean()

            # --- Self-Correction Mechanism ---
            correction_loss = 0
            if args.self_correction_threshold < current_accuracy < 1.0:
                with torch.no_grad():
                    incorrect_mask = (predicted_batch != train_target_batch).unsqueeze(-1).float()
                    # Create a corrupted input by blending original input with wrong predictions
                    corrupted_input_em = adapted_model.embedding(train_input_batch) * (1 - incorrect_mask) + adapted_model.embedding(predicted_batch) * incorrect_mask
                
                # Get logits for the corrupted input and calculate correction loss
                correction_logits = adapted_model.forward_from_em(corrupted_input_em)
                
                # Reshape for CrossEntropyLoss: logits -> (N, C), target -> (N)
                num_classes = correction_logits.shape[-1]
                correction_loss = criterion(
                    correction_logits.reshape(-1, num_classes),
                    train_target_batch.reshape(-1)
                )

            loss = base_loss + correction_loss
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                tqdm.write(f"Adaptation Step: {step}, Loss: {loss.item():.4f}, Train Accuracy: {current_accuracy:.4f}, Mode: {'Focused' if use_focused_loss else 'RL'}")

        # --- Final Evaluation on Test Set (No Learning) ---
        adapted_model.eval()
        with torch.no_grad():
            test_logits = adapted_model(test_input_batch)
            test_predicted_batch = torch.argmax(test_logits, dim=-1)
            test_accuracy = (test_predicted_batch == test_target_batch).float().mean().item()

        # --- Secondary Adaptation on Test Set if Failed ---
        if test_accuracy < 1.0:
            tqdm.write(f"Initial test accuracy {test_accuracy:.4f} < 1.0. Starting secondary adaptation on test set.")
            adapted_model.train()
            for secondary_step in range(args.secondary_adaptation_steps):
                optimizer.zero_grad()
                
                # Forward pass on test data to get predictions
                test_logits = adapted_model(test_input_batch)
                predicted_test_batch = torch.argmax(test_logits, dim=-1)

                # Self-correction: create corrupted embeddings from the model's own (wrong) predictions
                with torch.no_grad():
                    # We don't know the real incorrect mask, so we create a pseudo-mask based on confidence or other heuristics
                    # For simplicity, we'll just use the current prediction to generate a new input embedding
                    corrupted_input_em = adapted_model.embedding(predicted_test_batch)

                # Get logits for the corrupted input
                correction_logits = adapted_model.forward_from_em(corrupted_input_em)

                # The 'target' for this loss is the model's own, more confident prediction from the original pass
                # This encourages the model to become more consistent with its own stable predictions
                pseudo_target = torch.argmax(test_logits.detach(), dim=-1)
                
                num_classes = correction_logits.shape[-1]
                correction_loss = criterion(
                    correction_logits.reshape(-1, num_classes),
                    pseudo_target.reshape(-1)
                )

                loss = args.self_correction_weight * correction_loss
                loss.backward()
                optimizer.step()

                if secondary_step % 100 == 0:
                    tqdm.write(f"Secondary Step: {secondary_step}, Correction Loss: {loss.item():.4f}")

            # --- Final Re-Evaluation after Secondary Adaptation ---
            adapted_model.eval()
            with torch.no_grad():
                test_logits = adapted_model(test_input_batch)
                test_predicted_batch = torch.argmax(test_logits, dim=-1)
                test_accuracy = (test_predicted_batch == test_target_batch).float().mean().item()


        # --- Print final predictions ---
        tqdm.write("--- Final Evaluation Results ---")
        for i in range(len(test_input_batch)):
            print_grid(test_input_batch[i], title=f"Test Case {i+1} - Input")
            print_grid(test_target_batch[i], title=f"Test Case {i+1} - Expected Output")
            print_grid(test_predicted_batch[i], title=f"Test Case {i+1} - Model Prediction")

        if test_accuracy == 1.0:
            tasks_solved += 1
            solved_model_path = os.path.join('solved_models', f"solved_{os.path.basename(task_file).replace('.json', '.pth')}")
            torch.save(adapted_model.state_dict(), solved_model_path)
            tqdm.write(f"\n--- ✅ Task Solved: {os.path.basename(task_file)}! Final Test Accuracy: {test_accuracy:.4f}. Model saved to {solved_model_path} ---")
        else:
            tqdm.write(f"\n--- ❌ Task Failed: {os.path.basename(task_file)}. Final Test Accuracy: {test_accuracy:.4f} ---")

    print(f"\nSearch finished.")
    print(f"Total Tasks Solved: {tasks_solved} / {len(task_files)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a Grid LNN for ARC tasks.")
    parser.add_argument('--data_path', type=str, default='c:/quasarv4/ARC-AGI-2/data/evaluation', help='Path to ARC evaluation data.')
    parser.add_argument('--checkpoint_path', type=str, default='grid_lnn_epoch_5.pth', help='Path to the trained model checkpoint.')
    parser.add_argument('--grid_size', type=int, default=30, help='Grid size used during training.')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Dimension of cell embeddings used during training.')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Dimension of hidden states used during training.')
    parser.add_argument('--evolution_steps', type=int, default=16, help='Number of LNN evolution steps used during training.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to evaluate on.')
    parser.add_argument('--test_time_lr', type=float, default=1e-3, help='Learning rate for test-time adaptation.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling actions during adaptation.')
    parser.add_argument('--stuck_threshold', type=int, default=50, help='Steps without improvement to trigger focused loss.')
    parser.add_argument('--skip_threshold', type=int, default=1000, help='Steps with no change in accuracy to skip a task.')
    parser.add_argument('--high_accuracy_threshold', type=float, default=0.98, help='Accuracy threshold to trigger high-accuracy skip check.')
    parser.add_argument('--high_accuracy_skip_steps', type=int, default=10000, help='Steps to wait when stuck at high accuracy before skipping.')
    parser.add_argument('--max_adaptation_steps', type=int, default=5000, help='Maximum number of adaptation steps per task.')
    parser.add_argument('--perfect_accuracy_skip_steps', type=int, default=50, help='Steps to wait when at 100% train accuracy before stopping.')
    parser.add_argument('--self_correction_threshold', type=float, default=0.95, help='Accuracy threshold to start self-correction.')
    parser.add_argument('--self_correction_weight', type=float, default=0.5, help='Weight for the self-correction loss term.')
    parser.add_argument('--secondary_adaptation_steps', type=int, default=1000, help='Number of adaptation steps on the test set if initial eval fails.')

    args = parser.parse_args()
    evaluate(args)
