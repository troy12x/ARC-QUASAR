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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical

import json
import os
import numpy as np
from tqdm import tqdm
import argparse

# Assuming grid_lnn is in the parent directory or accessible in the python path
from quasar.grid_lnn import GridLNN, GridLNNConfig

# --- 1. Data Loading ---

def pad_grid(grid, size=(30, 30), pad_value=0):
    """Pads a 2D list to a target size."""
    padded_grid = np.full(size, pad_value, dtype=int)
    h, w = len(grid), len(grid[0])
    padded_grid[:h, :w] = grid
    return padded_grid

class ARCDataset(Dataset):
    """Dataset for ARC tasks."""
    def __init__(self, data_path, max_grid_size=(30, 30)):
        self.data_path = data_path
        self.task_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.json')]
        self.max_grid_size = max_grid_size
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for task_file in self.task_files:
            with open(task_file, 'r') as f:
                task_data = json.load(f)
                for sample in task_data['train']:
                    samples.append(sample)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_grid = torch.LongTensor(pad_grid(sample['input'], self.max_grid_size))
        output_grid = torch.LongTensor(pad_grid(sample['output'], self.max_grid_size))
        return input_grid, output_grid

# --- 2. Training Loop ---

def train(args):
    """Main training function."""
    # Config and Model
    config = GridLNNConfig(
        grid_size=(args.grid_size, args.grid_size),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        evolution_steps=args.evolution_steps,
        dt=args.dt
    )
    model = GridLNN(config).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.load_checkpoint:
        if os.path.exists(args.load_checkpoint):
            model.load_state_dict(torch.load(args.load_checkpoint, map_location=args.device))
            try:
                # e.g., grid_lnn_epoch_10.pth -> 10
                start_epoch = int(os.path.basename(args.load_checkpoint).split('_')[-1].split('.')[0])
                print(f"Loaded checkpoint '{args.load_checkpoint}'. Resuming from epoch {start_epoch}.")
            except (ValueError, IndexError):
                print(f"Loaded checkpoint '{args.load_checkpoint}', but could not parse epoch. Starting from 0.")
        else:
            print(f"Checkpoint file not found at '{args.load_checkpoint}'. Starting from scratch.")

    # Data
    dataset = ARCDataset(args.data_path, max_grid_size=(args.grid_size, args.grid_size))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print(f"Starting training on {len(dataset)} samples...")

    for epoch in range(start_epoch, args.epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        total_reward = 0

        for input_grid, target_grid in progress_bar:
            input_grid, target_grid = input_grid.to(args.device), target_grid.to(args.device)

            optimizer.zero_grad()

            # --- Greedy RL Step ---
            # 1. Get model's prediction (logits)
            logits = model(input_grid) # (B, H, W, C)
            
            # 2. Sample an action (a new grid) from the policy (logits)
            # Reshape for sampling: (B*H*W, C)
            dist = Categorical(logits=logits.view(-1, config.num_colors))
            action = dist.sample()
            
            # 3. Calculate Cell-Wise Rewards
            predicted_grid = action.view_as(input_grid)
            # rewards_per_cell is a tensor of 1s (correct) and 0s (incorrect)
            rewards_per_cell = (predicted_grid == target_grid).float()
            total_reward += rewards_per_cell.mean().item()

            # 4. Calculate Loss with a Baseline for Stability
            # Reshape for calculation
            log_probs = dist.log_prob(action) # (B*H*W)
            rewards = rewards_per_cell.view(-1) # (B*H*W)

            # Normalize rewards (subtracting the mean as a simple baseline)
            reward_baseline = rewards.mean()
            adjusted_rewards = rewards - reward_baseline

            # Policy gradient loss
            loss = -(log_probs * adjusted_rewards).mean()


            # 5. Backpropagate
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'reward': f'{rewards_per_cell.mean().item():.4f}'})

        avg_reward = total_reward / len(dataloader)
        print(f"Epoch {epoch+1} finished. Average Reward: {avg_reward:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), f'grid_lnn_epoch_{epoch+1}.pth')

    print("\nTraining finished.")
    final_checkpoint_path = f'grid_lnn_epoch_{args.epochs}.pth'
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"Final model saved to {final_checkpoint_path}")

# --- 3. Main Execution ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Grid LNN for ARC tasks.")
    parser.add_argument('--data_path', type=str, default='c:/quasarv4/ARC-AGI-2/data/training', help='Path to ARC training data.')
    parser.add_argument('--grid_size', type=int, default=30, help='Grid size to pad to.')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Dimension of cell embeddings.')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Dimension of hidden states.')
    parser.add_argument('--evolution_steps', type=int, default=16, help='Number of LNN evolution steps.')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step for LNN.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--save_interval', type=int, default=10, help='Epoch interval to save checkpoints.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to train on.')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to a checkpoint to load and continue training.')
    
    args = parser.parse_args()
    train(args)
