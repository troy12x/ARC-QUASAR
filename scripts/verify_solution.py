import torch
import argparse
import os
import json

from quasar.grid_lnn import GridLNN, GridLNNConfig
from scripts.train_grid_lnn import pad_grid

def print_grid(tensor, title=""):
    print(title)
    grid = tensor.cpu().numpy()
    for row in grid:
        print(" ".join(map(str, row)))
    print("\n")

def verify(args):
    config = GridLNNConfig(
        grid_size=(args.grid_size, args.grid_size),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        evolution_steps=args.evolution_steps
    )

    # Load specialist models
    if args.model_paths:
        model_paths = args.model_paths
        print(f"--- Verifying {len(model_paths)} specified models ---")
    elif args.model_path:
        model_paths = [args.model_path]
        print(f"--- Verifying single model: {os.path.basename(args.model_path)} ---")
    else:
        model_paths = [os.path.join(args.solved_models_dir, f) for f in os.listdir(args.solved_models_dir) if f.endswith('.pth')]
        print(f"--- Verifying all models in: {args.solved_models_dir} ---")
    models = {}
    for path in model_paths:
        model = GridLNN(config).to(args.device)
        try:
            state_dict = torch.load(path, map_location=args.device)
            model.load_state_dict(state_dict)
            model.eval()
            models[os.path.basename(path)] = model
        except Exception as e:
            print(f"Warning: Could not load model {path}. Error: {e}")

    if not models:
        print(f"Error: No valid models found in {args.solved_models_dir}")
        return

    print(f"Loaded {len(models)} specialist models.")

    if args.task_files:
        task_files = args.task_files
        print(f"--- On {len(task_files)} specified tasks ---")
    elif args.task_file:
        task_files = [args.task_file]
        print(f"--- On single task: {os.path.basename(args.task_file)} ---")
    else:
        task_files = [os.path.join(args.data_path, f) for f in os.listdir(args.data_path) if f.endswith('.json')]
        print(f"--- On all {len(task_files)} tasks in: {args.data_path} ---")

    solved_count = 0
    total_tasks = len(task_files)

    for task_file in task_files:
        task_filename = os.path.basename(task_file)
        with open(task_file, 'r') as f:
            task_data = json.load(f)

        test_cases = task_data.get('test', [])
        if not test_cases:
            continue

        inputs = [torch.LongTensor(pad_grid(case['input'], (args.grid_size, args.grid_size))) for case in test_cases]
        targets = [torch.LongTensor(pad_grid(case['output'], (args.grid_size, args.grid_size))) for case in test_cases]
        input_batch = torch.stack(inputs).to(args.device)
        target_batch = torch.stack(targets).to(args.device)

        task_solved_by_any = False
        print(f"--- Verifying Task: {task_filename} ---")
        for model_name, model in models.items():
            with torch.no_grad():
                logits = model(input_batch)
                prediction = torch.argmax(logits, dim=-1)
            
            # Calculate pixel-wise accuracy
            pixel_accuracy = (prediction == target_batch).float().mean().item()

            if pixel_accuracy == 1.0:
                print(f"  [✅] {model_name}: Solved (100.00% accuracy)")
                if not task_solved_by_any:
                    solved_count += 1
                    task_solved_by_any = True
            else:
                print(f"  [❌] {model_name}: Failed ({pixel_accuracy:.2%})")
        
        if not task_solved_by_any:
            print(f"--- Task Failed by all models ---")

    print(f"\n--- Evaluation Summary ---")
    print(f"Model Ensemble: {args.solved_models_dir}")
    print(f"Solved {solved_count} out of {total_tasks} tasks ({solved_count/total_tasks:.2%})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Verify a solved Grid LNN model.")
    parser.add_argument('--model_paths', type=str, nargs='+', default=None, help='List of paths to specific model checkpoints.')
    parser.add_argument('--model_path', type=str, default=None, help='Path to a single model checkpoint (overridden by model_paths).')
    parser.add_argument('--task_files', type=str, nargs='+', default=None, help='List of paths to specific task files.')
    parser.add_argument('--task_file', type=str, default=None, help='Path to a single task file.')
    parser.add_argument('--solved_models_dir', type=str, default='solved_models', help='Directory containing solved model checkpoints (used if model_path is not set).')
    parser.add_argument('--data_path', type=str, default='c:/quasarv4/ARC-AGI-2/data/evaluation', help='Path to ARC evaluation data (used if task_file is not set).')
    parser.add_argument('--grid_size', type=int, default=30, help='Grid size used during training.')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Dimension of cell embeddings used during training.')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Dimension of hidden states used during training.')
    parser.add_argument('--evolution_steps', type=int, default=16, help='Number of LNN evolution steps used during training.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to evaluate on.')
    
    args = parser.parse_args()
    verify(args)
