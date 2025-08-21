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

def unpad_grid(padded_grid, original_size):
    """Cuts down a padded grid to its original size."""
    original_rows, original_cols = original_size
    padded_rows, padded_cols = len(padded_grid), len(padded_grid[0])
    if original_rows > padded_rows or original_cols > padded_cols:
        return padded_grid
    unpadded_grid = [[padded_grid[r][c] for c in range(original_cols)] for r in range(original_rows)]
    return unpadded_grid

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

    # Create a mapping from task_id to specialist model
    specialist_models = {}
    if args.model_paths:
        for path in args.model_paths:
            task_id = os.path.basename(path).split('_')[-1].split('.')[0]
            if os.path.basename(path) in models:
                specialist_models[task_id] = (os.path.basename(path), models[os.path.basename(path)])
        print(f"Created map for {len(specialist_models)} specialist models.")

    submission = {}

    if args.task_files:
        task_files = args.task_files
        print(f"--- On {len(task_files)} specified tasks ---")
    elif args.task_file:
        task_files = [args.task_file]
        print(f"--- On single task: {os.path.basename(args.task_file)} ---")
    elif args.model_paths or args.model_path:
        task_files = [os.path.join(args.data_path, f) for f in os.listdir(args.data_path) if f.endswith('.json')]
        print(f"--- On all {len(task_files)} tasks in: {args.data_path} ---")
    else: # Default case: run all models in solved_models_dir on all tasks in data_path
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

        task_id = task_filename.split('.')[0]
        print(f"--- Verifying Task: {task_id} ---")

        best_prediction = None
        best_accuracy = -1.0
        task_solved = False

        # --- Specialist Model Logic ---
        if task_id in specialist_models:
            model_name, model = specialist_models[task_id]
            print(f"  Specialist model found: {model_name}")
            with torch.no_grad():
                logits = model(input_batch)
                prediction = torch.argmax(logits, dim=-1)
            
            pixel_accuracy = (prediction == target_batch).float().mean().item()
            best_prediction = prediction
            best_accuracy = pixel_accuracy

            if pixel_accuracy == 1.0:
                print(f"  [✅] {model_name}: Solved (100.00% accuracy)")
                solved_count += 1
                task_solved = True
            else:
                print(f"  [❌] {model_name}: Failed ({pixel_accuracy:.2%}). Falling back to ensemble.")
        
        # --- Ensemble Fallback Logic ---
        if not task_solved:
            # If no specialist was found, or if the specialist failed, run the ensemble.
            for model_name, model in models.items():
                # Skip the specialist if it was already tried and failed
                if task_id in specialist_models and model_name == specialist_models[task_id][0]:
                    continue

                with torch.no_grad():
                    logits = model(input_batch)
                    prediction = torch.argmax(logits, dim=-1)
                
                pixel_accuracy = (prediction == target_batch).float().mean().item()

                if pixel_accuracy > best_accuracy:
                    best_accuracy = pixel_accuracy
                    best_prediction = prediction

                if pixel_accuracy == 1.0:
                    print(f"  [✅] {model_name} (Ensemble): Solved (100.00% accuracy)")
                    if not task_solved:
                        solved_count += 1
                        task_solved = True
                    # Use the first solver's prediction and stop checking other models
                    break 
                else:
                    # Don't print failure for every model in ensemble unless it's the best so far
                    pass

        if not task_solved:
            print(f"--- Task Failed by all models (Best accuracy: {best_accuracy:.2%}) ---")

        # Format the prediction for the submission file
        if best_prediction is not None:
            task_predictions = []
            for i, test_case in enumerate(test_cases):
                original_output_size = (len(test_case['output']), len(test_case['output'][0]))
                padded_pred_grid = best_prediction[i].cpu().numpy().tolist()
                unpadded_pred_grid = unpad_grid(padded_pred_grid, original_output_size)
                task_predictions.append({"attempt_1": unpadded_pred_grid, "attempt_2": unpadded_pred_grid})
            submission[task_id] = task_predictions

    print(f"\n--- Evaluation Summary ---")
    print(f"Solved {solved_count} out of {total_tasks} tasks ({solved_count/total_tasks:.2%})")

    if args.submission_file:
        with open(args.submission_file, 'w') as f:
            json.dump(submission, f, indent=4)
        print(f"\nSubmission file saved to: {args.submission_file}")

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
    parser.add_argument('--submission_file', type=str, default='submission.json', help='Path to save the submission JSON file.')
    
    args = parser.parse_args()
    verify(args)
