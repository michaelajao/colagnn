import os
import itertools
import subprocess
import csv

# Configuration for all experiments
DATASET_CONFIGS = [
    # Dataset name, adjacency matrix name, horizons
    # Japan, US-region, and US-states with horizons 3, 5, 10, 15
    ('japan', 'japan-adj', [3, 5, 10, 15]),
    ('region785', 'region-adj', [3, 5, 10, 15]),  # US-region
    ('state360', 'state-adj-49', [3, 5, 10, 15]),  # US-states
    
    # Other datasets with horizons 3, 7, 14
    ('australia-covid', 'australia-adj', [3, 7, 14]),
    ('spain-covid', 'spain-adj', [3, 7, 14]),
    ('nhs_timeseries', 'nhs-adj', [3, 7, 14]),
    ('ltla_timeseries', 'ltla-adj', [3, 7, 14])
]

MODELS = ['cola_gnn', 'SelfAttnRNN', 'lstnet', 'dcrnn', 'CNNRNN_Res']

# Common parameters
COMMON_PARAMS = {
    'window': 20,
    'epochs': 1500,
    'batch': 128,  # Updated to 128
    'lr': 1e-3,
    'seed': 42,
    'train': 0.5,  # Added train split
    'val': 0.2,    # Added validation split
    'test': 0.3,   # Added test split
    'patience': 100,  # Added patience
    'gpu': 0,      # Added GPU selection
    'mylog': True  # Added mylog flag
}

def run_experiment(dataset, sim_mat, horizon, model):
    """Run a single experiment with the given parameters"""
    # Get the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    save_dir = os.path.join(project_root, 'save', model, dataset, f'h{horizon}')
    
    cmd = [
        'python', 'train.py',
        '--dataset', dataset,
        '--sim_mat', sim_mat,
        '--horizon', str(horizon),
        '--model', model,
        '--window', str(COMMON_PARAMS['window']),
        '--epochs', str(COMMON_PARAMS['epochs']),
        '--patience', str(COMMON_PARAMS['patience']),
        '--batch', str(COMMON_PARAMS['batch']),
        '--lr', str(COMMON_PARAMS['lr']),
        '--seed', str(COMMON_PARAMS['seed']),
        '--train', str(COMMON_PARAMS['train']),
        '--val', str(COMMON_PARAMS['val']),
        '--test', str(COMMON_PARAMS['test']),
        '--gpu', str(COMMON_PARAMS['gpu']),
        '--save_dir', save_dir
    ]
    
    # Add the mylog flag if true
    if COMMON_PARAMS['mylog']:
        cmd.append('--mylog')
    
    print(f"\n{'='*80}")
    print(f"Running experiment: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    # Create the save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Run the command
    subprocess.run(cmd)

def main():
    """Run all experiments"""
    # Get the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Create result directory for metrics
    result_dir = os.path.join(project_root, 'result')
    os.makedirs(result_dir, exist_ok=True)
    
    # Create metrics.csv with header if it doesn't exist
    csv_filename = os.path.join(result_dir, "metrics.csv")
    header = ['dataset', 'horizon', 'model', 'mae', 'std_mae', 'rmse', 'rmse_states', 'pcc', 'pcc_states', 'r2', 'r2_states', 'var', 'var_states', 'peak_mae']
    
    try:
        with open(csv_filename, 'x', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
    except FileExistsError:
        pass
    
    # Total number of experiments
    total_experiments = sum(len(horizons) * len(MODELS) for _, _, horizons in DATASET_CONFIGS)
    completed = 0
    
    print(f"Starting {total_experiments} experiments...\n")
    
    # Loop through all combinations
    for dataset_config in DATASET_CONFIGS:
        dataset, sim_mat, horizons = dataset_config
        for horizon, model in itertools.product(horizons, MODELS):
            completed += 1
            print(f"\nExperiment {completed}/{total_experiments}")
            print(f"Dataset: {dataset}, Horizon: {horizon}, Model: {model}")
            
            run_experiment(dataset, sim_mat, horizon, model)

if __name__ == "__main__":
    main()