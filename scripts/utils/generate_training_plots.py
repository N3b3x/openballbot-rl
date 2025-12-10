#!/usr/bin/env python3
"""
Generate training plots for all archived models.
Extracts data from progress.csv files and creates reward/episode length plots.
"""

import csv
import os
import sys
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

def extract_training_data(csv_path):
    """Extract training data from progress.csv file."""
    timesteps = []
    eval_rewards = []
    eval_lengths = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        # Find column names
        timesteps_col = None
        eval_reward_col = None
        eval_length_col = None
        
        for col in reader.fieldnames:
            if 'total_timesteps' in col:
                timesteps_col = col
            if 'eval/mean_reward' in col:
                eval_reward_col = col
            if 'eval/mean_ep_length' in col:
                eval_length_col = col
        
        # Extract data
        for row in reader:
            if timesteps_col and row.get(timesteps_col):
                try:
                    ts = float(row[timesteps_col])
                    timesteps.append(ts)
                    
                    if eval_reward_col and row.get(eval_reward_col):
                        val = row[eval_reward_col].strip()
                        if val:
                            eval_rewards.append(float(val))
                        else:
                            eval_rewards.append(None)
                    else:
                        eval_rewards.append(None)
                    
                    if eval_length_col and row.get(eval_length_col):
                        val = row[eval_length_col].strip()
                        if val:
                            eval_lengths.append(float(val))
                        else:
                            eval_lengths.append(None)
                    else:
                        eval_lengths.append(None)
                except (ValueError, KeyError):
                    continue
    
    return timesteps, eval_rewards, eval_lengths

def create_training_plot(model_name, timesteps, eval_rewards, eval_lengths, output_path):
    """Create a training progress plot with reward and episode length."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    # Filter out None values
    valid_indices = [i for i, r in enumerate(eval_rewards) if r is not None]
    if not valid_indices:
        print(f"Warning: No valid evaluation data for {model_name}")
        plt.close(fig)
        return False
    
    ts_valid = [timesteps[i] for i in valid_indices]
    rewards_valid = [eval_rewards[i] for i in valid_indices]
    lengths_valid = [eval_lengths[i] for i in valid_indices if eval_lengths[i] is not None]
    ts_lengths_valid = [timesteps[i] for i in valid_indices if eval_lengths[i] is not None]
    
    # Convert to millions of timesteps
    ts_valid_m = [t / 1e6 for t in ts_valid]
    ts_lengths_m = [t / 1e6 for t in ts_lengths_valid]
    
    # Plot reward
    ax1.plot(ts_valid_m, rewards_valid, 'b-', linewidth=2, label='Mean Reward')
    ax1.set_xlabel('Environment Timesteps (millions)', fontsize=10)
    ax1.set_ylabel('Mean Reward', fontsize=10)
    ax1.set_title(f'{model_name.replace("_", " ").replace("-", " ")} - Training Progress', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot episode length
    if lengths_valid:
        ax2.plot(ts_lengths_m, lengths_valid, 'r-', linewidth=2, label='Mean Episode Length')
        ax2.set_xlabel('Environment Timesteps (millions)', fontsize=10)
        ax2.set_ylabel('Mean Episode Length', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return True

def main():
    base_dir = "/Users/nebex/Library/Mobile Documents/com~apple~CloudDocs/Documents/John Hokins/1_CLASSES/FALL_2025/Reinforcement Learning/Final Project/OpenBallBot-RL/outputs/experiments/archived_models"
    output_dir = "/Users/nebex/Library/Mobile Documents/com~apple~CloudDocs/Documents/John Hokins/1_CLASSES/FALL_2025/Reinforcement Learning/Final Project/Submissions/report-overleaf/figures/training_models"
    
    models_data = []
    
    for dirname in sorted(os.listdir(base_dir)):
        if os.path.isdir(dirname) and dirname not in ['progress_plots', 'legacy_salehi-2025-original']:
            csv_path = os.path.join(base_dir, dirname, 'progress.csv')
            if os.path.exists(csv_path):
                timesteps, eval_rewards, eval_lengths = extract_training_data(csv_path)
                
                if timesteps:
                    # Get model info
                    info_path = os.path.join(base_dir, dirname, 'info.txt')
                    info = {}
                    if os.path.exists(info_path):
                        try:
                            with open(info_path, 'r') as f:
                                info = json.loads(f.read())
                        except:
                            pass
                    
                    # Create plot
                    safe_name = dirname.replace('/', '_')
                    output_path = os.path.join(output_dir, f"{safe_name}_training.png")
                    
                    if create_training_plot(dirname, timesteps, eval_rewards, eval_lengths, output_path):
                        # Extract summary stats
                        valid_rewards = [r for r in eval_rewards if r is not None]
                        valid_lengths = [l for l in eval_lengths if l is not None]
                        
                        models_data.append({
                            'name': dirname,
                            'max_timesteps': max(timesteps),
                            'final_reward': valid_rewards[-1] if valid_rewards else None,
                            'final_length': valid_lengths[-1] if valid_lengths else None,
                            'max_reward': max(valid_rewards) if valid_rewards else None,
                            'max_length': max(valid_lengths) if valid_lengths else None,
                            'num_eval_points': len(valid_rewards),
                            'info': info
                        })
                        print(f"✓ Created plot for {dirname}")
    
    # Print summary
    print("\n=== Model Summary ===")
    for data in models_data:
        print(f"{data['name']}:")
        print(f"  Max timesteps: {data['max_timesteps']/1e6:.2f}M")
        print(f"  Final reward: {data['final_reward']:.2f}" if data['final_reward'] else "  Final reward: N/A")
        print(f"  Final length: {data['final_length']:.0f}" if data['final_length'] else "  Final length: N/A")
        print(f"  Max reward: {data['max_reward']:.2f}" if data['max_reward'] else "  Max reward: N/A")
        print()
    
    return models_data

if __name__ == '__main__':
    main()

