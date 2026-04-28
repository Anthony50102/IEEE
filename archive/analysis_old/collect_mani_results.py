"""
Collect and Summarize Manifold Experiment Results
==================================================
After all parallel jobs complete, run this to collect results and generate
a summary comparison table.

Usage:
    python collect_mani_results.py --output-base /path/to/output --run-name mani_reg_sweep

Author: Anthony Poole
"""

import argparse
import os
import glob
import numpy as np
import yaml
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Collect Manifold Experiment Results")
    parser.add_argument("--output-base", type=str, required=True, help="Base output directory")
    parser.add_argument("--run-name", type=str, required=True, help="Run name prefix")
    args = parser.parse_args()
    
    # Find all result directories
    pattern = os.path.join(args.output_base, f"{args.run_name}_*")
    dirs = sorted(glob.glob(pattern))
    
    if not dirs:
        print(f"No directories found matching: {pattern}")
        return
    
    print("=" * 80)
    print("MANIFOLD EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    print(f"Output base: {args.output_base}")
    print(f"Run name: {args.run_name}")
    print(f"Found {len(dirs)} result directories")
    print("")
    
    # Collect results
    results = []
    pod_result = None
    
    for d in dirs:
        recon_file = os.path.join(d, "reconstruction_results.npz")
        preproc_file = os.path.join(d, "preprocessing_info.npz")
        
        if not os.path.exists(recon_file):
            print(f"  Skipping {d} - no reconstruction_results.npz")
            continue
        
        recon = np.load(recon_file)
        preproc = np.load(preproc_file)
        
        method = str(preproc.get('reduction_method', 'unknown'))
        r = int(preproc.get('r_actual', 0))
        train_err = float(recon['train_error_rel'])
        test_err = float(recon['test_error_rel'])
        
        dir_name = os.path.basename(d)
        
        if "POD" in dir_name or method == "linear":
            pod_result = {
                'name': dir_name,
                'method': 'POD',
                'r': r,
                'train_err': train_err,
                'test_err': test_err,
            }
        else:
            # Extract reg index from directory name
            try:
                reg_idx = int(dir_name.split('_reg')[-1])
            except:
                reg_idx = -1
            
            results.append({
                'name': dir_name,
                'method': 'Manifold',
                'reg_idx': reg_idx,
                'r': r,
                'train_err': train_err,
                'test_err': test_err,
            })
    
    # Sort by reg_idx
    results.sort(key=lambda x: x.get('reg_idx', 999))
    
    # Print POD baseline
    if pod_result:
        print("-" * 80)
        print("POD BASELINE")
        print("-" * 80)
        print(f"{'Method':<15} | {'r':>6} | {'Train Err%':>12} | {'Test Err%':>12}")
        print("-" * 55)
        print(f"{'POD':<15} | {pod_result['r']:>6} | {pod_result['train_err']*100:>12.4f} | {pod_result['test_err']*100:>12.4f}")
        print("")
    
    # Print manifold results
    print("-" * 80)
    print("MANIFOLD RESULTS")
    print("-" * 80)
    print(f"{'Reg Idx':<10} | {'r':>6} | {'Train Err%':>12} | {'Test Err%':>12} | {'Δ Train%':>10} | {'Δ Test%':>10}")
    print("-" * 80)
    
    for res in results:
        delta_train = ""
        delta_test = ""
        if pod_result:
            delta_train = f"{(res['train_err'] - pod_result['train_err'])*100:+.4f}"
            delta_test = f"{(res['test_err'] - pod_result['test_err'])*100:+.4f}"
        
        print(f"{res.get('reg_idx', '?'):<10} | {res['r']:>6} | {res['train_err']*100:>12.4f} | {res['test_err']*100:>12.4f} | {delta_train:>10} | {delta_test:>10}")
    
    print("")
    print("Negative Δ = Manifold better than POD")
    print("")
    
    # Find best manifold
    if results:
        best_train = min(results, key=lambda x: x['train_err'])
        best_test = min(results, key=lambda x: x['test_err'])
        print("-" * 80)
        print("BEST RESULTS")
        print("-" * 80)
        print(f"Best train error: reg_idx={best_train.get('reg_idx', '?')}, error={best_train['train_err']*100:.4f}%")
        print(f"Best test error:  reg_idx={best_test.get('reg_idx', '?')}, error={best_test['test_err']*100:.4f}%")
    
    # Save summary
    summary_file = os.path.join(args.output_base, f"{args.run_name}_summary.yaml")
    summary = {
        'timestamp': datetime.now().isoformat(),
        'pod': pod_result,
        'manifold_results': results,
    }
    with open(summary_file, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)
    print("")
    print(f"Summary saved to: {summary_file}")
    
    # List run directories for OpInf step 2
    print("")
    print("-" * 80)
    print("RUN DIRECTORIES FOR OPINF STEP 2")
    print("-" * 80)
    for d in dirs:
        status_file = os.path.join(d, "pipeline_status.yaml")
        if os.path.exists(status_file):
            print(f"  {d}")


if __name__ == "__main__":
    main()
