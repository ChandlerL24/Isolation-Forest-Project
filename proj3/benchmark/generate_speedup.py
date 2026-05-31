
"""
Speedup Graph Generator for Project 3 - Isolation Forest
Reads benchmark_results.csv and generates speedup graphs for all parallel implementations
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def read_benchmark_results(filename="benchmark_results.csv"):
    """Read benchmark results from CSV file"""
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row['dataset']
            mode = row['mode']
            threads = int(row['threads'])
            time = float(row['time_seconds'])
            
            results[dataset][mode][threads].append(time)
    
    return results

def calculate_speedups(results):
    """Calculate speedup for each dataset, mode, and thread count"""
    speedups = defaultdict(lambda: defaultdict(dict))
    thread_counts = [2, 4, 6, 8, 12]
    modes = ['parallel', 'bsp', 'workstealing']
    
    for dataset, data in results.items():
        # Get average sequential time
        seq_times = data.get('sequential', {}).get(1, [])
        if not seq_times:
            continue
        avg_seq_time = np.mean(seq_times)
        
        # Calculate speedup for each mode and thread count
        for mode in modes:
            for threads in thread_counts:
                par_times = data.get(mode, {}).get(threads, [])
                if par_times:
                    avg_par_time = np.mean(par_times)
                    speedup = avg_seq_time / avg_par_time
                    speedups[dataset][mode][threads] = speedup
    
    return speedups

def generate_speedup_graphs(speedups, output_prefix="speedup"):
    """Generate speedup graphs - one per parallel implementation"""
    thread_counts = [2, 4, 6, 8, 12]
    datasets = ['small', 'medium', 'large']
    modes = ['parallel', 'bsp', 'workstealing']
    mode_names = {
        'parallel': 'Channel-Based Parallel',
        'bsp': 'BSP (Bulk Synchronous Parallel)',
        'workstealing': 'Work-Stealing'
    }
    
    colors = {'small': '#1f77b4', 'medium': '#ff7f0e', 'large': '#2ca02c'}
    markers = {'small': 'o', 'medium': 's', 'large': '^'}
    
    # Generate one graph per mode
    for mode in modes:
        plt.figure(figsize=(10, 7))
        
        for dataset in datasets:
            if dataset in speedups and mode in speedups[dataset]:
                x_vals = []
                y_vals = []
                for threads in thread_counts:
                    if threads in speedups[dataset][mode]:
                        x_vals.append(threads)
                        y_vals.append(speedups[dataset][mode][threads])
                
                if x_vals:
                    plt.plot(x_vals, y_vals, marker=markers[dataset], color=colors[dataset],
                            linewidth=2, markersize=8, label=f'{dataset} dataset')
        
        # Add ideal speedup line
        plt.plot(thread_counts, thread_counts, 'k--', alpha=0.3, label='Ideal (linear)')
        
        plt.xlabel('Number of Threads', fontsize=12)
        plt.ylabel('Speedup (Sequential Time / Parallel Time)', fontsize=12)
        plt.title(f'Speedup: {mode_names[mode]}', fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(thread_counts)
        plt.ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_{mode}.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'{output_prefix}_{mode}.pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved {output_prefix}_{mode}.png and .pdf")
    
    # Generate combined comparison graph (all modes, medium dataset)
    plt.figure(figsize=(10, 7))
    mode_colors = {'parallel': '#1f77b4', 'bsp': '#ff7f0e', 'workstealing': '#2ca02c'}
    mode_markers = {'parallel': 'o', 'bsp': 's', 'workstealing': '^'}
    
    dataset = 'medium'  # Use medium dataset for comparison
    if dataset in speedups:
        for mode in modes:
            if mode in speedups[dataset]:
                x_vals = []
                y_vals = []
                for threads in thread_counts:
                    if threads in speedups[dataset][mode]:
                        x_vals.append(threads)
                        y_vals.append(speedups[dataset][mode][threads])
                
                if x_vals:
                    plt.plot(x_vals, y_vals, marker=mode_markers[mode], color=mode_colors[mode],
                            linewidth=2, markersize=8, label=mode_names[mode])
    
    plt.plot(thread_counts, thread_counts, 'k--', alpha=0.3, label='Ideal (linear)')
    
    plt.xlabel('Number of Threads', fontsize=12)
    plt.ylabel('Speedup (Sequential Time / Parallel Time)', fontsize=12)
    plt.title('Speedup Comparison: All Implementations (Medium Dataset)', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(thread_counts)
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_prefix}_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved {output_prefix}_comparison.png and .pdf")

def print_summary(results, speedups):
    """Print a summary of the benchmark results"""
    thread_counts = [2, 4, 6, 8, 12]
    datasets = ['small', 'medium', 'large']
    modes = ['parallel', 'bsp', 'workstealing']
    
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    
    for dataset in datasets:
        if dataset not in results:
            continue
            
        print(f"\n{dataset.upper()} Dataset:")
        print("-" * 60)
        
        # Sequential
        seq_times = results[dataset].get('sequential', {}).get(1, [])
        if seq_times:
            print(f"  Sequential: {np.mean(seq_times):.4f}s (avg of {len(seq_times)} runs)")
        
        # Each parallel mode
        for mode in modes:
            print(f"\n  {mode.upper()}:")
            for threads in thread_counts:
                par_times = results[dataset].get(mode, {}).get(threads, [])
                if par_times:
                    speedup = speedups[dataset].get(mode, {}).get(threads, 0)
                    print(f"    {threads:2d} threads: {np.mean(par_times):.4f}s (speedup: {speedup:.2f}x)")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    print("Reading benchmark results...")
    results = read_benchmark_results()
    
    print("Calculating speedups...")
    speedups = calculate_speedups(results)
    
    print_summary(results, speedups)
    
    print("\nGenerating speedup graphs...")
    generate_speedup_graphs(speedups)
    
    print("\nDone!")
