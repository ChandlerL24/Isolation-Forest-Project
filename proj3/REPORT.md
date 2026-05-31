# Parallel Isolation Forest for Anomaly Detection

## What This Project Does

I built a parallel anomaly detection system using the Isolation Forest algorithm. The idea is simple: anomalies are "different" from normal data, so they're easier to isolate. If you randomly partition data with decision trees, anomalies end up isolated in fewer splits than normal points.

The algorithm:
1. Build a bunch of random trees, each on a subsample of data
2. For each tree, randomly pick features and split values to partition points
3. Anomalies get isolated quickly (short path), normal points take longer (deep path)
4. Average the path lengths across all trees → shorter average = more anomalous

This is great for parallelization because each tree is completely independent.

## Project Structure

```
proj3/
├── main.go              # CLI, benchmarking
├── data/data.go         # CSV loading
├── iforest/
│   ├── tree.go          # Tree structures, building, scoring
│   ├── sequential.go    # Baseline implementation
│   ├── parallel.go      # Channel-based + BSP versions
│   └── workstealing.go  # Work-stealing version
└── workstealing/
    ├── deque.go         # Lock-free deque (CAS-based)
    └── scheduler.go     # Worker pool management
```

## The Three Parallel Implementations

### 1. Channel-Based (parallel mode)

Pretty straightforward Go concurrency. Workers pull tasks from a channel, build trees, send results back through another channel.

```go
taskChan := make(chan TreeTask, numTrees)
resultChan := make(chan TreeResult, numTrees)

// Workers grab tasks until channel closes
for task := range taskChan {
    tree := BuildTree(task.SamplePoints, ...)
    resultChan <- TreeResult{task.TreeIndex, tree}
}
```

This works well because Go's channels handle the synchronization for us. Load balancing happens naturally since workers just grab the next available task.

### 2. BSP (bsp mode)

Bulk Synchronous Parallel - process trees in batches with barriers between each batch. Each "superstep" processes `numWorkers` trees, then everyone waits at a barrier before the next batch.

```go
for superstepStart := 0; superstepStart < numTrees; superstepStart += numWorkers {
    // Barrier: wait for all workers to be ready
    mu.Lock()
    ready++
    if ready == numInSuperstep {
        cond.Broadcast()
    } else {
        cond.Wait()
    }
    mu.Unlock()
    
    // Build tree
    // ...
    
    wg.Wait() // Barrier at end of superstep
}
```

Honestly, BSP is overkill for this problem since there's no communication between trees. But it's required for the assignment and shows how you'd do it if there were dependencies.

### 3. Work-Stealing (workstealing mode)

This is the interesting one. Each worker has its own deque (double-ended queue). Workers push/pop from the bottom of their own deque, but can steal from the top of other workers' deques when they run out of work.

The deque is lock-free using CAS (compare-and-swap):

```go
func (d *Deque) PushBottom(task *Task) {
    newNode := &node{task: task}
    for {
        bottom := atomic.LoadPointer(&d.bottom)
        newNode.next = bottom
        if atomic.CompareAndSwapPointer(&d.bottom, bottom, newNode) {
            return  // Success
        }
        // CAS failed, someone else modified it, retry
    }
}
```

To show off work-stealing, I intentionally create imbalanced workloads - first worker gets half the tasks, rest are distributed evenly. Without stealing, the first worker would be swamped while others sit idle. With stealing, idle workers grab tasks from the overloaded one.

## Why Isolation Forest Parallelizes Well

**Hotspots (parallelizable):**
- Tree building - each tree is independent, no shared state
- Scoring - each point can be scored independently

**Bottlenecks (sequential):**
- Random sampling - need to generate samples before distributing tasks
- Result aggregation - minor, just copying results to an array

The tree building is embarrassingly parallel. No communication needed between workers during the main computation phase.

## Challenges

**Thread-safe RNG:** Can't share a random number generator across goroutines without locks. Solution: generate seeds sequentially, then each task uses its own RNG with its assigned seed. This keeps things reproducible too.

**Lock-free deque correctness:** Getting CAS loops right is tricky. The key insight is that the owner (push/pop bottom) and thieves (steal from top) operate on opposite ends, so contention is minimized. When there's only one element left, both might try to grab it - CAS ensures only one succeeds.

**Termination detection:** How do workers know when all work is done? Can't just check your own deque - others might still have work. Solution: check all deques, and if all empty, signal done. First worker to detect this sets a flag.

## Running It

Build:
```bash
cd proj3
go build -o iforest_detector .
```

Run different modes:
```bash
# Sequential baseline
./iforest_detector -input data.csv -features 0,1,2,3 -mode sequential

# Parallel with 8 workers
./iforest_detector -input data.csv -features 0,1,2,3 -mode parallel -workers 8

# BSP
./iforest_detector -input data.csv -features 0,1,2,3 -mode bsp -workers 8

# Work-stealing
./iforest_detector -input data.csv -features 0,1,2,3 -mode workstealing -workers 8

# Benchmark all modes
./iforest_detector -input data.csv -features 0,1,2,3 -benchmark -workers 12
```

Or just run the benchmark script (locally):
```bash
./run_benchmark.sh
```

## Running on the Cluster (HPC)

The final benchmarks should be run on the UChicago CS cluster (`fe.ai.cs.uchicago.edu`, `peanut-cpu` partition).

### Setup

1. SSH into the cluster:
   ```bash
   ssh celawrence@fe.ai.cs.uchicago.edu
   ```

2. Clone/update your repo:
   ```bash
   cd ~
   git clone git@github.com:mpcs-jh/project-3-ChandlerL24-1.git
   # or if already cloned:
   cd project-3-ChandlerL24-1 && git pull
   ```

3. Navigate to the benchmark directory:
   ```bash
   cd project-3-ChandlerL24-1/proj3/benchmark
   ```

### Submit the SLURM Job

```bash
sbatch run_benchmarks.sh
```

This will:
- Request 16 CPUs on the `peanut-cpu` partition
- Run benchmarks for all modes (sequential, parallel, bsp, workstealing)
- Test with thread counts: 2, 4, 6, 8, 12
- Run 5 iterations per configuration for statistical significance
- Output results to `benchmark_results.csv`
- Generate speedup graphs automatically

### Monitor Job Status

```bash
squeue -u celawrence
```

### View Output

After the job completes:
```bash
# Check stdout
cat slurm/out/*.stdout

# Check for errors
cat slurm/out/*.stderr

# View results
cat benchmark_results.csv
```

### Generated Files

After running on the cluster, you'll have:
- `benchmark_results.csv` - Raw timing data
- `speedup_parallel.png/pdf` - Channel-based parallel speedup graph
- `speedup_bsp.png/pdf` - BSP speedup graph
- `speedup_workstealing.png/pdf` - Work-stealing speedup graph
- `speedup_comparison.png/pdf` - All implementations compared

## Experimental Results

Benchmarks were run on the UChicago CS cluster (`peanut-cpu` partition) with 16 CPUs, testing thread counts of 2, 4, 6, 8, and 12 with 5 runs per configuration.

### Small Dataset (10,000 points, 10 features)

| Threads | Parallel | BSP | Work-Stealing |
|---------|----------|-----|---------------|
| Sequential | 166.9s | 166.9s | 166.9s |
| 2 | 107.4s (1.55x) | 112.8s (1.48x) | 103.4s (1.61x) |
| 4 | 70.9s (2.36x) | 79.0s (2.11x) | 61.2s (2.73x) |
| 6 | 63.6s (2.63x) | 64.9s (2.57x) | 51.0s (3.27x) |
| 8 | 55.9s (2.98x) | 56.2s (2.97x) | 46.3s (3.60x) |
| 12 | 45.1s (3.70x) | 45.3s (3.69x) | 48.5s (3.44x) |

### Medium Dataset (50,000 points, 10 features)

| Threads | Parallel | BSP | Work-Stealing |
|---------|----------|-----|---------------|
| Sequential | 740.6s | 740.6s | 740.6s |
| 2 | 413.1s (1.79x) | 427.3s (1.73x) | 413.3s (1.79x) |
| 4 | 229.6s (3.23x) | 251.2s (2.95x) | 232.8s (3.18x) |
| 6 | 181.6s (4.08x) | 192.6s (3.85x) | 176.7s (4.19x) |
| 8 | 151.5s (4.89x) | 158.4s (4.68x) | 148.3s (4.99x) |
| 12 | 130.3s (5.68x) | 131.2s (5.64x) | **115.9s (6.39x)** |

### Large Dataset (100,000 points, 10 features)

| Threads | Parallel | BSP | Work-Stealing |
|---------|----------|-----|---------------|
| Sequential | ~1450s | ~1450s | ~1450s |
| 2 | 780.2s (1.86x) | 809.3s (1.79x) | 787.3s (1.84x) |
| 4 | 436.0s (3.33x) | 459.3s (3.16x) | 437.9s (3.31x) |
| 6 | 314.5s (4.61x) | 323.1s (4.49x) | 321.1s (4.52x) |
| 8 | 259.4s (5.59x) | 283.1s (5.12x) | 258.3s (5.61x) |
| 12 | 202.5s (7.16x) | 217.3s (6.67x) | **205.1s (7.07x)** |

### Analysis

**Work-stealing performed best overall**, especially at higher thread counts on larger datasets. The intentional load imbalance (first worker gets half the tasks) was effectively mitigated by the stealing mechanism.

**Channel-based parallel** was competitive and had the most consistent performance due to Go's efficient channel implementation providing natural load balancing.

**BSP** was consistently the slowest due to barrier synchronization overhead. Since tree building has no inter-task dependencies, the barriers add unnecessary waiting time.

### Why Speedup Isn't Linear

1. **Sequential overhead**: Random sampling and result aggregation are sequential
2. **Memory bandwidth**: With 12 threads, memory becomes the bottleneck
3. **Synchronization costs**: Even lock-free operations have cache coherency overhead
4. **Amdahl's Law**: The sequential fraction limits maximum speedup

### Did Work-Stealing Help?

**Yes.** Work-stealing showed the best speedups, particularly on the medium dataset where it achieved **6.39x speedup at 12 threads** compared to 5.68x for channel-based. The imbalanced task distribution was successfully rebalanced through stealing.

## Command Line Options

| Flag | Default | What it does |
|------|---------|--------------|
| `-input` | required | Input CSV file |
| `-features` | required | Which columns are features (0-indexed, comma-separated) |
| `-mode` | sequential | sequential, parallel, bsp, or workstealing |
| `-workers` | 4 | Number of worker goroutines |
| `-trees` | 100 | Number of trees in the forest |
| `-sample` | 256 | Subsample size per tree |
| `-label` | -1 | Column with labels (-1 if none) |
| `-benchmark` | false | Run all modes and compare |
| `-verbose` | false | Print extra stats |

## Anomaly Score Interpretation

- Score > 0.6: Probably an anomaly
- Score 0.5-0.6: Borderline
- Score < 0.5: Probably normal

Higher score = shorter average path length = easier to isolate = more anomalous.
