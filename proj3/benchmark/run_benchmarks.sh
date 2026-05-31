#!/bin/bash
#
#SBATCH --mail-user=celawrence@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=proj3_iforest
#SBATCH --output=./slurm/out/%j.%N.stdout
#SBATCH --error=./slurm/out/%j.%N.stderr
#SBATCH --chdir=/home/celawrence/project-3-ChandlerL24-1/proj3/benchmark
#SBATCH --partition=peanut-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=900
#SBATCH --exclusive
#SBATCH --time=01:00:00

module load golang/1.19

# Configuration
OUTPUT_FILE="benchmark_results.csv"
DATA_DIR="../datasets"
TREES=100
SAMPLE_SIZE=256
SEED=42

# Test configurations
DATASET_SIZES=("small" "medium" "large")
THREAD_COUNTS=(2 4 6 8 12)
NUM_RUNS=5

# Build the project first
echo "Building project..."
cd ..
go build -o iforest_detector .
go build -o data_generator ./scripts/generate_data.go
cd benchmark

# Generate test datasets if they don't exist
mkdir -p "$DATA_DIR"
if [ ! -f "$DATA_DIR/small_gaussian.csv" ]; then
    echo "Generating test datasets..."
    ../data_generator -n 10000 -features 10 -anomaly-ratio 0.05 -type gaussian -output "$DATA_DIR/small_gaussian.csv"
    ../data_generator -n 50000 -features 10 -anomaly-ratio 0.05 -type gaussian -output "$DATA_DIR/medium_gaussian.csv"
    ../data_generator -n 100000 -features 10 -anomaly-ratio 0.05 -type gaussian -output "$DATA_DIR/large_gaussian.csv"
fi

# Feature columns for 10 features
FEATURES="0,1,2,3,4,5,6,7,8,9"
LABEL_COL=10

# Initialize CSV output
echo "dataset,mode,threads,run,time_seconds" > $OUTPUT_FILE

# Function to extract time from output (expects format like "Execution completed in 1.234s")
extract_time() {
    grep "Execution completed" | sed 's/.*in \([0-9.]*\).*/\1/' | sed 's/s$//' | sed 's/ms$/e-3/' | sed 's/µs$/e-6/'
}

echo "=============================================="
echo "Project 3: Isolation Forest Benchmark"
echo "=============================================="
echo ""

# Run sequential benchmarks
echo "Running sequential benchmarks..."
for size in "${DATASET_SIZES[@]}"; do
    dataset="$DATA_DIR/${size}_gaussian.csv"
    for run in $(seq 1 $NUM_RUNS); do
        time_result=$(../iforest_detector -input "$dataset" -features "$FEATURES" -label $LABEL_COL -trees $TREES -sample $SAMPLE_SIZE -seed $SEED -mode sequential 2>&1 | extract_time)
        echo "$size,sequential,1,$run,$time_result" >> $OUTPUT_FILE
        echo "Sequential $size run $run: ${time_result}s"
    done
done

# Run parallel (channel-based) benchmarks
echo ""
echo "Running parallel (channel-based) benchmarks..."
for size in "${DATASET_SIZES[@]}"; do
    dataset="$DATA_DIR/${size}_gaussian.csv"
    for threads in "${THREAD_COUNTS[@]}"; do
        for run in $(seq 1 $NUM_RUNS); do
            time_result=$(../iforest_detector -input "$dataset" -features "$FEATURES" -label $LABEL_COL -trees $TREES -sample $SAMPLE_SIZE -seed $SEED -mode parallel -workers $threads 2>&1 | extract_time)
            echo "$size,parallel,$threads,$run,$time_result" >> $OUTPUT_FILE
            echo "Parallel $size threads=$threads run $run: ${time_result}s"
        done
    done
done

# Run BSP benchmarks
echo ""
echo "Running BSP benchmarks..."
for size in "${DATASET_SIZES[@]}"; do
    dataset="$DATA_DIR/${size}_gaussian.csv"
    for threads in "${THREAD_COUNTS[@]}"; do
        for run in $(seq 1 $NUM_RUNS); do
            time_result=$(../iforest_detector -input "$dataset" -features "$FEATURES" -label $LABEL_COL -trees $TREES -sample $SAMPLE_SIZE -seed $SEED -mode bsp -workers $threads 2>&1 | extract_time)
            echo "$size,bsp,$threads,$run,$time_result" >> $OUTPUT_FILE
            echo "BSP $size threads=$threads run $run: ${time_result}s"
        done
    done
done

# Run work-stealing benchmarks
echo ""
echo "Running work-stealing benchmarks..."
for size in "${DATASET_SIZES[@]}"; do
    dataset="$DATA_DIR/${size}_gaussian.csv"
    for threads in "${THREAD_COUNTS[@]}"; do
        for run in $(seq 1 $NUM_RUNS); do
            time_result=$(../iforest_detector -input "$dataset" -features "$FEATURES" -label $LABEL_COL -trees $TREES -sample $SAMPLE_SIZE -seed $SEED -mode workstealing -workers $threads 2>&1 | extract_time)
            echo "$size,workstealing,$threads,$run,$time_result" >> $OUTPUT_FILE
            echo "Work-stealing $size threads=$threads run $run: ${time_result}s"
        done
    done
done

echo ""
echo "=============================================="
echo "Benchmarks complete! Results saved to $OUTPUT_FILE"
echo "=============================================="

# Generate speedup graphs
echo ""
echo "Generating speedup graphs..."
python3 generate_speedup.py

echo "Done."
