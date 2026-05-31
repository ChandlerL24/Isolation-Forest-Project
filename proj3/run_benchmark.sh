#!/bin/bash

# Benchmark script for Parallel Anomaly Detection System
# This script generates test data and runs benchmarks across all implementations

set -e

# Configuration
OUTPUT_DIR="benchmark_results"
DATA_DIR="datasets"
TREES=100
SAMPLE_SIZE=256
SEED=42

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$DATA_DIR"

echo "=============================================="
echo "Parallel Anomaly Detection System - Benchmark"
echo "=============================================="
echo ""

# Build the project
echo "Building project..."
cd "$(dirname "$0")"
go build -o iforest_detector .
go build -o data_generator ./scripts/generate_data.go
echo "Build complete."
echo ""

# Generate test datasets
echo "Generating test datasets..."

# Small dataset (10K points)
./data_generator -n 10000 -features 10 -anomaly-ratio 0.05 -type gaussian -output "$DATA_DIR/small_gaussian.csv"

# Medium dataset (50K points)
./data_generator -n 50000 -features 10 -anomaly-ratio 0.05 -type gaussian -output "$DATA_DIR/medium_gaussian.csv"

# Large dataset (100K points)
./data_generator -n 100000 -features 10 -anomaly-ratio 0.05 -type gaussian -output "$DATA_DIR/large_gaussian.csv"

# Mixed dataset (variable complexity)
./data_generator -n 50000 -features 20 -anomaly-ratio 0.05 -type mixed -output "$DATA_DIR/mixed_data.csv"

# Cluster dataset
./data_generator -n 50000 -features 15 -anomaly-ratio 0.05 -type clusters -output "$DATA_DIR/cluster_data.csv"

echo "Datasets generated."
echo ""

# Function to run benchmark for a dataset
run_benchmark() {
    local dataset=$1
    local name=$2
    local features=$3
    local label_col=$4
    
    echo "=============================================="
    echo "Benchmarking: $name"
    echo "=============================================="
    
    # Run benchmark mode
    ./iforest_detector \
        -input "$dataset" \
        -features "$features" \
        -label "$label_col" \
        -trees $TREES \
        -sample $SAMPLE_SIZE \
        -seed $SEED \
        -benchmark \
        -workers 12 \
        -verbose 2>&1 | tee "$OUTPUT_DIR/${name}_benchmark.txt"
    
    echo ""
}

# Generate feature column string (0,1,2,...,n-1)
generate_features() {
    local n=$1
    local result=""
    for ((i=0; i<n; i++)); do
        if [ $i -gt 0 ]; then
            result="$result,"
        fi
        result="$result$i"
    done
    echo "$result"
}

# Run benchmarks on all datasets
echo ""
echo "Running benchmarks..."
echo ""

# Small dataset benchmark
run_benchmark "$DATA_DIR/small_gaussian.csv" "small_10k" "$(generate_features 10)" 10

# Medium dataset benchmark
run_benchmark "$DATA_DIR/medium_gaussian.csv" "medium_50k" "$(generate_features 10)" 10

# Large dataset benchmark
run_benchmark "$DATA_DIR/large_gaussian.csv" "large_100k" "$(generate_features 10)" 10

# Mixed dataset benchmark
run_benchmark "$DATA_DIR/mixed_data.csv" "mixed_50k" "$(generate_features 20)" 20

# Cluster dataset benchmark
run_benchmark "$DATA_DIR/cluster_data.csv" "cluster_50k" "$(generate_features 15)" 15

echo ""
echo "=============================================="
echo "Benchmark Complete!"
echo "=============================================="
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Individual run examples:"
echo ""
echo "  Sequential:"
echo "    ./iforest_detector -input $DATA_DIR/medium_gaussian.csv -features $(generate_features 10) -label 10 -mode sequential"
echo ""
echo "  Parallel (8 workers):"
echo "    ./iforest_detector -input $DATA_DIR/medium_gaussian.csv -features $(generate_features 10) -label 10 -mode parallel -workers 8"
echo ""
echo "  BSP (8 workers):"
echo "    ./iforest_detector -input $DATA_DIR/medium_gaussian.csv -features $(generate_features 10) -label 10 -mode bsp -workers 8"
echo ""
echo "  Work-Stealing (8 workers):"
echo "    ./iforest_detector -input $DATA_DIR/medium_gaussian.csv -features $(generate_features 10) -label 10 -mode workstealing -workers 8"
