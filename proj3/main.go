// Isolation Forest anomaly detector - supports sequential, parallel, BSP, and work-stealing modes
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"sort"
	"strconv"
	"time"

	"proj3-redesigned/data"
	"proj3-redesigned/iforest"
)

type ExecutionMode string

const (
	ModeSequential   ExecutionMode = "sequential"
	ModeParallel     ExecutionMode = "parallel"
	ModeBSP          ExecutionMode = "bsp"
	ModeWorkStealing ExecutionMode = "workstealing"
)

func main() {
	// CLI flags for configuring the detector
	inputFile := flag.String("input", "", "Path to input CSV file")
	outputFile := flag.String("output", "results.csv", "Path to output CSV file")
	mode := flag.String("mode", "sequential", "Execution mode: sequential, parallel, bsp, workstealing")
	numTrees := flag.Int("trees", 100, "Number of trees in the forest")
	sampleSize := flag.Int("sample", 256, "Sample size for each tree")
	numWorkers := flag.Int("workers", 4, "Number of worker threads (for parallel modes)")
	topN := flag.Int("top", 10, "Number of top anomalies to display")
	seed := flag.Int64("seed", 42, "Random seed for reproducibility")
	featureCols := flag.String("features", "", "Comma-separated list of feature column indices (0-indexed)")
	labelCol := flag.Int("label", -1, "Column index for labels (-1 if no labels)")
	skipHeader := flag.Bool("header", true, "Skip header row in CSV")
	benchmark := flag.Bool("benchmark", false, "Run benchmark mode (compare all implementations)")
	verbose := flag.Bool("verbose", false, "Print verbose output")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Parallel Anomaly Detection System using Isolation Forest\n\n")
		fmt.Fprintf(os.Stderr, "Usage: %s [options]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  %s -input data.csv -features 0,1,2,3 -mode sequential\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -input data.csv -features 0,1,2,3 -mode parallel -workers 8\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -input data.csv -features 0,1,2,3 -mode workstealing -workers 8\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -input data.csv -features 0,1,2,3 -benchmark -workers 8\n", os.Args[0])
	}

	flag.Parse()

	// Make sure we have required args
	if *inputFile == "" {
		fmt.Fprintln(os.Stderr, "Error: input file is required")
		flag.Usage()
		os.Exit(1)
	}

	if *featureCols == "" {
		fmt.Fprintln(os.Stderr, "Error: feature columns are required")
		flag.Usage()
		os.Exit(1)
	}

	// Parse the comma-separated feature indices
	featureIndices, err := parseIntList(*featureCols)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing feature columns: %v\n", err)
		os.Exit(1)
	}

	// Load the dataset
	fmt.Printf("Loading dataset from %s...\n", *inputFile)
	dataset, err := data.LoadCSV(*inputFile, featureIndices, *labelCol, *skipHeader)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading dataset: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Loaded %d data points with %d features\n", dataset.Size(), dataset.NumFeatures)

	// Benchmark mode runs all implementations and compares them
	if *benchmark {
		runBenchmark(dataset, *numTrees, *sampleSize, *numWorkers, *seed, *verbose)
		return
	}

	// Run the selected mode
	execMode := ExecutionMode(*mode)
	results, duration := runMode(execMode, dataset, *numTrees, *sampleSize, *numWorkers, *seed, *verbose)

	// Print results
	fmt.Printf("\nExecution completed in %v\n", duration)
	fmt.Printf("\nTop %d Anomalies:\n", *topN)
	fmt.Println("----------------------------------------")

	topAnomalies := iforest.GetTopAnomalies(results, *topN)
	for i, r := range topAnomalies {
		labelStr := ""
		if r.Label != "" {
			labelStr = fmt.Sprintf(" (Label: %s)", r.Label)
		}
		fmt.Printf("%d. Point ID: %d, Score: %.6f, Avg Path Length: %.2f%s\n",
			i+1, r.PointID, r.Score, r.AvgPathLength, labelStr)
	}

	// Save to CSV
	err = saveResults(results, *outputFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error saving results: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("\nResults saved to %s\n", *outputFile)

	// If we have labels, show how well we did
	if *labelCol >= 0 {
		computeMetrics(results, *topN)
	}
}

// runMode executes the specified implementation and returns results + timing
func runMode(mode ExecutionMode, dataset *data.Dataset, numTrees, sampleSize, numWorkers int, seed int64, verbose bool) ([]iforest.ScoreResult, time.Duration) {
	forest := iforest.NewIsolationForest(numTrees, sampleSize)

	var results []iforest.ScoreResult
	var fitDuration, scoreDuration time.Duration

	switch mode {
	case ModeSequential:
		fmt.Println("\nRunning Sequential Implementation...")
		start := time.Now()
		forest.FitSequential(dataset, seed)
		fitDuration = time.Since(start)

		start = time.Now()
		results = forest.ScoreAllSequential(dataset)
		scoreDuration = time.Since(start)

	case ModeParallel:
		fmt.Printf("\nRunning Parallel Implementation with %d workers...\n", numWorkers)
		start := time.Now()
		forest.FitParallel(dataset, seed, numWorkers)
		fitDuration = time.Since(start)

		start = time.Now()
		results = forest.ScoreAllParallel(dataset, numWorkers)
		scoreDuration = time.Since(start)

	case ModeBSP:
		fmt.Printf("\nRunning BSP Implementation with %d workers...\n", numWorkers)
		start := time.Now()
		forest.FitParallelBSP(dataset, seed, numWorkers)
		fitDuration = time.Since(start)

		start = time.Now()
		results = forest.ScoreAllParallel(dataset, numWorkers)
		scoreDuration = time.Since(start)

	case ModeWorkStealing:
		fmt.Printf("\nRunning Work-Stealing Implementation with %d workers...\n", numWorkers)
		start := time.Now()
		totalTasks, totalSteals, perWorkerTasks, perWorkerSteals := forest.FitWorkStealingWithStats(dataset, seed, numWorkers)
		fitDuration = time.Since(start)

		// Show stealing stats if verbose
		if verbose {
			fmt.Printf("  Total tasks: %d, Total steals: %d\n", totalTasks, totalSteals)
			for i := 0; i < numWorkers; i++ {
				fmt.Printf("  Worker %d: %d tasks, %d steals\n", i, perWorkerTasks[i], perWorkerSteals[i])
			}
		}

		start = time.Now()
		results = forest.ScoreAllWorkStealing(dataset, numWorkers)
		scoreDuration = time.Since(start)

	default:
		fmt.Fprintf(os.Stderr, "Unknown mode: %s\n", mode)
		os.Exit(1)
	}

	if verbose {
		fmt.Printf("  Fit time: %v\n", fitDuration)
		fmt.Printf("  Score time: %v\n", scoreDuration)
	}

	return results, fitDuration + scoreDuration
}

// runBenchmark compares all implementations across different worker counts
func runBenchmark(dataset *data.Dataset, numTrees, sampleSize, maxWorkers int, seed int64, verbose bool) {
	fmt.Println("\n========================================")
	fmt.Println("BENCHMARK MODE")
	fmt.Println("========================================")
	fmt.Printf("Dataset size: %d points, %d features\n", dataset.Size(), dataset.NumFeatures)
	fmt.Printf("Forest: %d trees, sample size %d\n", numTrees, sampleSize)
	fmt.Println()

	// Get sequential baseline first
	fmt.Println("Running Sequential (baseline)...")
	_, seqDuration := runMode(ModeSequential, dataset, numTrees, sampleSize, 1, seed, verbose)
	fmt.Printf("Sequential time: %v\n\n", seqDuration)

	// Figure out which worker counts to test
	workerCounts := []int{2, 4, 6, 8, 12}
	if maxWorkers < 12 {
		workerCounts = []int{}
		for w := 2; w <= maxWorkers; w += 2 {
			workerCounts = append(workerCounts, w)
		}
	}

	// Test each parallel implementation
	fmt.Println("Parallel Implementation Speedups:")
	fmt.Println("Workers\tTime\t\tSpeedup")
	fmt.Println("----------------------------------------")
	for _, w := range workerCounts {
		_, duration := runMode(ModeParallel, dataset, numTrees, sampleSize, w, seed, false)
		speedup := float64(seqDuration) / float64(duration)
		fmt.Printf("%d\t%v\t%.2fx\n", w, duration, speedup)
	}

	fmt.Println("\nBSP Implementation Speedups:")
	fmt.Println("Workers\tTime\t\tSpeedup")
	fmt.Println("----------------------------------------")
	for _, w := range workerCounts {
		_, duration := runMode(ModeBSP, dataset, numTrees, sampleSize, w, seed, false)
		speedup := float64(seqDuration) / float64(duration)
		fmt.Printf("%d\t%v\t%.2fx\n", w, duration, speedup)
	}

	fmt.Println("\nWork-Stealing Implementation Speedups:")
	fmt.Println("Workers\tTime\t\tSpeedup")
	fmt.Println("----------------------------------------")
	for _, w := range workerCounts {
		_, duration := runMode(ModeWorkStealing, dataset, numTrees, sampleSize, w, seed, false)
		speedup := float64(seqDuration) / float64(duration)
		fmt.Printf("%d\t%v\t%.2fx\n", w, duration, speedup)
	}
}

// parseIntList splits a comma-separated string into ints
func parseIntList(s string) ([]int, error) {
	var result []int
	start := 0
	for i := 0; i <= len(s); i++ {
		if i == len(s) || s[i] == ',' {
			if start < i {
				val, err := strconv.Atoi(s[start:i])
				if err != nil {
					return nil, err
				}
				result = append(result, val)
			}
			start = i + 1
		}
	}
	return result, nil
}

// saveResults writes anomaly scores to a CSV file, sorted by score descending
func saveResults(results []iforest.ScoreResult, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Header
	err = writer.Write([]string{"PointID", "AnomalyScore", "AvgPathLength", "Label"})
	if err != nil {
		return err
	}

	// Sort by score (highest first)
	sorted := make([]iforest.ScoreResult, len(results))
	copy(sorted, results)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Score > sorted[j].Score
	})

	// Write each result
	for _, r := range sorted {
		err = writer.Write([]string{
			strconv.Itoa(r.PointID),
			fmt.Sprintf("%.6f", r.Score),
			fmt.Sprintf("%.4f", r.AvgPathLength),
			r.Label,
		})
		if err != nil {
			return err
		}
	}

	return nil
}

// computeMetrics shows precision/recall if we have ground truth labels
func computeMetrics(results []iforest.ScoreResult, topN int) {
	topAnomalies := iforest.GetTopAnomalies(results, topN)

	// Count how many of our top predictions are actually anomalies
	truePositives := 0
	for _, r := range topAnomalies {
		if isAnomalyLabel(r.Label) {
			truePositives++
		}
	}

	// Count total anomalies in dataset
	totalAnomalies := 0
	for _, r := range results {
		if isAnomalyLabel(r.Label) {
			totalAnomalies++
		}
	}

	fmt.Printf("\nDetection Metrics (Top %d):\n", topN)
	fmt.Println("----------------------------------------")
	fmt.Printf("True Positives: %d / %d\n", truePositives, topN)
	fmt.Printf("Precision: %.2f%%\n", float64(truePositives)/float64(topN)*100)
	if totalAnomalies > 0 {
		fmt.Printf("Total anomalies in dataset: %d\n", totalAnomalies)
		fmt.Printf("Recall (in top %d): %.2f%%\n", topN, float64(truePositives)/float64(totalAnomalies)*100)
	}
}

// isAnomalyLabel checks if a label indicates an anomaly
// Handles common conventions from various datasets
func isAnomalyLabel(label string) bool {
	anomalyLabels := []string{
		"anomaly", "attack", "fraud", "intrusion", "malicious",
		"1", "true", "yes", "positive",
	}

	// Convert to lowercase
	labelLower := ""
	for _, c := range label {
		if c >= 'A' && c <= 'Z' {
			labelLower += string(c + 32)
		} else {
			labelLower += string(c)
		}
	}

	// Check against known anomaly labels
	for _, anomaly := range anomalyLabels {
		if contains(labelLower, anomaly) {
			return true
		}
	}

	// KDD dataset convention: anything not "normal" is an attack
	if labelLower != "normal" && labelLower != "normal." && labelLower != "" && labelLower != "0" {
		return true
	}

	return false
}

// contains checks if substr is in s
func contains(s, substr string) bool {
	if len(substr) > len(s) {
		return false
	}
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
