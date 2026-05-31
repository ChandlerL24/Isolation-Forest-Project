// Data generation script for testing the anomaly detection system
// Generates synthetic datasets with known anomalies for benchmarking
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
)

func main() {
	numPoints := flag.Int("n", 10000, "Number of data points")
	numFeatures := flag.Int("features", 10, "Number of features")
	anomalyRatio := flag.Float64("anomaly-ratio", 0.05, "Ratio of anomalies (0.0-1.0)")
	outputFile := flag.String("output", "synthetic_data.csv", "Output file path")
	seed := flag.Int64("seed", 42, "Random seed")
	dataType := flag.String("type", "gaussian", "Data type: gaussian, clusters, mixed")

	flag.Parse()

	rng := rand.New(rand.NewSource(*seed))

	var data [][]float64
	var labels []string

	switch *dataType {
	case "gaussian":
		data, labels = generateGaussianData(*numPoints, *numFeatures, *anomalyRatio, rng)
	case "clusters":
		data, labels = generateClusterData(*numPoints, *numFeatures, *anomalyRatio, rng)
	case "mixed":
		data, labels = generateMixedData(*numPoints, *numFeatures, *anomalyRatio, rng)
	default:
		fmt.Fprintf(os.Stderr, "Unknown data type: %s\n", *dataType)
		os.Exit(1)
	}

	// Write to CSV
	err := writeCSV(*outputFile, data, labels, *numFeatures)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error writing CSV: %v\n", err)
		os.Exit(1)
	}

	// Count anomalies
	anomalyCount := 0
	for _, l := range labels {
		if l == "anomaly" {
			anomalyCount++
		}
	}

	fmt.Printf("Generated %d data points with %d features\n", *numPoints, *numFeatures)
	fmt.Printf("Anomalies: %d (%.2f%%)\n", anomalyCount, float64(anomalyCount)/float64(*numPoints)*100)
	fmt.Printf("Output: %s\n", *outputFile)
}

// generateGaussianData creates data from a multivariate Gaussian distribution
// Anomalies are points far from the mean
func generateGaussianData(n, numFeatures int, anomalyRatio float64, rng *rand.Rand) ([][]float64, []string) {
	data := make([][]float64, n)
	labels := make([]string, n)

	numAnomalies := int(float64(n) * anomalyRatio)

	for i := 0; i < n; i++ {
		point := make([]float64, numFeatures)

		if i < numAnomalies {
			// Generate anomaly: far from center
			for j := 0; j < numFeatures; j++ {
				// Anomalies are 3-5 standard deviations away
				sign := 1.0
				if rng.Float64() < 0.5 {
					sign = -1.0
				}
				point[j] = sign * (3.0 + rng.Float64()*2.0) * (1.0 + rng.Float64())
			}
			labels[i] = "anomaly"
		} else {
			// Generate normal point: standard normal distribution
			for j := 0; j < numFeatures; j++ {
				point[j] = rng.NormFloat64()
			}
			labels[i] = "normal"
		}

		data[i] = point
	}

	// Shuffle data
	shuffle(data, labels, rng)

	return data, labels
}

// generateClusterData creates data with multiple clusters
// Anomalies are points that don't belong to any cluster
func generateClusterData(n, numFeatures int, anomalyRatio float64, rng *rand.Rand) ([][]float64, []string) {
	data := make([][]float64, n)
	labels := make([]string, n)

	numAnomalies := int(float64(n) * anomalyRatio)
	numClusters := 5

	// Generate cluster centers
	centers := make([][]float64, numClusters)
	for c := 0; c < numClusters; c++ {
		centers[c] = make([]float64, numFeatures)
		for j := 0; j < numFeatures; j++ {
			centers[c][j] = rng.Float64()*10 - 5 // Centers between -5 and 5
		}
	}

	for i := 0; i < n; i++ {
		point := make([]float64, numFeatures)

		if i < numAnomalies {
			// Generate anomaly: random point far from all clusters
			for j := 0; j < numFeatures; j++ {
				point[j] = rng.Float64()*20 - 10 // Wider range
			}
			// Ensure it's far from all cluster centers
			for {
				minDist := math.MaxFloat64
				for _, center := range centers {
					dist := euclideanDistance(point, center)
					if dist < minDist {
						minDist = dist
					}
				}
				if minDist > 5.0 { // Far enough from all clusters
					break
				}
				// Regenerate
				for j := 0; j < numFeatures; j++ {
					point[j] = rng.Float64()*30 - 15
				}
			}
			labels[i] = "anomaly"
		} else {
			// Generate normal point: belongs to a random cluster
			clusterIdx := rng.Intn(numClusters)
			for j := 0; j < numFeatures; j++ {
				point[j] = centers[clusterIdx][j] + rng.NormFloat64()*0.5
			}
			labels[i] = "normal"
		}

		data[i] = point
	}

	shuffle(data, labels, rng)

	return data, labels
}

// generateMixedData creates data with varying densities and patterns
// This creates more challenging scenarios for anomaly detection
func generateMixedData(n, numFeatures int, anomalyRatio float64, rng *rand.Rand) ([][]float64, []string) {
	data := make([][]float64, n)
	labels := make([]string, n)

	numAnomalies := int(float64(n) * anomalyRatio)

	for i := 0; i < n; i++ {
		point := make([]float64, numFeatures)

		if i < numAnomalies {
			// Different types of anomalies
			anomalyType := rng.Intn(3)
			switch anomalyType {
			case 0:
				// Global outlier: extreme values
				for j := 0; j < numFeatures; j++ {
					sign := 1.0
					if rng.Float64() < 0.5 {
						sign = -1.0
					}
					point[j] = sign * (5.0 + rng.Float64()*5.0)
				}
			case 1:
				// Local outlier: normal in some dimensions, extreme in others
				extremeDims := rng.Intn(numFeatures/2) + 1
				for j := 0; j < numFeatures; j++ {
					if j < extremeDims {
						sign := 1.0
						if rng.Float64() < 0.5 {
							sign = -1.0
						}
						point[j] = sign * (4.0 + rng.Float64()*3.0)
					} else {
						point[j] = rng.NormFloat64()
					}
				}
			case 2:
				// Contextual outlier: unusual combination of normal values
				for j := 0; j < numFeatures; j++ {
					// Values that are individually normal but unusual together
					if j%2 == 0 {
						point[j] = 2.0 + rng.Float64()*0.5
					} else {
						point[j] = -2.0 - rng.Float64()*0.5
					}
				}
			}
			labels[i] = "anomaly"
		} else {
			// Normal data: mixture of distributions
			distType := rng.Intn(3)
			switch distType {
			case 0:
				// Standard normal
				for j := 0; j < numFeatures; j++ {
					point[j] = rng.NormFloat64()
				}
			case 1:
				// Shifted normal
				for j := 0; j < numFeatures; j++ {
					point[j] = rng.NormFloat64() + 1.0
				}
			case 2:
				// Scaled normal
				for j := 0; j < numFeatures; j++ {
					point[j] = rng.NormFloat64() * 0.5
				}
			}
			labels[i] = "normal"
		}

		data[i] = point
	}

	shuffle(data, labels, rng)

	return data, labels
}

func euclideanDistance(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

func shuffle(data [][]float64, labels []string, rng *rand.Rand) {
	n := len(data)
	for i := n - 1; i > 0; i-- {
		j := rng.Intn(i + 1)
		data[i], data[j] = data[j], data[i]
		labels[i], labels[j] = labels[j], labels[i]
	}
}

func writeCSV(filename string, data [][]float64, labels []string, numFeatures int) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	header := make([]string, numFeatures+1)
	for i := 0; i < numFeatures; i++ {
		header[i] = fmt.Sprintf("feature_%d", i)
	}
	header[numFeatures] = "label"
	if err := writer.Write(header); err != nil {
		return err
	}

	// Write data
	for i, point := range data {
		row := make([]string, numFeatures+1)
		for j, val := range point {
			row[j] = strconv.FormatFloat(val, 'f', 6, 64)
		}
		row[numFeatures] = labels[i]
		if err := writer.Write(row); err != nil {
			return err
		}
	}

	return nil
}
