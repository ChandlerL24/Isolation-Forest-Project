// CSV data loading for anomaly detection
package data

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
)

// Point is a single data point with its features
type Point struct {
	Features []float64
	Label    string // optional - for validation if we have ground truth
	ID       int    // original index in the dataset
}

// Dataset holds all the points and some metadata
type Dataset struct {
	Points       []Point
	NumFeatures  int
	FeatureNames []string
}

// LoadCSV reads a CSV file and extracts the specified feature columns
// featureCols: which columns to use as features (0-indexed)
// labelCol: which column has labels (-1 if none)
// skipHeader: whether to skip the first row
func LoadCSV(filename string, featureCols []int, labelCol int, skipHeader bool) (*Dataset, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(bufio.NewReader(file))

	var dataset Dataset
	dataset.NumFeatures = len(featureCols)

	lineNum := 0
	pointID := 0

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("error reading line %d: %w", lineNum, err)
		}

		lineNum++

		// Handle header row
		if skipHeader && lineNum == 1 {
			// Grab feature names if available
			for _, col := range featureCols {
				if col < len(record) {
					dataset.FeatureNames = append(dataset.FeatureNames, record[col])
				}
			}
			continue
		}

		// Parse the feature values
		features := make([]float64, len(featureCols))
		valid := true
		for i, col := range featureCols {
			if col >= len(record) {
				valid = false
				break
			}
			val, err := strconv.ParseFloat(record[col], 64)
			if err != nil {
				valid = false
				break
			}
			features[i] = val
		}

		// Skip rows with invalid/missing data
		if !valid {
			continue
		}

		// Get label if specified
		label := ""
		if labelCol >= 0 && labelCol < len(record) {
			label = record[labelCol]
		}

		point := Point{
			Features: features,
			Label:    label,
			ID:       pointID,
		}
		dataset.Points = append(dataset.Points, point)
		pointID++
	}

	if len(dataset.Points) == 0 {
		return nil, fmt.Errorf("no valid data points found in file")
	}

	return &dataset, nil
}

// Subset returns a new dataset with only the specified indices
func (d *Dataset) Subset(indices []int) *Dataset {
	subset := &Dataset{
		NumFeatures:  d.NumFeatures,
		FeatureNames: d.FeatureNames,
		Points:       make([]Point, len(indices)),
	}

	for i, idx := range indices {
		subset.Points[i] = d.Points[idx]
	}

	return subset
}

// GetFeatureRange returns min and max values for a feature
func (d *Dataset) GetFeatureRange(featureIdx int) (float64, float64) {
	if len(d.Points) == 0 || featureIdx >= d.NumFeatures {
		return 0, 0
	}

	min := d.Points[0].Features[featureIdx]
	max := d.Points[0].Features[featureIdx]

	for _, p := range d.Points[1:] {
		if p.Features[featureIdx] < min {
			min = p.Features[featureIdx]
		}
		if p.Features[featureIdx] > max {
			max = p.Features[featureIdx]
		}
	}

	return min, max
}

// Size returns number of points
func (d *Dataset) Size() int {
	return len(d.Points)
}
