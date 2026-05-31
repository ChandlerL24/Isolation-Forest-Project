package data

import (
	"os"
	"path/filepath"
	"testing"
)

func TestPointStruct(t *testing.T) {
	p := Point{
		Features: []float64{1.0, 2.0, 3.0},
		Label:    "normal",
		ID:       42,
	}

	if len(p.Features) != 3 {
		t.Errorf("Expected 3 features, got %d", len(p.Features))
	}
	if p.Label != "normal" {
		t.Errorf("Expected label 'normal', got '%s'", p.Label)
	}
	if p.ID != 42 {
		t.Errorf("Expected ID 42, got %d", p.ID)
	}
}

func TestDatasetSize(t *testing.T) {
	ds := &Dataset{
		Points: []Point{
			{Features: []float64{1.0}, ID: 0},
			{Features: []float64{2.0}, ID: 1},
			{Features: []float64{3.0}, ID: 2},
		},
		NumFeatures: 1,
	}

	if ds.Size() != 3 {
		t.Errorf("Expected size 3, got %d", ds.Size())
	}
}

func TestDatasetSubset(t *testing.T) {
	ds := &Dataset{
		Points: []Point{
			{Features: []float64{1.0}, ID: 0},
			{Features: []float64{2.0}, ID: 1},
			{Features: []float64{3.0}, ID: 2},
			{Features: []float64{4.0}, ID: 3},
		},
		NumFeatures:  1,
		FeatureNames: []string{"feature1"},
	}

	subset := ds.Subset([]int{1, 3})

	if subset.Size() != 2 {
		t.Errorf("Expected subset size 2, got %d", subset.Size())
	}
	if subset.Points[0].ID != 1 {
		t.Errorf("Expected first point ID 1, got %d", subset.Points[0].ID)
	}
	if subset.Points[1].ID != 3 {
		t.Errorf("Expected second point ID 3, got %d", subset.Points[1].ID)
	}
	if subset.NumFeatures != 1 {
		t.Errorf("Expected NumFeatures 1, got %d", subset.NumFeatures)
	}
}

func TestGetFeatureRange(t *testing.T) {
	ds := &Dataset{
		Points: []Point{
			{Features: []float64{1.0, 10.0}},
			{Features: []float64{5.0, 20.0}},
			{Features: []float64{3.0, 15.0}},
		},
		NumFeatures: 2,
	}

	min0, max0 := ds.GetFeatureRange(0)
	if min0 != 1.0 || max0 != 5.0 {
		t.Errorf("Feature 0: expected range [1.0, 5.0], got [%f, %f]", min0, max0)
	}

	min1, max1 := ds.GetFeatureRange(1)
	if min1 != 10.0 || max1 != 20.0 {
		t.Errorf("Feature 1: expected range [10.0, 20.0], got [%f, %f]", min1, max1)
	}
}

func TestGetFeatureRangeEmpty(t *testing.T) {
	ds := &Dataset{
		Points:      []Point{},
		NumFeatures: 1,
	}

	min, max := ds.GetFeatureRange(0)
	if min != 0 || max != 0 {
		t.Errorf("Empty dataset: expected range [0, 0], got [%f, %f]", min, max)
	}
}

func TestGetFeatureRangeInvalidIndex(t *testing.T) {
	ds := &Dataset{
		Points: []Point{
			{Features: []float64{1.0}},
		},
		NumFeatures: 1,
	}

	min, max := ds.GetFeatureRange(5) // Invalid index
	if min != 0 || max != 0 {
		t.Errorf("Invalid index: expected range [0, 0], got [%f, %f]", min, max)
	}
}

func TestLoadCSV(t *testing.T) {
	// Create a temporary CSV file
	tmpDir := t.TempDir()
	csvPath := filepath.Join(tmpDir, "test.csv")

	csvContent := `feature1,feature2,label
1.0,2.0,normal
3.0,4.0,anomaly
5.0,6.0,normal
`
	err := os.WriteFile(csvPath, []byte(csvContent), 0644)
	if err != nil {
		t.Fatalf("Failed to create test CSV: %v", err)
	}

	// Load the CSV
	ds, err := LoadCSV(csvPath, []int{0, 1}, 2, true)
	if err != nil {
		t.Fatalf("LoadCSV failed: %v", err)
	}

	if ds.Size() != 3 {
		t.Errorf("Expected 3 points, got %d", ds.Size())
	}
	if ds.NumFeatures != 2 {
		t.Errorf("Expected 2 features, got %d", ds.NumFeatures)
	}
	if len(ds.FeatureNames) != 2 {
		t.Errorf("Expected 2 feature names, got %d", len(ds.FeatureNames))
	}
	if ds.FeatureNames[0] != "feature1" {
		t.Errorf("Expected feature name 'feature1', got '%s'", ds.FeatureNames[0])
	}

	// Check first point
	if ds.Points[0].Features[0] != 1.0 || ds.Points[0].Features[1] != 2.0 {
		t.Errorf("First point features incorrect: %v", ds.Points[0].Features)
	}
	if ds.Points[0].Label != "normal" {
		t.Errorf("First point label incorrect: %s", ds.Points[0].Label)
	}
}

func TestLoadCSVNoHeader(t *testing.T) {
	tmpDir := t.TempDir()
	csvPath := filepath.Join(tmpDir, "test_no_header.csv")

	csvContent := `1.0,2.0
3.0,4.0
`
	err := os.WriteFile(csvPath, []byte(csvContent), 0644)
	if err != nil {
		t.Fatalf("Failed to create test CSV: %v", err)
	}

	ds, err := LoadCSV(csvPath, []int{0, 1}, -1, false)
	if err != nil {
		t.Fatalf("LoadCSV failed: %v", err)
	}

	if ds.Size() != 2 {
		t.Errorf("Expected 2 points, got %d", ds.Size())
	}
}

func TestLoadCSVFileNotFound(t *testing.T) {
	_, err := LoadCSV("/nonexistent/path/file.csv", []int{0}, -1, false)
	if err == nil {
		t.Error("Expected error for nonexistent file, got nil")
	}
}

func TestLoadCSVInvalidData(t *testing.T) {
	tmpDir := t.TempDir()
	csvPath := filepath.Join(tmpDir, "invalid.csv")

	// All rows have invalid data
	csvContent := `not_a_number,also_not
abc,def
`
	err := os.WriteFile(csvPath, []byte(csvContent), 0644)
	if err != nil {
		t.Fatalf("Failed to create test CSV: %v", err)
	}

	_, err = LoadCSV(csvPath, []int{0, 1}, -1, false)
	if err == nil {
		t.Error("Expected error for invalid data, got nil")
	}
}
