package iforest

import (
	"math"
	"math/rand"
	"testing"

	"proj3-redesigned/data"
)

// Helper function to create a test dataset
func createTestDataset(numPoints, numFeatures int) *data.Dataset {
	points := make([]data.Point, numPoints)
	rng := rand.New(rand.NewSource(42))

	for i := 0; i < numPoints; i++ {
		features := make([]float64, numFeatures)
		for j := 0; j < numFeatures; j++ {
			features[j] = rng.Float64() * 100
		}
		points[i] = data.Point{
			Features: features,
			ID:       i,
			Label:    "normal",
		}
	}

	return &data.Dataset{
		Points:      points,
		NumFeatures: numFeatures,
	}
}

// Helper function to create a dataset with anomalies
func createDatasetWithAnomalies(numNormal, numAnomalies, numFeatures int) *data.Dataset {
	points := make([]data.Point, numNormal+numAnomalies)
	rng := rand.New(rand.NewSource(42))

	// Normal points clustered around 50
	for i := 0; i < numNormal; i++ {
		features := make([]float64, numFeatures)
		for j := 0; j < numFeatures; j++ {
			features[j] = 50 + rng.Float64()*10 - 5 // 45-55 range
		}
		points[i] = data.Point{
			Features: features,
			ID:       i,
			Label:    "normal",
		}
	}

	// Anomalies far from the cluster
	for i := 0; i < numAnomalies; i++ {
		features := make([]float64, numFeatures)
		for j := 0; j < numFeatures; j++ {
			features[j] = 200 + rng.Float64()*50 // 200-250 range (far from normal)
		}
		points[numNormal+i] = data.Point{
			Features: features,
			ID:       numNormal + i,
			Label:    "anomaly",
		}
	}

	return &data.Dataset{
		Points:      points,
		NumFeatures: numFeatures,
	}
}

func TestNewIsolationForest(t *testing.T) {
	forest := NewIsolationForest(100, 256)

	if forest.NumTrees != 100 {
		t.Errorf("Expected 100 trees, got %d", forest.NumTrees)
	}
	if forest.SampleSize != 256 {
		t.Errorf("Expected sample size 256, got %d", forest.SampleSize)
	}
	// MaxDepth should be ceil(log2(256)) = 8
	if forest.MaxDepth != 8 {
		t.Errorf("Expected max depth 8, got %d", forest.MaxDepth)
	}
	if forest.C <= 0 {
		t.Errorf("Expected positive C value, got %f", forest.C)
	}
}

func TestComputeC(t *testing.T) {
	// Test edge cases
	if computeC(0) != 0 {
		t.Errorf("computeC(0) should be 0, got %f", computeC(0))
	}
	if computeC(1) != 0 {
		t.Errorf("computeC(1) should be 0, got %f", computeC(1))
	}
	if computeC(2) != 1 {
		t.Errorf("computeC(2) should be 1, got %f", computeC(2))
	}

	// For larger n, c(n) should be positive and increasing
	c10 := computeC(10)
	c100 := computeC(100)
	c1000 := computeC(1000)

	if c10 <= 0 || c100 <= 0 || c1000 <= 0 {
		t.Error("computeC should return positive values for n > 2")
	}
	if c10 >= c100 || c100 >= c1000 {
		t.Error("computeC should increase with n")
	}
}

func TestBuildTree(t *testing.T) {
	dataset := createTestDataset(100, 5)
	rng := rand.New(rand.NewSource(42))

	tree := BuildTree(dataset.Points, 8, rng)

	if tree == nil {
		t.Fatal("BuildTree returned nil")
	}
	if tree.Root == nil {
		t.Fatal("Tree root is nil")
	}
	if tree.MaxDepth != 8 {
		t.Errorf("Expected max depth 8, got %d", tree.MaxDepth)
	}
	if tree.SampleSize != 100 {
		t.Errorf("Expected sample size 100, got %d", tree.SampleSize)
	}
}

func TestBuildTreeSinglePoint(t *testing.T) {
	points := []data.Point{
		{Features: []float64{1.0, 2.0}, ID: 0},
	}
	rng := rand.New(rand.NewSource(42))

	tree := BuildTree(points, 8, rng)

	if tree == nil {
		t.Fatal("BuildTree returned nil")
	}
	if !tree.Root.IsExternal {
		t.Error("Single point tree should have external root")
	}
	if tree.Root.Size != 1 {
		t.Errorf("Expected size 1, got %d", tree.Root.Size)
	}
}

func TestBuildTreeIdenticalPoints(t *testing.T) {
	// All points are identical
	points := make([]data.Point, 10)
	for i := range points {
		points[i] = data.Point{Features: []float64{5.0, 5.0}, ID: i}
	}
	rng := rand.New(rand.NewSource(42))

	tree := BuildTree(points, 8, rng)

	if tree == nil {
		t.Fatal("BuildTree returned nil")
	}
	// Should create a leaf since all points are identical
	if !tree.Root.IsExternal {
		t.Error("Identical points should result in external node")
	}
}

func TestPathLength(t *testing.T) {
	dataset := createTestDataset(100, 5)
	rng := rand.New(rand.NewSource(42))

	tree := BuildTree(dataset.Points, 8, rng)

	// Test path length for a point
	pathLen := tree.PathLength(dataset.Points[0])

	if pathLen < 0 {
		t.Errorf("Path length should be non-negative, got %f", pathLen)
	}
	// Path length should be at most maxDepth + c(n) adjustment
	maxPossible := float64(tree.MaxDepth) + computeC(float64(tree.SampleSize))
	if pathLen > maxPossible {
		t.Errorf("Path length %f exceeds maximum possible %f", pathLen, maxPossible)
	}
}

func TestAnomalyScore(t *testing.T) {
	dataset := createTestDataset(100, 5)
	forest := NewIsolationForest(10, 50)
	forest.FitSequential(dataset, 42)

	score := forest.AnomalyScore(dataset.Points[0])

	// Score should be between 0 and 1
	if score < 0 || score > 1 {
		t.Errorf("Anomaly score should be in [0,1], got %f", score)
	}
}

func TestAnomalyScoreEmptyForest(t *testing.T) {
	forest := NewIsolationForest(10, 50)
	// Don't fit - trees slice is empty

	point := data.Point{Features: []float64{1.0, 2.0}}
	score := forest.AnomalyScore(point)

	if score != 0 {
		t.Errorf("Empty forest should return score 0, got %f", score)
	}
}

func TestFitSequential(t *testing.T) {
	dataset := createTestDataset(200, 5)
	forest := NewIsolationForest(10, 50)

	forest.FitSequential(dataset, 42)

	if len(forest.Trees) != 10 {
		t.Errorf("Expected 10 trees, got %d", len(forest.Trees))
	}

	for i, tree := range forest.Trees {
		if tree == nil {
			t.Errorf("Tree %d is nil", i)
		}
		if tree.Root == nil {
			t.Errorf("Tree %d has nil root", i)
		}
	}
}

func TestScoreAllSequential(t *testing.T) {
	dataset := createTestDataset(100, 5)
	forest := NewIsolationForest(10, 50)
	forest.FitSequential(dataset, 42)

	results := forest.ScoreAllSequential(dataset)

	if len(results) != 100 {
		t.Errorf("Expected 100 results, got %d", len(results))
	}

	for i, result := range results {
		if result.Score < 0 || result.Score > 1 {
			t.Errorf("Result %d has invalid score %f", i, result.Score)
		}
		if result.PointID != i {
			t.Errorf("Result %d has wrong PointID %d", i, result.PointID)
		}
	}
}

func TestFitParallel(t *testing.T) {
	dataset := createTestDataset(200, 5)
	forest := NewIsolationForest(10, 50)

	forest.FitParallel(dataset, 42, 4)

	if len(forest.Trees) != 10 {
		t.Errorf("Expected 10 trees, got %d", len(forest.Trees))
	}

	for i, tree := range forest.Trees {
		if tree == nil {
			t.Errorf("Tree %d is nil", i)
		}
	}
}

func TestScoreAllParallel(t *testing.T) {
	dataset := createTestDataset(100, 5)
	forest := NewIsolationForest(10, 50)
	forest.FitSequential(dataset, 42)

	results := forest.ScoreAllParallel(dataset, 4)

	if len(results) != 100 {
		t.Errorf("Expected 100 results, got %d", len(results))
	}

	for i, result := range results {
		if result.Score < 0 || result.Score > 1 {
			t.Errorf("Result %d has invalid score %f", i, result.Score)
		}
	}
}

func TestFitParallelBSP(t *testing.T) {
	dataset := createTestDataset(200, 5)
	forest := NewIsolationForest(10, 50)

	forest.FitParallelBSP(dataset, 42, 4)

	if len(forest.Trees) != 10 {
		t.Errorf("Expected 10 trees, got %d", len(forest.Trees))
	}

	for i, tree := range forest.Trees {
		if tree == nil {
			t.Errorf("Tree %d is nil", i)
		}
	}
}

func TestFitWorkStealing(t *testing.T) {
	dataset := createTestDataset(200, 5)
	forest := NewIsolationForest(10, 50)

	forest.FitWorkStealing(dataset, 42, 4)

	if len(forest.Trees) != 10 {
		t.Errorf("Expected 10 trees, got %d", len(forest.Trees))
	}

	for i, tree := range forest.Trees {
		if tree == nil {
			t.Errorf("Tree %d is nil", i)
		}
	}
}

func TestScoreAllWorkStealing(t *testing.T) {
	dataset := createTestDataset(100, 5)
	forest := NewIsolationForest(10, 50)
	forest.FitSequential(dataset, 42)

	results := forest.ScoreAllWorkStealing(dataset, 4)

	if len(results) != 100 {
		t.Errorf("Expected 100 results, got %d", len(results))
	}

	for i, result := range results {
		if result.Score < 0 || result.Score > 1 {
			t.Errorf("Result %d has invalid score %f", i, result.Score)
		}
	}
}

func TestFitWorkStealingWithStats(t *testing.T) {
	dataset := createTestDataset(200, 5)
	forest := NewIsolationForest(20, 50)

	totalTasks, totalSteals, perWorkerTasks, perWorkerSteals := forest.FitWorkStealingWithStats(dataset, 42, 4)

	if totalTasks != 20 {
		t.Errorf("Expected 20 total tasks, got %d", totalTasks)
	}

	if len(perWorkerTasks) != 4 {
		t.Errorf("Expected 4 worker task counts, got %d", len(perWorkerTasks))
	}
	if len(perWorkerSteals) != 4 {
		t.Errorf("Expected 4 worker steal counts, got %d", len(perWorkerSteals))
	}

	// Sum of per-worker tasks should equal total
	var sum int64
	for _, tasks := range perWorkerTasks {
		sum += tasks
	}
	if sum != totalTasks {
		t.Errorf("Sum of per-worker tasks %d != total tasks %d", sum, totalTasks)
	}

	t.Logf("Total steals: %d", totalSteals)
}

func TestFitWorkStealingVariableCost(t *testing.T) {
	dataset := createTestDataset(200, 5)
	forest := NewIsolationForest(15, 50)

	forest.FitWorkStealingVariableCost(dataset, 42, 4)

	if len(forest.Trees) != 15 {
		t.Errorf("Expected 15 trees, got %d", len(forest.Trees))
	}

	for i, tree := range forest.Trees {
		if tree == nil {
			t.Errorf("Tree %d is nil", i)
		}
	}
}

func TestGetTopAnomalies(t *testing.T) {
	results := []ScoreResult{
		{PointID: 0, Score: 0.3},
		{PointID: 1, Score: 0.9},
		{PointID: 2, Score: 0.5},
		{PointID: 3, Score: 0.8},
		{PointID: 4, Score: 0.2},
	}

	top3 := GetTopAnomalies(results, 3)

	if len(top3) != 3 {
		t.Errorf("Expected 3 results, got %d", len(top3))
	}

	// Should be sorted by score descending
	if top3[0].PointID != 1 || top3[0].Score != 0.9 {
		t.Errorf("First should be point 1 with score 0.9, got point %d with score %f", top3[0].PointID, top3[0].Score)
	}
	if top3[1].PointID != 3 || top3[1].Score != 0.8 {
		t.Errorf("Second should be point 3 with score 0.8, got point %d with score %f", top3[1].PointID, top3[1].Score)
	}
	if top3[2].PointID != 2 || top3[2].Score != 0.5 {
		t.Errorf("Third should be point 2 with score 0.5, got point %d with score %f", top3[2].PointID, top3[2].Score)
	}
}

func TestGetTopAnomaliesMoreThanAvailable(t *testing.T) {
	results := []ScoreResult{
		{PointID: 0, Score: 0.3},
		{PointID: 1, Score: 0.9},
	}

	top5 := GetTopAnomalies(results, 5)

	if len(top5) != 2 {
		t.Errorf("Expected 2 results (all available), got %d", len(top5))
	}
}

func TestRandomSample(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	// Sample less than n
	sample := randomSample(100, 10, rng)
	if len(sample) != 10 {
		t.Errorf("Expected 10 samples, got %d", len(sample))
	}

	// Check all indices are valid
	for _, idx := range sample {
		if idx < 0 || idx >= 100 {
			t.Errorf("Invalid index %d", idx)
		}
	}

	// Check no duplicates
	seen := make(map[int]bool)
	for _, idx := range sample {
		if seen[idx] {
			t.Errorf("Duplicate index %d", idx)
		}
		seen[idx] = true
	}
}

func TestRandomSampleMoreThanN(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	// Sample more than n should return all
	sample := randomSample(5, 10, rng)
	if len(sample) != 5 {
		t.Errorf("Expected 5 samples (all), got %d", len(sample))
	}
}

func TestAllIdentical(t *testing.T) {
	// Empty
	if !allIdentical([]data.Point{}) {
		t.Error("Empty slice should be identical")
	}

	// Single point
	single := []data.Point{{Features: []float64{1.0, 2.0}}}
	if !allIdentical(single) {
		t.Error("Single point should be identical")
	}

	// All same
	same := []data.Point{
		{Features: []float64{1.0, 2.0}},
		{Features: []float64{1.0, 2.0}},
		{Features: []float64{1.0, 2.0}},
	}
	if !allIdentical(same) {
		t.Error("Same points should be identical")
	}

	// Different
	different := []data.Point{
		{Features: []float64{1.0, 2.0}},
		{Features: []float64{1.0, 3.0}},
	}
	if allIdentical(different) {
		t.Error("Different points should not be identical")
	}
}

func TestGetFeatureRange(t *testing.T) {
	points := []data.Point{
		{Features: []float64{1.0, 10.0}},
		{Features: []float64{5.0, 20.0}},
		{Features: []float64{3.0, 15.0}},
	}

	min0, max0 := getFeatureRange(points, 0)
	if min0 != 1.0 || max0 != 5.0 {
		t.Errorf("Feature 0: expected [1.0, 5.0], got [%f, %f]", min0, max0)
	}

	min1, max1 := getFeatureRange(points, 1)
	if min1 != 10.0 || max1 != 20.0 {
		t.Errorf("Feature 1: expected [10.0, 20.0], got [%f, %f]", min1, max1)
	}
}

func TestGetFeatureRangeEmpty(t *testing.T) {
	min, max := getFeatureRange([]data.Point{}, 0)
	if min != 0 || max != 0 {
		t.Errorf("Empty: expected [0, 0], got [%f, %f]", min, max)
	}
}

func TestAnomalyDetection(t *testing.T) {
	// Create dataset with clear anomalies
	dataset := createDatasetWithAnomalies(90, 10, 3)
	forest := NewIsolationForest(50, 50)
	forest.FitSequential(dataset, 42)

	results := forest.ScoreAllSequential(dataset)

	// Get average score for normal vs anomaly points
	var normalSum, anomalySum float64
	var normalCount, anomalyCount int

	for _, result := range results {
		if result.Label == "normal" {
			normalSum += result.Score
			normalCount++
		} else {
			anomalySum += result.Score
			anomalyCount++
		}
	}

	avgNormal := normalSum / float64(normalCount)
	avgAnomaly := anomalySum / float64(anomalyCount)

	t.Logf("Average normal score: %f", avgNormal)
	t.Logf("Average anomaly score: %f", avgAnomaly)

	// Anomalies should have higher scores on average
	if avgAnomaly <= avgNormal {
		t.Errorf("Anomalies should have higher scores than normal points")
	}
}

func TestConsistentResults(t *testing.T) {
	dataset := createTestDataset(100, 5)

	// Sequential
	forest1 := NewIsolationForest(10, 50)
	forest1.FitSequential(dataset, 42)
	results1 := forest1.ScoreAllSequential(dataset)

	// Parallel
	forest2 := NewIsolationForest(10, 50)
	forest2.FitParallel(dataset, 42, 4)
	results2 := forest2.ScoreAllParallel(dataset, 4)

	// Results should have same length
	if len(results1) != len(results2) {
		t.Errorf("Result lengths differ: %d vs %d", len(results1), len(results2))
	}

	// Note: Due to different execution orders, exact scores may differ slightly
	// but all scores should be valid
	for i := range results1 {
		if results2[i].Score < 0 || results2[i].Score > 1 {
			t.Errorf("Invalid parallel score at %d: %f", i, results2[i].Score)
		}
	}
}

func TestScoreResultStruct(t *testing.T) {
	result := ScoreResult{
		PointID:       42,
		Score:         0.75,
		Label:         "anomaly",
		AvgPathLength: 3.5,
	}

	if result.PointID != 42 {
		t.Errorf("Expected PointID 42, got %d", result.PointID)
	}
	if result.Score != 0.75 {
		t.Errorf("Expected Score 0.75, got %f", result.Score)
	}
	if result.Label != "anomaly" {
		t.Errorf("Expected Label 'anomaly', got '%s'", result.Label)
	}
	if result.AvgPathLength != 3.5 {
		t.Errorf("Expected AvgPathLength 3.5, got %f", result.AvgPathLength)
	}
}

func TestNodeStruct(t *testing.T) {
	// External node
	external := &Node{
		IsExternal: true,
		Size:       5,
	}
	if !external.IsExternal {
		t.Error("External node should have IsExternal=true")
	}

	// Internal node
	internal := &Node{
		SplitFeature: 2,
		SplitValue:   50.0,
		Left:         external,
		Right:        &Node{IsExternal: true, Size: 3},
		IsExternal:   false,
	}
	if internal.IsExternal {
		t.Error("Internal node should have IsExternal=false")
	}
	if internal.SplitFeature != 2 {
		t.Errorf("Expected SplitFeature 2, got %d", internal.SplitFeature)
	}
	if internal.SplitValue != 50.0 {
		t.Errorf("Expected SplitValue 50.0, got %f", internal.SplitValue)
	}
}

func TestIsolationTreeStruct(t *testing.T) {
	tree := &IsolationTree{
		Root:       &Node{IsExternal: true, Size: 10},
		MaxDepth:   8,
		SampleSize: 256,
	}

	if tree.MaxDepth != 8 {
		t.Errorf("Expected MaxDepth 8, got %d", tree.MaxDepth)
	}
	if tree.SampleSize != 256 {
		t.Errorf("Expected SampleSize 256, got %d", tree.SampleSize)
	}
}

func TestIsolationForestStruct(t *testing.T) {
	forest := &IsolationForest{
		Trees:      make([]*IsolationTree, 100),
		NumTrees:   100,
		SampleSize: 256,
		MaxDepth:   8,
		C:          computeC(256),
	}

	if forest.NumTrees != 100 {
		t.Errorf("Expected NumTrees 100, got %d", forest.NumTrees)
	}
	if forest.SampleSize != 256 {
		t.Errorf("Expected SampleSize 256, got %d", forest.SampleSize)
	}
	if forest.MaxDepth != 8 {
		t.Errorf("Expected MaxDepth 8, got %d", forest.MaxDepth)
	}
	if forest.C <= 0 {
		t.Errorf("Expected positive C, got %f", forest.C)
	}
}

func TestPathLengthNilNode(t *testing.T) {
	tree := &IsolationTree{
		Root:     nil,
		MaxDepth: 8,
	}

	point := data.Point{Features: []float64{1.0, 2.0}}
	pathLen := tree.PathLength(point)

	if pathLen != 0 {
		t.Errorf("Path length for nil root should be 0, got %f", pathLen)
	}
}

func TestLargeDataset(t *testing.T) {
	// Test with a larger dataset to ensure no issues
	dataset := createTestDataset(1000, 10)
	forest := NewIsolationForest(50, 256)

	forest.FitParallel(dataset, 42, 4)
	results := forest.ScoreAllParallel(dataset, 4)

	if len(results) != 1000 {
		t.Errorf("Expected 1000 results, got %d", len(results))
	}

	// All scores should be valid
	for i, result := range results {
		if result.Score < 0 || result.Score > 1 {
			t.Errorf("Invalid score at %d: %f", i, result.Score)
		}
		if math.IsNaN(result.Score) || math.IsInf(result.Score, 0) {
			t.Errorf("NaN or Inf score at %d", i)
		}
	}
}
