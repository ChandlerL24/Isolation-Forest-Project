// Sequential baseline implementation - single threaded, no parallelism
package iforest

import (
	"math/rand"
	"sort"

	"proj3-redesigned/data"
)

// FitSequential builds all trees one at a time
func (f *IsolationForest) FitSequential(dataset *data.Dataset, seed int64) {
	rng := rand.New(rand.NewSource(seed))
	n := dataset.Size()

	f.Trees = make([]*IsolationTree, f.NumTrees)

	for i := 0; i < f.NumTrees; i++ {
		// Grab a random subsample for this tree
		sampleIndices := randomSample(n, f.SampleSize, rng)
		samplePoints := make([]data.Point, f.SampleSize)
		for j, idx := range sampleIndices {
			samplePoints[j] = dataset.Points[idx]
		}

		// Each tree gets its own RNG seeded from the main one
		// This keeps things reproducible
		treeSeed := rng.Int63()
		treeRng := rand.New(rand.NewSource(treeSeed))
		f.Trees[i] = BuildTree(samplePoints, f.MaxDepth, treeRng)
	}
}

// ScoreAllSequential computes anomaly scores for every point, one at a time
func (f *IsolationForest) ScoreAllSequential(dataset *data.Dataset) []ScoreResult {
	results := make([]ScoreResult, dataset.Size())

	for i, point := range dataset.Points {
		// Get average path length across all trees
		var totalPathLength float64
		for _, tree := range f.Trees {
			totalPathLength += tree.PathLength(point)
		}
		avgPathLength := totalPathLength / float64(len(f.Trees))

		results[i] = ScoreResult{
			PointID:       point.ID,
			Score:         f.AnomalyScore(point),
			Label:         point.Label,
			AvgPathLength: avgPathLength,
		}
	}

	return results
}

// randomSample uses Fisher-Yates to get sampleSize random indices without replacement
func randomSample(n, sampleSize int, rng *rand.Rand) []int {
	// If we want more samples than we have, just return all indices
	if sampleSize >= n {
		indices := make([]int, n)
		for i := range indices {
			indices[i] = i
		}
		return indices
	}

	// Fisher-Yates partial shuffle
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}

	for i := 0; i < sampleSize; i++ {
		j := i + rng.Intn(n-i)
		indices[i], indices[j] = indices[j], indices[i]
	}

	return indices[:sampleSize]
}

// GetTopAnomalies returns the top N points sorted by anomaly score (highest first)
func GetTopAnomalies(results []ScoreResult, topN int) []ScoreResult {
	// Make a copy so we don't mess with the original order
	sorted := make([]ScoreResult, len(results))
	copy(sorted, results)

	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Score > sorted[j].Score
	})

	if topN > len(sorted) {
		topN = len(sorted)
	}

	return sorted[:topN]
}
