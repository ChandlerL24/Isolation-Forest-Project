// Core isolation tree structures and tree building logic
package iforest

import (
	"math"
	"math/rand"

	"proj3-redesigned/data"
)

// Node in an isolation tree - internal nodes have splits, external nodes are leaves
type Node struct {
	Left         *Node   // points with feature < splitValue go here
	Right        *Node   // points with feature >= splitValue go here
	SplitFeature int     // which feature we split on
	SplitValue   float64 // the threshold value
	Size         int     // how many points ended up here (only for leaves)
	IsExternal   bool    // true if this is a leaf
}

// IsolationTree is a single tree in the forest
type IsolationTree struct {
	Root       *Node
	MaxDepth   int
	SampleSize int
}

// IsolationForest is a collection of isolation trees
type IsolationForest struct {
	Trees      []*IsolationTree
	NumTrees   int
	SampleSize int
	MaxDepth   int
	C          float64 // normalization constant for scoring
}

// NewIsolationForest creates a forest with the given params
// MaxDepth is set to ceil(log2(sampleSize)) as per the original paper
func NewIsolationForest(numTrees, sampleSize int) *IsolationForest {
	maxDepth := int(math.Ceil(math.Log2(float64(sampleSize))))
	c := computeC(float64(sampleSize))

	return &IsolationForest{
		Trees:      make([]*IsolationTree, 0, numTrees),
		NumTrees:   numTrees,
		SampleSize: sampleSize,
		MaxDepth:   maxDepth,
		C:          c,
	}
}

// computeC returns the average path length of an unsuccessful BST search
// Used to normalize anomaly scores - formula from the original iForest paper
func computeC(n float64) float64 {
	if n <= 1 {
		return 0
	}
	if n == 2 {
		return 1
	}
	// c(n) = 2H(n-1) - (2(n-1)/n), where H(i) ≈ ln(i) + euler's constant
	return 2.0*(math.Log(n-1)+0.5772156649) - (2.0 * (n - 1) / n)
}

// BuildTree constructs a single isolation tree from the given points
func BuildTree(points []data.Point, maxDepth int, rng *rand.Rand) *IsolationTree {
	tree := &IsolationTree{
		MaxDepth:   maxDepth,
		SampleSize: len(points),
	}
	tree.Root = buildNode(points, 0, maxDepth, rng)
	return tree
}

// buildNode recursively builds the tree
// Stops when we hit max depth, have <=1 point, or all points are identical
func buildNode(points []data.Point, currentDepth, maxDepth int, rng *rand.Rand) *Node {
	n := len(points)

	// Base cases - create a leaf
	if currentDepth >= maxDepth || n <= 1 {
		return &Node{IsExternal: true, Size: n}
	}

	if allIdentical(points) {
		return &Node{IsExternal: true, Size: n}
	}

	numFeatures := len(points[0].Features)

	// Pick a random feature to split on
	splitFeature := rng.Intn(numFeatures)
	minVal, maxVal := getFeatureRange(points, splitFeature)

	// If this feature has no variance, try to find one that does
	if minVal == maxVal {
		found := false
		for i := 0; i < numFeatures; i++ {
			splitFeature = (splitFeature + 1) % numFeatures
			minVal, maxVal = getFeatureRange(points, splitFeature)
			if minVal != maxVal {
				found = true
				break
			}
		}
		if !found {
			// All features are constant - make a leaf
			return &Node{IsExternal: true, Size: n}
		}
	}

	// Pick a random split value between min and max
	splitValue := minVal + rng.Float64()*(maxVal-minVal)

	// Partition the points
	var leftPoints, rightPoints []data.Point
	for _, p := range points {
		if p.Features[splitFeature] < splitValue {
			leftPoints = append(leftPoints, p)
		} else {
			rightPoints = append(rightPoints, p)
		}
	}

	// Edge case: all points went to one side (shouldn't happen but just in case)
	if len(leftPoints) == 0 || len(rightPoints) == 0 {
		return &Node{IsExternal: true, Size: n}
	}

	return &Node{
		SplitFeature: splitFeature,
		SplitValue:   splitValue,
		Left:         buildNode(leftPoints, currentDepth+1, maxDepth, rng),
		Right:        buildNode(rightPoints, currentDepth+1, maxDepth, rng),
		IsExternal:   false,
	}
}

// allIdentical checks if all points have the exact same feature values
func allIdentical(points []data.Point) bool {
	if len(points) <= 1 {
		return true
	}
	first := points[0].Features
	for _, p := range points[1:] {
		for i, v := range p.Features {
			if v != first[i] {
				return false
			}
		}
	}
	return true
}

// getFeatureRange finds min and max for a specific feature across all points
func getFeatureRange(points []data.Point, featureIdx int) (float64, float64) {
	if len(points) == 0 {
		return 0, 0
	}
	minVal := points[0].Features[featureIdx]
	maxVal := points[0].Features[featureIdx]
	for _, p := range points[1:] {
		v := p.Features[featureIdx]
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}
	return minVal, maxVal
}

// PathLength returns how many edges a point traverses to reach a leaf
// Includes an adjustment for leaves with multiple points
func (t *IsolationTree) PathLength(point data.Point) float64 {
	return pathLengthRecursive(t.Root, point, 0)
}

// pathLengthRecursive walks the tree and counts depth
func pathLengthRecursive(node *Node, point data.Point, currentDepth int) float64 {
	if node == nil {
		return float64(currentDepth)
	}
	if node.IsExternal {
		// Add adjustment for the remaining points at this leaf
		// (they would need more splits to fully isolate)
		return float64(currentDepth) + computeC(float64(node.Size))
	}
	if point.Features[node.SplitFeature] < node.SplitValue {
		return pathLengthRecursive(node.Left, point, currentDepth+1)
	}
	return pathLengthRecursive(node.Right, point, currentDepth+1)
}

// AnomalyScore computes the anomaly score for a point
// Returns value in [0,1] - higher means more anomalous
func (f *IsolationForest) AnomalyScore(point data.Point) float64 {
	if len(f.Trees) == 0 {
		return 0
	}

	// Average path length across all trees
	var totalPathLength float64
	for _, tree := range f.Trees {
		totalPathLength += tree.PathLength(point)
	}
	avgPathLength := totalPathLength / float64(len(f.Trees))

	// Score formula: s(x,n) = 2^(-E(h(x))/c(n))
	// Short paths -> high score -> anomaly
	if f.C == 0 {
		return 0
	}
	return math.Pow(2, -avgPathLength/f.C)
}

// ScoreResult holds the anomaly score and metadata for a single point
type ScoreResult struct {
	PointID       int
	Score         float64
	Label         string
	AvgPathLength float64
}
