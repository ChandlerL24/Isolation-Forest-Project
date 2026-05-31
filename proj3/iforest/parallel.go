// Channel-based parallel and BSP implementations
package iforest

import (
	"math/rand"
	"proj3-redesigned/data"
	"sync"
)

// TreeTask is what we send to workers - everything needed to build one tree
type TreeTask struct {
	TreeIndex    int
	SamplePoints []data.Point
	Seed         int64
}

// TreeResult is what workers send back
type TreeResult struct {
	TreeIndex int
	Tree      *IsolationTree
}

// FitParallel builds trees using goroutines and channels
// Workers pull tasks from a channel, build trees, send results back
func (f *IsolationForest) FitParallel(dataset *data.Dataset, seed int64, numWorkers int) {
	rng := rand.New(rand.NewSource(seed))
	n := dataset.Size()

	f.Trees = make([]*IsolationTree, f.NumTrees)

	// Buffered channels so we don't block
	taskChan := make(chan TreeTask, f.NumTrees)
	resultChan := make(chan TreeResult, f.NumTrees)

	// Spin up workers
	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			// Keep grabbing tasks until channel closes
			for task := range taskChan {
				treeRng := rand.New(rand.NewSource(task.Seed))
				tree := BuildTree(task.SamplePoints, f.MaxDepth, treeRng)
				resultChan <- TreeResult{TreeIndex: task.TreeIndex, Tree: tree}
			}
		}()
	}

	// Feed tasks to workers
	go func() {
		for i := 0; i < f.NumTrees; i++ {
			sampleIndices := randomSample(n, f.SampleSize, rng)
			samplePoints := make([]data.Point, f.SampleSize)
			for j, idx := range sampleIndices {
				samplePoints[j] = dataset.Points[idx]
			}
			taskChan <- TreeTask{
				TreeIndex:    i,
				SamplePoints: samplePoints,
				Seed:         rng.Int63(),
			}
		}
		close(taskChan)
	}()

	// Close result channel when all workers are done
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Collect results
	for result := range resultChan {
		f.Trees[result.TreeIndex] = result.Tree
	}
}

// ScoreTask is a batch of points to score
type ScoreTask struct {
	StartIdx int
	EndIdx   int
	Points   []data.Point
}

// ScoreResultBatch holds results for a batch
type ScoreResultBatch struct {
	StartIdx int
	Results  []ScoreResult
}

// ScoreAllParallel scores points in parallel batches
func (f *IsolationForest) ScoreAllParallel(dataset *data.Dataset, numWorkers int) []ScoreResult {
	n := dataset.Size()
	results := make([]ScoreResult, n)

	// Split points into batches, one per worker
	batchSize := (n + numWorkers - 1) / numWorkers

	taskChan := make(chan ScoreTask, numWorkers)
	resultChan := make(chan ScoreResultBatch, numWorkers)

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for task := range taskChan {
				batchResults := make([]ScoreResult, len(task.Points))
				for i, point := range task.Points {
					// Compute path length across all trees
					var totalPathLength float64
					for _, tree := range f.Trees {
						totalPathLength += tree.PathLength(point)
					}
					avgPathLength := totalPathLength / float64(len(f.Trees))

					batchResults[i] = ScoreResult{
						PointID:       point.ID,
						Score:         f.AnomalyScore(point),
						Label:         point.Label,
						AvgPathLength: avgPathLength,
					}
				}
				resultChan <- ScoreResultBatch{StartIdx: task.StartIdx, Results: batchResults}
			}
		}()
	}

	// Send batches
	go func() {
		for i := 0; i < n; i += batchSize {
			end := i + batchSize
			if end > n {
				end = n
			}
			taskChan <- ScoreTask{
				StartIdx: i,
				EndIdx:   end,
				Points:   dataset.Points[i:end],
			}
		}
		close(taskChan)
	}()

	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Collect and place results in correct positions
	for batch := range resultChan {
		copy(results[batch.StartIdx:], batch.Results)
	}

	return results
}

// FitParallelBSP uses Bulk Synchronous Parallel pattern
// Processes trees in supersteps with barriers between each batch
func (f *IsolationForest) FitParallelBSP(dataset *data.Dataset, seed int64, numWorkers int) {
	rng := rand.New(rand.NewSource(seed))
	n := dataset.Size()

	f.Trees = make([]*IsolationTree, f.NumTrees)

	// Prepare all tasks upfront
	tasks := make([]TreeTask, f.NumTrees)
	for i := 0; i < f.NumTrees; i++ {
		sampleIndices := randomSample(n, f.SampleSize, rng)
		samplePoints := make([]data.Point, f.SampleSize)
		for j, idx := range sampleIndices {
			samplePoints[j] = dataset.Points[idx]
		}
		tasks[i] = TreeTask{
			TreeIndex:    i,
			SamplePoints: samplePoints,
			Seed:         rng.Int63(),
		}
	}

	// Process in supersteps - each superstep handles numWorkers trees
	treesPerSuperstep := numWorkers
	for superstepStart := 0; superstepStart < f.NumTrees; superstepStart += treesPerSuperstep {
		superstepEnd := superstepStart + treesPerSuperstep
		if superstepEnd > f.NumTrees {
			superstepEnd = f.NumTrees
		}

		var wg sync.WaitGroup
		var mu sync.Mutex
		cond := sync.NewCond(&mu)
		ready := 0
		numInSuperstep := superstepEnd - superstepStart

		for i := superstepStart; i < superstepEnd; i++ {
			wg.Add(1)
			go func(task TreeTask) {
				defer wg.Done()

				// Barrier entry - wait for all goroutines in this superstep
				mu.Lock()
				ready++
				if ready == numInSuperstep {
					// Last one in, wake everyone up
					cond.Broadcast()
				} else {
					// Wait for others
					for ready < numInSuperstep {
						cond.Wait()
					}
				}
				mu.Unlock()

				// Now everyone proceeds together
				treeRng := rand.New(rand.NewSource(task.Seed))
				tree := BuildTree(task.SamplePoints, f.MaxDepth, treeRng)

				// Each goroutine writes to a different index, so no lock needed
				f.Trees[task.TreeIndex] = tree
			}(tasks[i])
		}

		// Wait for this superstep to finish before starting next
		wg.Wait()
	}
}
