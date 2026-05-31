// Work-stealing implementation using lock-free deques
package iforest

import (
	"math/rand"
	"proj3-redesigned/data"
	"proj3-redesigned/workstealing"
	"sync"
	"sync/atomic"
)

// TreeBuildTask has everything needed to build one tree
type TreeBuildTask struct {
	TreeIndex    int
	SamplePoints []data.Point
	Seed         int64
	MaxDepth     int
}

// FitWorkStealing builds trees using work-stealing scheduler
// Intentionally creates imbalanced workload to show off stealing
func (f *IsolationForest) FitWorkStealing(dataset *data.Dataset, seed int64, numWorkers int) {
	rng := rand.New(rand.NewSource(seed))
	n := dataset.Size()

	f.Trees = make([]*IsolationTree, f.NumTrees)

	// Create all the tasks
	tasks := make([]*workstealing.Task, f.NumTrees)
	for i := 0; i < f.NumTrees; i++ {
		sampleIndices := randomSample(n, f.SampleSize, rng)
		samplePoints := make([]data.Point, f.SampleSize)
		for j, idx := range sampleIndices {
			samplePoints[j] = dataset.Points[idx]
		}

		tasks[i] = &workstealing.Task{
			ID: i,
			Data: TreeBuildTask{
				TreeIndex:    i,
				SamplePoints: samplePoints,
				Seed:         rng.Int63(),
				MaxDepth:     f.MaxDepth,
			},
		}
	}

	var mu sync.Mutex

	// Set up the worker pool with our tree-building function
	pool := workstealing.NewWorkerPool(numWorkers, func(task *workstealing.Task) {
		buildTask := task.Data.(TreeBuildTask)
		treeRng := rand.New(rand.NewSource(buildTask.Seed))
		tree := BuildTree(buildTask.SamplePoints, buildTask.MaxDepth, treeRng)

		mu.Lock()
		f.Trees[buildTask.TreeIndex] = tree
		mu.Unlock()
	})

	// Distribute tasks unevenly - first worker gets half
	// This creates imbalance that work-stealing will fix
	for i, task := range tasks {
		workerIdx := 0
		if i >= f.NumTrees/2 {
			workerIdx = i % numWorkers
		}
		pool.Submit(workerIdx, task)
	}

	pool.Start()
	pool.Wait()
}

// ScorePointTask is for scoring a single point
type ScorePointTask struct {
	PointIndex int
	Point      data.Point
}

// ScoreAllWorkStealing scores all points using work-stealing
func (f *IsolationForest) ScoreAllWorkStealing(dataset *data.Dataset, numWorkers int) []ScoreResult {
	n := dataset.Size()
	results := make([]ScoreResult, n)

	// One task per point
	tasks := make([]*workstealing.Task, n)
	for i, point := range dataset.Points {
		tasks[i] = &workstealing.Task{
			ID: i,
			Data: ScorePointTask{
				PointIndex: i,
				Point:      point,
			},
		}
	}

	pool := workstealing.NewWorkerPool(numWorkers, func(task *workstealing.Task) {
		scoreTask := task.Data.(ScorePointTask)
		point := scoreTask.Point

		var totalPathLength float64
		for _, tree := range f.Trees {
			totalPathLength += tree.PathLength(point)
		}
		avgPathLength := totalPathLength / float64(len(f.Trees))

		// Each task writes to its own index, no lock needed
		results[scoreTask.PointIndex] = ScoreResult{
			PointID:       point.ID,
			Score:         f.AnomalyScore(point),
			Label:         point.Label,
			AvgPathLength: avgPathLength,
		}
	})

	// Same uneven distribution
	for i, task := range tasks {
		workerIdx := 0
		if i >= n/2 {
			workerIdx = i % numWorkers
		}
		pool.Submit(workerIdx, task)
	}

	pool.Start()
	pool.Wait()

	return results
}

// FitWorkStealingWithStats is like FitWorkStealing but returns stealing statistics
// Useful for seeing how much work got redistributed
func (f *IsolationForest) FitWorkStealingWithStats(dataset *data.Dataset, seed int64, numWorkers int) (totalTasks, totalSteals int64, perWorkerTasks, perWorkerSteals []int64) {
	rng := rand.New(rand.NewSource(seed))
	n := dataset.Size()

	f.Trees = make([]*IsolationTree, f.NumTrees)

	tasks := make([]*workstealing.Task, f.NumTrees)
	for i := 0; i < f.NumTrees; i++ {
		sampleIndices := randomSample(n, f.SampleSize, rng)
		samplePoints := make([]data.Point, f.SampleSize)
		for j, idx := range sampleIndices {
			samplePoints[j] = dataset.Points[idx]
		}

		tasks[i] = &workstealing.Task{
			ID: i,
			Data: TreeBuildTask{
				TreeIndex:    i,
				SamplePoints: samplePoints,
				Seed:         rng.Int63(),
				MaxDepth:     f.MaxDepth,
			},
		}
	}

	var mu sync.Mutex

	pool := workstealing.NewWorkerPool(numWorkers, func(task *workstealing.Task) {
		buildTask := task.Data.(TreeBuildTask)
		treeRng := rand.New(rand.NewSource(buildTask.Seed))
		tree := BuildTree(buildTask.SamplePoints, buildTask.MaxDepth, treeRng)

		mu.Lock()
		f.Trees[buildTask.TreeIndex] = tree
		mu.Unlock()
	})

	// Uneven distribution
	for i, task := range tasks {
		workerIdx := 0
		if i >= f.NumTrees/2 {
			workerIdx = i % numWorkers
		}
		pool.Submit(workerIdx, task)
	}

	pool.Start()
	pool.Wait()

	// Grab the stats
	totalTasks, totalSteals = pool.GetPoolStats()
	perWorkerTasks = make([]int64, numWorkers)
	perWorkerSteals = make([]int64, numWorkers)
	for i, worker := range pool.Workers {
		perWorkerTasks[i], perWorkerSteals[i] = worker.GetStats()
	}

	return
}

// FitWorkStealingVariableCost creates tasks with different costs
// Some trees get more samples (more work), really shows off work-stealing benefits
func (f *IsolationForest) FitWorkStealingVariableCost(dataset *data.Dataset, seed int64, numWorkers int) {
	rng := rand.New(rand.NewSource(seed))
	n := dataset.Size()

	f.Trees = make([]*IsolationTree, f.NumTrees)

	tasks := make([]*workstealing.Task, f.NumTrees)
	for i := 0; i < f.NumTrees; i++ {
		// Vary the sample size to create load imbalance
		variableSampleSize := f.SampleSize
		if i%3 == 0 {
			variableSampleSize = f.SampleSize * 2 // Double work for every 3rd tree
		} else if i%5 == 0 {
			variableSampleSize = f.SampleSize / 2 // Half work for every 5th
		}
		// Clamp to reasonable bounds
		if variableSampleSize > n {
			variableSampleSize = n
		}
		if variableSampleSize < 10 {
			variableSampleSize = 10
		}

		sampleIndices := randomSample(n, variableSampleSize, rng)
		samplePoints := make([]data.Point, variableSampleSize)
		for j, idx := range sampleIndices {
			samplePoints[j] = dataset.Points[idx]
		}

		tasks[i] = &workstealing.Task{
			ID: i,
			Data: TreeBuildTask{
				TreeIndex:    i,
				SamplePoints: samplePoints,
				Seed:         rng.Int63(),
				MaxDepth:     f.MaxDepth,
			},
		}
	}

	var completed int64

	pool := workstealing.NewWorkerPool(numWorkers, func(task *workstealing.Task) {
		buildTask := task.Data.(TreeBuildTask)
		treeRng := rand.New(rand.NewSource(buildTask.Seed))
		tree := BuildTree(buildTask.SamplePoints, buildTask.MaxDepth, treeRng)

		f.Trees[buildTask.TreeIndex] = tree
		atomic.AddInt64(&completed, 1)
	})

	// Put all the heavy tasks (i%3==0) on the first worker
	// This maximizes stealing opportunities
	for i, task := range tasks {
		workerIdx := 0
		if i%3 != 0 {
			workerIdx = i % numWorkers
		}
		pool.Submit(workerIdx, task)
	}

	pool.Start()
	pool.Wait()
}
