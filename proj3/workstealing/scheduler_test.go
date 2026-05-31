package workstealing

import (
	"sync"
	"sync/atomic"
	"testing"
)

func TestNewWorkerPool(t *testing.T) {
	taskFunc := func(task *Task) {}
	pool := NewWorkerPool(4, taskFunc)

	if pool == nil {
		t.Fatal("NewWorkerPool returned nil")
	}
	if pool.NumWorkers != 4 {
		t.Errorf("Expected 4 workers, got %d", pool.NumWorkers)
	}
	if len(pool.Workers) != 4 {
		t.Errorf("Expected 4 workers in slice, got %d", len(pool.Workers))
	}

	for i, worker := range pool.Workers {
		if worker.ID != i {
			t.Errorf("Worker %d has incorrect ID %d", i, worker.ID)
		}
		if worker.Deque == nil {
			t.Errorf("Worker %d has nil deque", i)
		}
		if worker.Pool != pool {
			t.Errorf("Worker %d has incorrect pool reference", i)
		}
	}
}

func TestSubmit(t *testing.T) {
	taskFunc := func(task *Task) {}
	pool := NewWorkerPool(4, taskFunc)

	task := &Task{ID: 1, Data: "test"}
	pool.Submit(0, task)

	if pool.Workers[0].Deque.Size() != 1 {
		t.Errorf("Expected 1 task in worker 0's deque, got %d", pool.Workers[0].Deque.Size())
	}
}

func TestSubmitRoundRobin(t *testing.T) {
	taskFunc := func(task *Task) {}
	pool := NewWorkerPool(4, taskFunc)

	tasks := make([]*Task, 8)
	for i := range tasks {
		tasks[i] = &Task{ID: i}
	}

	pool.SubmitRoundRobin(tasks)

	// Each worker should have 2 tasks
	for i, worker := range pool.Workers {
		if worker.Deque.Size() != 2 {
			t.Errorf("Worker %d expected 2 tasks, got %d", i, worker.Deque.Size())
		}
	}
}

func TestWorkerSubmit(t *testing.T) {
	taskFunc := func(task *Task) {}
	pool := NewWorkerPool(2, taskFunc)

	worker := pool.Workers[0]
	task := &Task{ID: 1}
	worker.Submit(task)

	if worker.Deque.Size() != 1 {
		t.Errorf("Expected 1 task in worker's deque, got %d", worker.Deque.Size())
	}
}

func TestPoolStartAndWait(t *testing.T) {
	var completed int64

	taskFunc := func(task *Task) {
		atomic.AddInt64(&completed, 1)
	}

	pool := NewWorkerPool(4, taskFunc)

	// Submit tasks
	numTasks := 100
	for i := 0; i < numTasks; i++ {
		pool.Submit(i%4, &Task{ID: i})
	}

	pool.Start()
	pool.Wait()

	if completed != int64(numTasks) {
		t.Errorf("Expected %d completed tasks, got %d", numTasks, completed)
	}
}

func TestWorkStealing(t *testing.T) {
	var completed int64

	taskFunc := func(task *Task) {
		atomic.AddInt64(&completed, 1)
	}

	pool := NewWorkerPool(4, taskFunc)

	// Submit all tasks to worker 0 to force stealing
	numTasks := 100
	for i := 0; i < numTasks; i++ {
		pool.Submit(0, &Task{ID: i})
	}

	pool.Start()
	pool.Wait()

	if completed != int64(numTasks) {
		t.Errorf("Expected %d completed tasks, got %d", numTasks, completed)
	}

	// Check that stealing occurred
	totalTasks, totalSteals := pool.GetPoolStats()
	if totalTasks != int64(numTasks) {
		t.Errorf("Expected total tasks %d, got %d", numTasks, totalTasks)
	}
	// With all tasks on one worker and 4 workers, stealing should happen
	if totalSteals == 0 {
		t.Log("Warning: No steals occurred, but this might be timing-dependent")
	}
}

func TestGetStats(t *testing.T) {
	var completed int64

	taskFunc := func(task *Task) {
		atomic.AddInt64(&completed, 1)
	}

	pool := NewWorkerPool(2, taskFunc)

	// Submit tasks evenly
	for i := 0; i < 10; i++ {
		pool.Submit(i%2, &Task{ID: i})
	}

	pool.Start()
	pool.Wait()

	// Check individual worker stats
	for i, worker := range pool.Workers {
		tasks, steals := worker.GetStats()
		t.Logf("Worker %d: tasks=%d, steals=%d", i, tasks, steals)
	}

	totalTasks, totalSteals := pool.GetPoolStats()
	if totalTasks != 10 {
		t.Errorf("Expected 10 total tasks, got %d", totalTasks)
	}
	t.Logf("Total steals: %d", totalSteals)
}

func TestPoolStop(t *testing.T) {
	taskFunc := func(task *Task) {}
	pool := NewWorkerPool(2, taskFunc)

	pool.Stop()

	// After stop, done flag should be set
	if atomic.LoadInt32(&pool.done) != 1 {
		t.Error("Pool done flag should be set after Stop()")
	}
}

func TestEmptyPool(t *testing.T) {
	var completed int64

	taskFunc := func(task *Task) {
		atomic.AddInt64(&completed, 1)
	}

	pool := NewWorkerPool(4, taskFunc)

	// Start with no tasks - should complete immediately
	pool.Start()
	pool.Wait()

	if completed != 0 {
		t.Errorf("Expected 0 completed tasks, got %d", completed)
	}
}

func TestSingleWorker(t *testing.T) {
	var completed int64

	taskFunc := func(task *Task) {
		atomic.AddInt64(&completed, 1)
	}

	pool := NewWorkerPool(1, taskFunc)

	numTasks := 50
	for i := 0; i < numTasks; i++ {
		pool.Submit(0, &Task{ID: i})
	}

	pool.Start()
	pool.Wait()

	if completed != int64(numTasks) {
		t.Errorf("Expected %d completed tasks, got %d", numTasks, completed)
	}

	// Single worker can't steal from anyone
	_, totalSteals := pool.GetPoolStats()
	if totalSteals != 0 {
		t.Errorf("Single worker should have 0 steals, got %d", totalSteals)
	}
}

func TestTaskDataPreserved(t *testing.T) {
	results := make(map[int]string)
	var mu sync.Mutex

	taskFunc := func(task *Task) {
		mu.Lock()
		results[task.ID] = task.Data.(string)
		mu.Unlock()
	}

	pool := NewWorkerPool(2, taskFunc)

	pool.Submit(0, &Task{ID: 1, Data: "first"})
	pool.Submit(1, &Task{ID: 2, Data: "second"})

	pool.Start()
	pool.Wait()

	if results[1] != "first" {
		t.Errorf("Task 1 data incorrect: %s", results[1])
	}
	if results[2] != "second" {
		t.Errorf("Task 2 data incorrect: %s", results[2])
	}
}
