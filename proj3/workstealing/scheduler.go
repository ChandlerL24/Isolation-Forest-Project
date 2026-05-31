// Work-stealing scheduler - manages workers and coordinates task execution
package workstealing

import (
	"math/rand"
	"sync"
	"sync/atomic"
	"time"
)

// Worker is a single worker thread with its own deque
type Worker struct {
	ID        int
	Deque     *Deque
	Pool      *WorkerPool
	rng       *rand.Rand // for random victim selection
	tasksDone int64      // stats
	steals    int64      // stats
}

// WorkerPool manages all the workers
type WorkerPool struct {
	Workers    []*Worker
	NumWorkers int
	done       int32 // atomic flag for termination
	wg         sync.WaitGroup
	taskFunc   func(*Task) // what to do with each task
}

// NewWorkerPool creates a pool with the given number of workers
func NewWorkerPool(numWorkers int, taskFunc func(*Task)) *WorkerPool {
	pool := &WorkerPool{
		Workers:    make([]*Worker, numWorkers),
		NumWorkers: numWorkers,
		taskFunc:   taskFunc,
	}

	for i := 0; i < numWorkers; i++ {
		pool.Workers[i] = &Worker{
			ID:    i,
			Deque: NewDeque(),
			Pool:  pool,
			// Each worker gets its own RNG to avoid contention
			rng: rand.New(rand.NewSource(time.Now().UnixNano() + int64(i))),
		}
	}

	return pool
}

// Submit adds a task to a specific worker's deque
func (p *WorkerPool) Submit(workerID int, task *Task) {
	p.Workers[workerID%p.NumWorkers].Deque.PushBottom(task)
}

// SubmitRoundRobin distributes tasks evenly across workers
func (p *WorkerPool) SubmitRoundRobin(tasks []*Task) {
	for i, task := range tasks {
		p.Workers[i%p.NumWorkers].Deque.PushBottom(task)
	}
}

// Submit adds a task to this worker's deque
func (w *Worker) Submit(task *Task) {
	w.Deque.PushBottom(task)
}

// Start kicks off all workers
func (p *WorkerPool) Start() {
	atomic.StoreInt32(&p.done, 0)

	for _, worker := range p.Workers {
		p.wg.Add(1)
		go worker.run()
	}
}

// Wait blocks until all workers finish
func (p *WorkerPool) Wait() {
	p.wg.Wait()
}

// Stop signals workers to terminate
func (p *WorkerPool) Stop() {
	atomic.StoreInt32(&p.done, 1)
}

// run is the main worker loop
func (w *Worker) run() {
	defer w.Pool.wg.Done()

	backoffCount := 0
	maxBackoff := 10

	for {
		// Check if we should stop
		if atomic.LoadInt32(&w.Pool.done) == 1 {
			// Drain any remaining work before exiting
			for {
				task := w.Deque.PopBottom()
				if task == nil {
					break
				}
				w.Pool.taskFunc(task)
				atomic.AddInt64(&w.tasksDone, 1)
			}
			return
		}

		// Try to get work from our own deque first
		task := w.Deque.PopBottom()
		if task != nil {
			w.Pool.taskFunc(task)
			atomic.AddInt64(&w.tasksDone, 1)
			backoffCount = 0
			continue
		}

		// Our deque is empty, try stealing from someone else
		stolen := w.trySteal()
		if stolen {
			backoffCount = 0
			continue
		}

		// No work anywhere - check if we're done
		if w.allEmpty() {
			atomic.StoreInt32(&w.Pool.done, 1)
			return
		}

		// Back off a bit before trying again
		// Exponential backoff up to a limit
		backoffCount++
		if backoffCount > maxBackoff {
			backoffCount = maxBackoff
		}
		time.Sleep(time.Duration(backoffCount) * time.Microsecond)
	}
}

// trySteal attempts to steal work from another worker
// Picks a random victim and cycles through all if needed
func (w *Worker) trySteal() bool {
	numWorkers := w.Pool.NumWorkers
	if numWorkers <= 1 {
		return false
	}

	// Start at a random worker to avoid everyone hitting the same victim
	startIdx := w.rng.Intn(numWorkers)

	for i := 0; i < numWorkers; i++ {
		victimIdx := (startIdx + i) % numWorkers
		if victimIdx == w.ID {
			continue // don't steal from yourself
		}

		victim := w.Pool.Workers[victimIdx]
		task := victim.Deque.Steal()
		if task != nil {
			w.Pool.taskFunc(task)
			atomic.AddInt64(&w.tasksDone, 1)
			atomic.AddInt64(&w.steals, 1)
			return true
		}
	}

	return false
}

// allEmpty checks if every worker's deque is empty
func (w *Worker) allEmpty() bool {
	for _, worker := range w.Pool.Workers {
		if !worker.Deque.IsEmpty() {
			return false
		}
	}
	return true
}

// GetStats returns how many tasks this worker completed and how many it stole
func (w *Worker) GetStats() (tasksDone, steals int64) {
	return atomic.LoadInt64(&w.tasksDone), atomic.LoadInt64(&w.steals)
}

// GetPoolStats returns aggregate stats across all workers
func (p *WorkerPool) GetPoolStats() (totalTasks, totalSteals int64) {
	for _, worker := range p.Workers {
		tasks, steals := worker.GetStats()
		totalTasks += tasks
		totalSteals += steals
	}
	return
}
