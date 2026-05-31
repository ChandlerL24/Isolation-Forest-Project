package workstealing

import (
	"sync"
	"testing"
)

func TestNewDeque(t *testing.T) {
	d := NewDeque()
	if d == nil {
		t.Fatal("NewDeque returned nil")
	}
	if !d.IsEmpty() {
		t.Error("New deque should be empty")
	}
	if d.Size() != 0 {
		t.Errorf("New deque size should be 0, got %d", d.Size())
	}
}

func TestPushPopBottom(t *testing.T) {
	d := NewDeque()

	task1 := &Task{ID: 1, Data: "task1"}
	task2 := &Task{ID: 2, Data: "task2"}
	task3 := &Task{ID: 3, Data: "task3"}

	d.PushBottom(task1)
	d.PushBottom(task2)
	d.PushBottom(task3)

	if d.Size() != 3 {
		t.Errorf("Expected size 3, got %d", d.Size())
	}

	// Pop should return in LIFO order (last pushed first)
	popped := d.PopBottom()
	if popped == nil || popped.ID != 3 {
		t.Errorf("Expected task 3, got %v", popped)
	}

	popped = d.PopBottom()
	if popped == nil || popped.ID != 2 {
		t.Errorf("Expected task 2, got %v", popped)
	}

	popped = d.PopBottom()
	if popped == nil || popped.ID != 1 {
		t.Errorf("Expected task 1, got %v", popped)
	}

	// Should be empty now
	popped = d.PopBottom()
	if popped != nil {
		t.Errorf("Expected nil from empty deque, got %v", popped)
	}
}

func TestPopBottomEmpty(t *testing.T) {
	d := NewDeque()
	task := d.PopBottom()
	if task != nil {
		t.Errorf("PopBottom on empty deque should return nil, got %v", task)
	}
}

func TestSteal(t *testing.T) {
	d := NewDeque()

	task1 := &Task{ID: 1, Data: "task1"}
	task2 := &Task{ID: 2, Data: "task2"}
	task3 := &Task{ID: 3, Data: "task3"}

	d.PushBottom(task1)
	d.PushBottom(task2)
	d.PushBottom(task3)

	// Steal should return from the top (oldest first - FIFO for thieves)
	stolen := d.Steal()
	if stolen == nil || stolen.ID != 1 {
		t.Errorf("Expected to steal task 1, got %v", stolen)
	}

	stolen = d.Steal()
	if stolen == nil || stolen.ID != 2 {
		t.Errorf("Expected to steal task 2, got %v", stolen)
	}
}

func TestStealEmpty(t *testing.T) {
	d := NewDeque()
	task := d.Steal()
	if task != nil {
		t.Errorf("Steal on empty deque should return nil, got %v", task)
	}
}

func TestMixedPushPopSteal(t *testing.T) {
	d := NewDeque()

	// Push 3 tasks
	d.PushBottom(&Task{ID: 1})
	d.PushBottom(&Task{ID: 2})
	d.PushBottom(&Task{ID: 3})

	// Steal one (should get task 1)
	stolen := d.Steal()
	if stolen == nil || stolen.ID != 1 {
		t.Errorf("Expected to steal task 1, got %v", stolen)
	}

	// Pop one (should get task 3)
	popped := d.PopBottom()
	if popped == nil || popped.ID != 3 {
		t.Errorf("Expected to pop task 3, got %v", popped)
	}

	// Only task 2 should remain
	if d.Size() != 1 {
		t.Errorf("Expected size 1, got %d", d.Size())
	}
}

func TestIsEmpty(t *testing.T) {
	d := NewDeque()

	if !d.IsEmpty() {
		t.Error("New deque should be empty")
	}

	d.PushBottom(&Task{ID: 1})
	if d.IsEmpty() {
		t.Error("Deque with one task should not be empty")
	}

	d.PopBottom()
	if !d.IsEmpty() {
		t.Error("Deque after popping all tasks should be empty")
	}
}

func TestConcurrentPushPop(t *testing.T) {
	d := NewDeque()
	numTasks := 100

	var wg sync.WaitGroup

	// Push tasks concurrently
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < numTasks; i++ {
			d.PushBottom(&Task{ID: i})
		}
	}()

	wg.Wait()

	// Verify all tasks were pushed
	if d.Size() != int64(numTasks) {
		t.Errorf("Expected size %d, got %d", numTasks, d.Size())
	}

	// Pop all tasks
	count := 0
	for {
		task := d.PopBottom()
		if task == nil {
			break
		}
		count++
	}

	if count != numTasks {
		t.Errorf("Expected to pop %d tasks, got %d", numTasks, count)
	}
}

func TestConcurrentSteal(t *testing.T) {
	d := NewDeque()
	numTasks := 100

	// Push all tasks first
	for i := 0; i < numTasks; i++ {
		d.PushBottom(&Task{ID: i})
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	stolenCount := 0

	// Multiple thieves stealing concurrently
	numThieves := 4
	for i := 0; i < numThieves; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				task := d.Steal()
				if task == nil {
					return
				}
				mu.Lock()
				stolenCount++
				mu.Unlock()
			}
		}()
	}

	wg.Wait()

	if stolenCount != numTasks {
		t.Errorf("Expected to steal %d tasks, got %d", numTasks, stolenCount)
	}
}

func TestTaskStruct(t *testing.T) {
	task := &Task{
		ID:   42,
		Data: "test data",
	}

	if task.ID != 42 {
		t.Errorf("Expected ID 42, got %d", task.ID)
	}
	if task.Data.(string) != "test data" {
		t.Errorf("Expected data 'test data', got %v", task.Data)
	}
}
