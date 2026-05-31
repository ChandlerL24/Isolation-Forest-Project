// Lock-free work-stealing deque using CAS operations
// Owner pushes/pops from bottom, thieves steal from top
package workstealing

import (
	"sync/atomic"
	"unsafe"
)

// Task is a unit of work
type Task struct {
	ID   int
	Data interface{}
}

// node is a doubly-linked list node for the deque
type node struct {
	task *Task
	next unsafe.Pointer // points toward top
	prev unsafe.Pointer // points toward bottom
}

// Deque is a lock-free double-ended queue
// The key insight: owner and thieves work on opposite ends, so contention is rare
type Deque struct {
	top    unsafe.Pointer // thieves steal from here (FIFO for them)
	bottom unsafe.Pointer // owner pushes/pops here (LIFO for owner)
	size   int64          // approximate, used for quick empty checks
}

// NewDeque creates an empty deque with sentinel nodes
func NewDeque() *Deque {
	sentinel := &node{}
	return &Deque{
		top:    unsafe.Pointer(sentinel),
		bottom: unsafe.Pointer(sentinel),
		size:   0,
	}
}

// PushBottom adds a task to the bottom (owner only)
// Uses CAS loop in case of contention
func (d *Deque) PushBottom(task *Task) {
	newNode := &node{task: task}

	for {
		bottom := (*node)(atomic.LoadPointer(&d.bottom))
		newNode.next = unsafe.Pointer(bottom)

		// Try to swing bottom to point to new node
		if atomic.CompareAndSwapPointer(&d.bottom, unsafe.Pointer(bottom), unsafe.Pointer(newNode)) {
			// Success - link the old bottom back to us
			atomic.StorePointer(&bottom.prev, unsafe.Pointer(newNode))
			atomic.AddInt64(&d.size, 1)
			return
		}
		// CAS failed, someone else modified bottom, try again
	}
}

// PopBottom removes a task from the bottom (owner only)
// Returns nil if empty
func (d *Deque) PopBottom() *Task {
	for {
		bottom := (*node)(atomic.LoadPointer(&d.bottom))
		top := (*node)(atomic.LoadPointer(&d.top))

		// Empty check
		if bottom == top {
			return nil
		}

		task := bottom.task
		if task == nil {
			return nil
		}

		// Get the next node (toward top)
		next := (*node)(atomic.LoadPointer(&bottom.next))
		if next == nil {
			return nil
		}

		// Try to move bottom up
		if atomic.CompareAndSwapPointer(&d.bottom, unsafe.Pointer(bottom), unsafe.Pointer(next)) {
			atomic.AddInt64(&d.size, -1)
			return task
		}
		// CAS failed, retry
	}
}

// Steal takes a task from the top (thieves only)
// Returns nil if empty or if we lost the race
func (d *Deque) Steal() *Task {
	for {
		top := (*node)(atomic.LoadPointer(&d.top))
		bottom := (*node)(atomic.LoadPointer(&d.bottom))

		// Empty check
		if top == bottom {
			return nil
		}

		// Get the node before top (toward bottom)
		prev := (*node)(atomic.LoadPointer(&top.prev))
		if prev == nil {
			return nil
		}

		task := prev.task
		if task == nil {
			return nil
		}

		// Try to move top down (toward bottom)
		if atomic.CompareAndSwapPointer(&d.top, unsafe.Pointer(top), unsafe.Pointer(prev)) {
			atomic.AddInt64(&d.size, -1)
			return task
		}
		// Lost the race, try again
	}
}

// IsEmpty returns true if the deque looks empty
// Note: this is approximate due to concurrent access
func (d *Deque) IsEmpty() bool {
	return atomic.LoadInt64(&d.size) <= 0
}

// Size returns approximate number of tasks
func (d *Deque) Size() int64 {
	size := atomic.LoadInt64(&d.size)
	if size < 0 {
		return 0
	}
	return size
}
