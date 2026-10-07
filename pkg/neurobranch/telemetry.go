package neurobranch

import (
	"sync"
	"time"
)

// TelemetryEvent captures diagnostic metadata for ambiguous, fallback, or multi-intent routing queries.
type TelemetryEvent struct {
	InputText      string  `json:"input_text"`
	PredictedLabel string  `json:"predicted_label"`
	SecondaryLabel string  `json:"secondary_label,omitempty"`
	Confidence     float64 `json:"confidence"`
	Entropy        float64 `json:"entropy"`
	IsAmbiguous    bool    `json:"is_ambiguous"`
	IsPipeline     bool    `json:"is_pipeline"`
	IsFallback     bool    `json:"is_fallback"`
	TimestampNano  int64   `json:"timestamp_nano"`
}

// TelemetryRingBuffer is a thread-safe, bounded ring buffer for asynchronous telemetry feedback.
type TelemetryRingBuffer struct {
	mu       sync.Mutex
	capacity int
	events   []TelemetryEvent
	head     int
	tail     int
	count    int
}

// NewTelemetryRingBuffer creates a bounded ring buffer with the specified event capacity.
func NewTelemetryRingBuffer(capacity int) *TelemetryRingBuffer {
	if capacity <= 0 {
		capacity = 1024
	}
	return &TelemetryRingBuffer{
		capacity: capacity,
		events:   make([]TelemetryEvent, capacity),
	}
}

// Push records an event into the ring buffer, overwriting the oldest entry if capacity is reached.
func (rb *TelemetryRingBuffer) Push(event TelemetryEvent) {
	rb.mu.Lock()
	defer rb.mu.Unlock()

	event.TimestampNano = time.Now().UnixNano()
	rb.events[rb.head] = event
	rb.head = (rb.head + 1) % rb.capacity

	if rb.count < rb.capacity {
		rb.count++
	} else {
		rb.tail = (rb.tail + 1) % rb.capacity
	}
}

// Drain extracts and clears all recorded events in FIFO order for retraining and drift analysis.
func (rb *TelemetryRingBuffer) Drain() []TelemetryEvent {
	rb.mu.Lock()
	defer rb.mu.Unlock()

	if rb.count == 0 {
		return nil
	}

	result := make([]TelemetryEvent, rb.count)
	for i := 0; i < rb.count; i++ {
		idx := (rb.tail + i) % rb.capacity
		result[i] = rb.events[idx]
	}

	rb.head = 0
	rb.tail = 0
	rb.count = 0

	return result
}

// Count returns the number of currently buffered events.
func (rb *TelemetryRingBuffer) Count() int {
	rb.mu.Lock()
	defer rb.mu.Unlock()
	return rb.count
}
