package neurobranch

import (
	"fmt"
	"sync"
	"sync/atomic"
)

// Router coordinates in-memory inference routing with atomic hot-swap, 3-tier safety, and telemetry feedback.
type Router struct {
	mu        sync.RWMutex
	model     atomic.Pointer[InferenceModel]
	threshold float64
	policy    DispatchPolicy
	telemetry *TelemetryRingBuffer
	samples   []DataSample
}

// AI is the primary domain artificial intelligence engine for intelligent control flow branching.
// It directly executes in-memory neural inference to replace brittle static logic.
type AI = Router

// NewAI loads a binary model file into memory and constructs an active AI engine.
var NewAI = NewRouter

// NewAIFromModel wraps an in-memory InferenceModel directly into an active AI engine without disk I/O.
var NewAIFromModel = NewRouterFromModel

// NewRouter loads a binary model file into memory once and constructs an immutable routing core.
func NewRouter(modelPath string, defaultThreshold float64) (*Router, error) {
	model, err := LoadBinaryModel(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize inference model: %w", err)
	}

	router := NewRouterFromModel(model, defaultThreshold)
	return router, nil
}

// NewRouterFromModel wraps an in-memory InferenceModel directly into an active Router without disk I/O.
func NewRouterFromModel(model *InferenceModel, defaultThreshold ...float64) *Router {
	threshold := 0.60
	if len(defaultThreshold) > 0 && defaultThreshold[0] > 0.0 {
		threshold = defaultThreshold[0]
	}

	policy := DefaultDispatchPolicy()
	policy.HighThreshold = threshold
	policy.LowThreshold = threshold * 0.6

	if model.Header.CalibratedMinEnergy > 0 {
		policy.MinLogSumExp = float64(model.Header.CalibratedMinEnergy)
	}
	if model.Header.CalibratedMargin > 0 {
		policy.RawLogitMargin = model.Header.CalibratedMargin
	}

	router := &Router{
		threshold: threshold,
		policy:    policy,
		telemetry: NewTelemetryRingBuffer(1024),
	}
	router.model.Store(model)
	return router
}

// Retrain retrains the underlying model in-process using the provided samples and atomically swaps weights without downtime.
func (r *Router) Retrain(samples []DataSample, configs ...TrainConfig) error {
	cfg := DefaultTrainConfig()
	if len(configs) > 0 {
		cfg = configs[0]
	}

	newModel, err := TrainModel(samples, cfg)
	if err != nil {
		return fmt.Errorf("in-process retraining failed: %w", err)
	}

	r.mu.Lock()
	r.samples = append([]DataSample(nil), samples...)
	if newModel.Header.CalibratedMinEnergy > 0 {
		r.policy.MinLogSumExp = float64(newModel.Header.CalibratedMinEnergy)
	}
	if newModel.Header.CalibratedMargin > 0 {
		r.policy.RawLogitMargin = newModel.Header.CalibratedMargin
	}
	r.mu.Unlock()

	r.SwapModel(newModel)
	return nil
}

// AppendData appends new samples to the existing training dataset, retrains in-process, and atomically swaps weights.
func (r *Router) AppendData(newSamples []DataSample, configs ...TrainConfig) error {
	r.mu.Lock()
	merged := make([]DataSample, len(r.samples)+len(newSamples))
	copy(merged, r.samples)
	copy(merged[len(r.samples):], newSamples)
	r.mu.Unlock()

	return r.Retrain(merged, configs...)
}

// AppendDataMap converts a label-to-phrases map into DataSamples and appends them for instant retraining.
func (r *Router) AppendDataMap(data map[string][]string, configs ...TrainConfig) error {
	var samples []DataSample
	for label, phrases := range data {
		for _, phrase := range phrases {
			samples = append(samples, DataSample{Text: phrase, Label: label})
		}
	}
	return r.AppendData(samples, configs...)
}

// Reload parses, validates, and atomically swaps model weights without interrupting active traffic.
func (r *Router) Reload(modelPath string) error {
	newModel, err := LoadBinaryModel(modelPath)
	if err != nil {
		return fmt.Errorf("failed to reload model: %w", err)
	}

	r.model.Store(newModel)
	return nil
}

// SwapModel replaces the active inference model atomically.
func (r *Router) SwapModel(newModel *InferenceModel) {
	r.model.Store(newModel)
}

// Model returns the currently active InferenceModel snapshot.
func (r *Router) Model() *InferenceModel {
	return r.model.Load()
}

// EnableTelemetry configures or resizes the telemetry ring buffer for active learning feedback.
func (r *Router) EnableTelemetry(capacity int) *Router {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.telemetry = NewTelemetryRingBuffer(capacity)
	return r
}

// DrainTelemetry extracts all recorded routing events for drift monitoring and active learning retraining.
func (r *Router) DrainTelemetry() []TelemetryEvent {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if r.telemetry == nil {
		return nil
	}
	return r.telemetry.Drain()
}

func (r *Router) recordTelemetry(text, primary, secondary string, conf, entropy float64, isAmbiguous, isPipeline, isFallback bool) {
	if r.telemetry != nil && (isAmbiguous || isPipeline || isFallback) {
		r.telemetry.Push(TelemetryEvent{
			InputText:      text,
			PredictedLabel: primary,
			SecondaryLabel: secondary,
			Confidence:     conf,
			Entropy:        entropy,
			IsAmbiguous:    isAmbiguous,
			IsPipeline:     isPipeline,
			IsFallback:     isFallback,
		})
	}
}

// SetPolicy updates the 3-tier routing thresholds and OOD entropy boundary.
func (r *Router) SetPolicy(policy DispatchPolicy) *Router {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.policy = policy
	return r
}

// SetSingleCharRatioCutoff configures the Layer 1 unlearned single-character token ratio threshold.
func (r *Router) SetSingleCharRatioCutoff(cutoff float64) *Router {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.policy.MaxSingleCharRatio = cutoff
	return r
}

// SetTemperature configures the temperature scaling factor used in softmax calculations.
func (r *Router) SetTemperature(t float32) *Router {
	r.mu.Lock()
	defer r.mu.Unlock()
	if m := r.model.Load(); m != nil {
		m.Temperature = t
	}
	return r
}

// Temperature returns the current temperature scaling factor.
func (r *Router) Temperature() float32 {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if m := r.model.Load(); m != nil {
		return m.Temperature
	}
	return 0
}
