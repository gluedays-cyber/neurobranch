package neurobranch

// Option defines a functional configuration option for Router and AI instances.
type Option func(*Router)

// WithPolicy overrides the entire DispatchPolicy for the router.
func WithPolicy(policy DispatchPolicy) Option {
	return func(r *Router) {
		r.SetPolicy(policy)
	}
}

// WithEnergyThreshold overrides the minimum Free Energy (LogSumExp) cutoff for strict OOD isolation.
func WithEnergyThreshold(threshold float64) Option {
	return func(r *Router) {
		r.SetEnergyThreshold(threshold)
	}
}

// WithConfidenceThreshold overrides the high confidence execution threshold, and optionally the low fallback threshold.
func WithConfidenceThreshold(high float64, low ...float64) Option {
	return func(r *Router) {
		r.SetConfidenceThreshold(high, low...)
	}
}

// WithMarginCutoff sets the minimum required confidence gap between Top-1 and Top-2 predictions.
func WithMarginCutoff(margin float64) Option {
	return func(r *Router) {
		r.mu.Lock()
		defer r.mu.Unlock()
		r.policy.MarginCutoff = margin
	}
}

// WithMaxEntropy sets the maximum allowable prediction entropy before triggering OOD Fallback.
func WithMaxEntropy(maxEntropy float64) Option {
	return func(r *Router) {
		r.mu.Lock()
		defer r.mu.Unlock()
		r.policy.MaxEntropy = maxEntropy
	}
}

// WithPipelineThreshold sets the minimum secondary confidence to qualify for multi-intent pipeline chaining.
func WithPipelineThreshold(threshold float64) Option {
	return func(r *Router) {
		r.mu.Lock()
		defer r.mu.Unlock()
		r.policy.PipelineThreshold = threshold
	}
}

// WithPatternGuard enables or disables zero-allocation pattern filtering for hex/base64 noise.
func WithPatternGuard(enabled bool) Option {
	return func(r *Router) {
		r.mu.Lock()
		defer r.mu.Unlock()
		r.policy.EnablePatternGuard = enabled
	}
}
