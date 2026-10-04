package neurobranch

// DispatchPolicy defines the 3-tier confidence criteria, multi-intent threshold, and OOD entropy boundary.
type DispatchPolicy struct {
	HighThreshold        float64 `json:"high_threshold"`            // Minimum confidence for definite execution (default: 0.75)
	LowThreshold         float64 `json:"low_threshold"`             // Minimum confidence below which request is isolated to Fallback (default: 0.40)
	MarginCutoff         float64 `json:"margin_cutoff"`             // Minimum required gap between Top-1 and Top-2 (default: 0.15)
	MaxEntropy           float64 `json:"max_entropy"`               // Maximum allowable prediction entropy before triggering OOD Fallback (default: 2.0)
	PipelineThreshold    float64 `json:"pipeline_threshold"`        // Minimum secondary confidence to qualify for multi-intent pipeline (default: 0.30)
	MinLogSumExp         float64 `json:"min_log_sum_exp,omitempty"` // Minimum log-sum-exp energy boundary before OOD isolation (0 disables)
	MaxSingleCharRatio   float64 `json:"max_single_char_ratio"`     // Layer 1: Max ratio of single-char fallback tokens (default: 0.70)
	MaxUnknownTokenRatio float64 `json:"max_unknown_token_ratio"`   // Layer 1: Max ratio of UNK tokens (default: 0.30)
	MinUniqueTokenRatio  float64 `json:"min_unique_token_ratio"`    // Layer 1: Min ratio of unique tokens to block flood/repetition (default: 0.25)
	EnablePatternGuard   bool    `json:"enable_pattern_guard"`      // Layer 0: Zero-alloc scan for random hex/base64 patterns (default: true)
	RawLogitMargin       float32 `json:"raw_logit_margin"`          // Layer 2: Minimum required gap between Top-1 and Top-2 raw logits (default: 0.35)
}

// DefaultDispatchPolicy creates standard production-ready 3-tier routing criteria.
func DefaultDispatchPolicy() DispatchPolicy {
	return DispatchPolicy{
		HighThreshold:        0.75,
		LowThreshold:         0.40,
		MarginCutoff:         0.15,
		MaxEntropy:           2.0,
		PipelineThreshold:    0.30,
		MinLogSumExp:         0.0,
		MaxSingleCharRatio:   0.70,
		MaxUnknownTokenRatio: 0.30,
		MinUniqueTokenRatio:  0.25,
		EnablePatternGuard:   true,
		RawLogitMargin:       0.35,
	}
}

// CalculateUniqueTokenRatio computes the ratio of unique tokens in a sequence with zero heap allocation.
func CalculateUniqueTokenRatio(tokens []uint32) float64 {
	n := len(tokens)
	if n == 0 {
		return 0.0
	}
	uniqueCount := 0
	for i := 0; i < n; i++ {
		seen := false
		for j := 0; j < i; j++ {
			if tokens[i] == tokens[j] {
				seen = true
				break
			}
		}
		if !seen {
			uniqueCount++
		}
	}
	return float64(uniqueCount) / float64(n)
}
