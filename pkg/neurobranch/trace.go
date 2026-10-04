package neurobranch

// RouteDecision encapsulates the verified intent result and diagnostic signals for fail-safe error handling.
type RouteDecision struct {
	Intent              string  `json:"intent"`
	Confidence          float64 `json:"confidence"`
	Entropy             float64 `json:"entropy"`
	Energy              float64 `json:"energy"`
	Margin              float64 `json:"margin"`
	SingleCharRatio     float64 `json:"single_char_ratio"`
	UnknownTokenRatio   float64 `json:"unknown_token_ratio"`
	UniqueTokenRatio    float64 `json:"unique_token_ratio"`
	LogitMargin         float32 `json:"logit_margin"`
	CoActiveCount       uint8   `json:"co_active_count"`
	SecondaryIntent     string  `json:"secondary_intent,omitempty"`
	SecondaryConfidence float64 `json:"secondary_confidence,omitempty"`
}

// RouteTrace encapsulates comprehensive diagnostic metadata explaining a routing decision.
type RouteTrace struct {
	InputText          string             `json:"input_text"`
	TokenIDs           []uint32           `json:"token_ids"`
	Subwords           []string           `json:"subwords"`
	SingleCharRatio    float64            `json:"single_char_ratio"`
	UnknownTokenRatio  float64            `json:"unknown_token_ratio"`
	UniqueTokenRatio   float64            `json:"unique_token_ratio"`
	ClassProbabilities map[string]float32 `json:"class_probabilities"`
	PredictedLabel     string             `json:"predicted_label"`
	SecondaryLabel     string             `json:"secondary_label,omitempty"`
	Confidence         float64            `json:"confidence"`
	Margin             float64            `json:"margin"`
	LogitMargin        float32            `json:"logit_margin"`
	CoActiveCount      uint8              `json:"co_active_count"`
	Entropy            float64            `json:"entropy"`
	Energy             float64            `json:"energy"`
	Threshold          float64            `json:"threshold"`
	IsAmbiguous        bool               `json:"is_ambiguous"`
	IsPipeline         bool               `json:"is_pipeline"`
	IsFallback         bool               `json:"is_fallback"`
	FallbackReason     string             `json:"fallback_reason,omitempty"`
	LatencyMicros      int64              `json:"latency_micros"`
}

// BranchTrace captures comprehensive runtime metrics across all three geometric heads.
type BranchTrace struct {
	InputText          string             `json:"input_text"`
	TokenIDs           []uint32           `json:"token_ids"`
	Subwords           []string           `json:"subwords"`
	UnknownTokenRatio  float64            `json:"unknown_token_ratio"`
	CosineSimilarity   float32            `json:"cosine_similarity"`
	IsOOD              bool               `json:"is_ood"`
	AnchorBitmask      uint64             `json:"anchor_bitmask"`
	TriggeredAnchors   []string           `json:"triggered_anchors"`
	ClassProbabilities map[string]float32 `json:"class_probabilities"`
	PredictedLabel     string             `json:"predicted_label"`
	SecondaryLabel     string             `json:"secondary_label,omitempty"`
	Confidence         float64            `json:"confidence"`
	Margin             float64            `json:"margin"`
	LogitMargin        float32            `json:"logit_margin"`
	CoActiveCount      uint8              `json:"co_active_count"`
	Entropy            float64            `json:"entropy"`
	LogSumExp          float64            `json:"log_sum_exp"`
	FreeEnergy         float64            `json:"free_energy"`
	Threshold          float64            `json:"threshold"`
	IsAmbiguous        bool               `json:"is_ambiguous"`
	IsPipeline         bool               `json:"is_pipeline"`
	IsFallback         bool               `json:"is_fallback"`
	FallbackReason     string             `json:"fallback_reason,omitempty"`
	LatencyMicros      int64              `json:"latency_micros"`
}

// GateTrace is an alias for BranchTrace for backward compatibility.
type GateTrace = BranchTrace

