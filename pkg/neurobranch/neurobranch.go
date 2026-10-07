package neurobranch

import (
	"errors"
	"fmt"
	"math"
	"math/bits"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	// MaxBranchClasses defines the maximum supported classes for zero-allocation stack buffers.
	MaxBranchClasses = 16
	// MaxGateClasses is an alias for backward compatibility.
	MaxGateClasses = MaxBranchClasses

	// MaxBranchEmbDim defines the maximum supported embedding dimension for zero-allocation stack buffers.
	MaxBranchEmbDim = 64
	// MaxGateEmbDim is an alias for backward compatibility.
	MaxGateEmbDim = MaxBranchEmbDim
)

var (
	ErrNeuroBranchModelNil = errors.New("neurobranch: underlying model is nil")
	ErrClassLimitExceeded  = errors.New("neurobranch: registered classes exceed MaxBranchClasses")
)

// NeuroBranch coordinates a single shared neural backbone with a geometric 3-head zero-allocation branch engine.
type NeuroBranch struct {
	model atomic.Pointer[InferenceModel]

	mu           sync.RWMutex
	labels       [MaxBranchClasses]string
	classCount   int
	labelToIndex map[string]int

	anchorDict     map[string]uint64
	anchorTokenMap map[uint32]uint64
	anchorRules    []AnchorRule
	maxAnchorBoost float32

	domainCentroid [MaxBranchEmbDim]float32
	hasCentroid    bool
	minCosineSim   float32
	domainMeanSim  float32
	domainStdDev   float32

	policy DispatchPolicy
}

// NewNeuroBranch loads an InferenceModel from disk and prepares a high-performance NeuroBranch.
func NewNeuroBranch(modelPath string) (*NeuroBranch, error) {
	model, err := LoadBinaryModel(modelPath)
	if err != nil {
		return nil, fmt.Errorf("neurobranch: failed to load model %s: %w", modelPath, err)
	}
	return NewNeuroBranchWithModel(model), nil
}

// NewNeuroBranchWithModel wraps an existing InferenceModel into a NeuroBranch instance.
func NewNeuroBranchWithModel(model *InferenceModel) *NeuroBranch {
	nb := &NeuroBranch{
		labelToIndex:   make(map[string]int),
		anchorDict:     make(map[string]uint64),
		anchorTokenMap: make(map[uint32]uint64),
		policy:         DefaultDispatchPolicy(),
		minCosineSim:   0.25,
		maxAnchorBoost: DefaultMaxAnchorBoost,
	}
	nb.policy.MaxSingleCharRatio = 0.85
	nb.model.Store(model)

	// Pre-register model labels up to MaxBranchClasses
	for i, lbl := range model.Labels {
		if i >= MaxBranchClasses {
			break
		}
		nb.labels[i] = lbl
		nb.labelToIndex[lbl] = i
		nb.classCount++
	}

	// Apply self-calibrated thresholds from binary header if present
	if model.Header.CalibratedMinEnergy > 0 {
		nb.policy.MinLogSumExp = float64(model.Header.CalibratedMinEnergy)
		nb.policy.EnergyThreshold = float64(model.Header.CalibratedMinEnergy)
	}
	if model.Header.CalibratedMargin > 0 {
		nb.policy.RawLogitMargin = model.Header.CalibratedMargin
	}
	if model.Header.CalibratedMinCosine != 0 {
		nb.minCosineSim = model.Header.CalibratedMinCosine
	}

	// Auto-compute baseline domain centroid from valid vocabulary embeddings
	nb.computeBaselineCentroid(model)

	return nb
}

// SetMaxAnchorBoost sets the maximum cumulative soft-bias logit boost allowed per class (prevents logit explosion).
func (g *NeuroBranch) SetMaxAnchorBoost(cap float32) *NeuroBranch {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.maxAnchorBoost = cap
	return g
}

// MaxAnchorBoost returns the current cap for cumulative anchor logit boost.
func (g *NeuroBranch) MaxAnchorBoost() float32 {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.maxAnchorBoost
}

// SetDomainBoundary configures the reference L2 centroid and minimum cosine similarity for OOD rejection.
func (g *NeuroBranch) SetDomainBoundary(centroid []float32, minCosine float32) *NeuroBranch {
	g.mu.Lock()
	defer g.mu.Unlock()

	dim := len(centroid)
	if dim > MaxGateEmbDim {
		dim = MaxGateEmbDim
	}
	copy(g.domainCentroid[:dim], centroid[:dim])
	L2Normalize(g.domainCentroid[:dim], g.domainCentroid[:dim])
	g.hasCentroid = true
	g.minCosineSim = minCosine
	return g
}

// SetMinCosineSim configures the cosine boundary threshold for the auto-computed centroid.
func (g *NeuroBranch) SetMinCosineSim(threshold float32) *NeuroBranch {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.minCosineSim = threshold
	if threshold <= 0 {
		g.hasCentroid = false
	} else {
		g.hasCentroid = true
	}
	return g
}

// SetPolicy updates the dispatch thresholds, margin gap, and entropy cutoff.
func (g *NeuroBranch) SetPolicy(policy DispatchPolicy) *NeuroBranch {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.policy = policy
	return g
}

// SetSingleCharRatioCutoff configures the Layer 1 unlearned single-character token ratio threshold.
func (g *NeuroBranch) SetSingleCharRatioCutoff(cutoff float64) *NeuroBranch {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.policy.MaxSingleCharRatio = cutoff
	return g
}

// SetEnergyThreshold configures the minimum Free Energy (LogSumExp) cutoff for strict OOD rejection.
func (g *NeuroBranch) SetEnergyThreshold(threshold float64) *NeuroBranch {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.policy.EnergyThreshold = threshold
	g.policy.MinLogSumExp = threshold
	return g
}

// SetTemperature configures the temperature scaling factor used in softmax calculations.
func (g *NeuroBranch) SetTemperature(t float32) *NeuroBranch {
	g.mu.Lock()
	defer g.mu.Unlock()
	if m := g.model.Load(); m != nil {
		m.Temperature = t
	}
	return g
}

// Temperature returns the current temperature scaling factor.
func (g *NeuroBranch) Temperature() float32 {
	g.mu.RLock()
	defer g.mu.RUnlock()
	if m := g.model.Load(); m != nil {
		return m.Temperature
	}
	return 0
}

// WithAnchor registers anchor keywords for a target class label, returning a BranchRouteBuilder for chaining.
func (g *NeuroBranch) WithAnchor(label string, weight float32, keywords ...string) *BranchRouteBuilder {
	g.mu.Lock()
	idx, exists := g.labelToIndex[label]
	if !exists {
		if g.classCount >= MaxBranchClasses {
			g.mu.Unlock()
			panic(ErrClassLimitExceeded)
		}
		idx = g.classCount
		g.labels[idx] = label
		g.labelToIndex[label] = idx
		g.classCount++
	}
	g.mu.Unlock()

	builder := &BranchRouteBuilder{
		branch:     g,
		classIndex: idx,
		label:      label,
	}
	return builder.WithAnchor(weight, keywords...)
}

// Select returns the verified intent label for native Go switch-case statements.
// Returns empty string ("") if OOD or failing confidence thresholds, falling into default:.
func (g *NeuroBranch) Select(text string) string {
	trace := g.Inspect(text)
	if trace.IsFallback || trace.IsOOD || trace.IsAmbiguous {
		return ""
	}
	return trace.PredictedLabel
}

// Is evaluates whether the query matches targetLabel with confidence exceeding threshold.
func (g *NeuroBranch) Is(text string, targetLabel string, threshold ...float64) bool {
	trace := g.Inspect(text)
	if trace.IsOOD || trace.PredictedLabel != targetLabel {
		return false
	}
	limit := g.policy.HighThreshold
	if len(threshold) > 0 && threshold[0] > 0.0 {
		limit = threshold[0]
	}
	return trace.Confidence >= limit
}

// If evaluates whether the query matches targetLabel, providing an intuitive conditional keyword for if statements.
func (g *NeuroBranch) If(text string, targetLabel string, threshold ...float64) bool {
	return g.Is(text, targetLabel, threshold...)
}

// SwapModel atomically updates the underlying inference model without downtime.
func (g *NeuroBranch) SwapModel(newModel *InferenceModel) {
	g.model.Store(newModel)
}

// Model returns the current atomic inference model pointer.
func (g *NeuroBranch) Model() *InferenceModel {
	return g.model.Load()
}

// Inspect evaluates input text across all 3 heads and returns detailed diagnostics without mutations.
func (g *NeuroBranch) Inspect(text string) BranchTrace {
	start := time.Now()
	model := g.model.Load()
	if model == nil {
		return BranchTrace{
			InputText:      text,
			IsFallback:     true,
			FallbackReason: "model not loaded",
			LatencyMicros:  time.Since(start).Microseconds(),
		}
	}

	if g.policy.EnablePatternGuard && ScanUnlearnedPatterns(text) {
		return BranchTrace{
			InputText:      text,
			IsFallback:     true,
			FallbackReason: "unlearned pattern detected",
			Threshold:      g.policy.HighThreshold,
			LatencyMicros:  time.Since(start).Microseconds(),
		}
	}

	if len(text) > MaxInputBytes {
		text = TruncateToRuneBoundary(text, MaxInputBytes)
	}

	tokens := model.Tokenizer.Encode(text)
	if len(tokens) == 0 {
		return BranchTrace{
			InputText:      text,
			IsFallback:     true,
			FallbackReason: "empty input tokens",
			Threshold:      g.policy.HighThreshold,
			LatencyMicros:  time.Since(start).Microseconds(),
		}
	}
	if len(tokens) > MaxSequenceTokens {
		tokens = tokens[:MaxSequenceTokens]
	}

	singleRatio, _ := model.Tokenizer.AnalyzeUnlearnedRatio(tokens)
	subwords := make([]string, len(tokens))
	unkCount := 0
	unkID, hasUnk := model.Tokenizer.VocabMap["[UNK]"]
	for i, id := range tokens {
		if hasUnk && id == unkID {
			unkCount++
		}
		if int(id) < len(model.Vocab) {
			subwords[i] = model.Vocab[id]
		}
	}
	unkRatio := float64(unkCount) / float64(len(tokens))

	g.mu.RLock()
	defer g.mu.RUnlock()

	embDim := int(model.Header.EmbeddingDim)
	if embDim > MaxGateEmbDim {
		embDim = MaxGateEmbDim
	}
	numClasses := g.classCount

	// Stack memory allocation for strictly 0 B/op feature extraction
	var pooledStack [MaxGateEmbDim]float32
	var rawLogitsStack [MaxGateClasses]float32
	var normPooled [MaxGateEmbDim]float32

	_ = model.PredictFeatures(tokens, pooledStack[:embDim], rawLogitsStack[:numClasses])

	// -------------------------------------------------------------
	// [Head 1]: Geometric L2 Cosine Out-of-Domain (OOD) Guard
	// -------------------------------------------------------------
	L2Normalize(pooledStack[:embDim], normPooled[:embDim])
	var cosineSim float32 = 1.0
	isOOD := false
	if g.hasCentroid {
		cosineSim = DotProduct(normPooled[:embDim], g.domainCentroid[:embDim])
		if cosineSim < g.minCosineSim {
			isOOD = true
		}
	}

	// -------------------------------------------------------------
	// [Head 2]: Tokenizer Anchor Bitmask & Symbolic Soft-Bias
	// -------------------------------------------------------------
	var textBitmask uint64 = 0
	var triggeredAnchors []string
	lowerText := strings.ToLower(text)
	for kw, mask := range g.anchorDict {
		if strings.Contains(lowerText, kw) {
			textBitmask |= mask
			triggeredAnchors = append(triggeredAnchors, kw)
		}
	}

	var adjustedLogits [MaxGateClasses]float32
	copy(adjustedLogits[:numClasses], rawLogitsStack[:numClasses])
	var classDeltas [MaxGateClasses]float32
	for _, rule := range g.anchorRules {
		matchedBits := textBitmask & rule.Mask
		if matchedBits != 0 {
			count := float32(bits.OnesCount64(matchedBits))
			classDeltas[rule.ClassIndex] += rule.Weight * count
			for _, inhClass := range rule.InhibitClasses {
				if inhClass >= 0 && inhClass < numClasses {
					classDeltas[inhClass] -= rule.Penalty * count
				}
			}
		}
	}
	for c := 0; c < numClasses; c++ {
		delta := classDeltas[c]
		if delta > 0 && g.maxAnchorBoost > 0 && delta > g.maxAnchorBoost {
			delta = g.maxAnchorBoost
		}
		adjustedLogits[c] += delta
	}

	// -------------------------------------------------------------
	// [Head 3]: Stack Softmax, Margin, and Shannon Entropy Gating
	// -------------------------------------------------------------
	var probs [MaxGateClasses]float32
	_ = Softmax(adjustedLogits[:numClasses], model.Temperature, probs[:numClasses])

	var bestIdx, secondIdx int = 0, 1
	var bestScore, secondScore float32 = -1.0, -1.0
	probMap := make(map[string]float32, numClasses)

	for i := 0; i < numClasses; i++ {
		lbl := g.labels[i]
		p := probs[i]
		probMap[lbl] = p
		if p > bestScore {
			secondScore = bestScore
			secondIdx = bestIdx
			bestScore = p
			bestIdx = i
		} else if p > secondScore {
			secondScore = p
			secondIdx = i
		}
	}

	// Calibrate confidence by unknown token ratio penalty
	calibratedConfidence := float64(bestScore) * (1.0 - unkRatio)
	calibratedSecond := float64(secondScore) * (1.0 - unkRatio)
	margin := calibratedConfidence - calibratedSecond
	entropy := float64(computeEntropy(probs[:numClasses]))

	bestLabel := g.labels[bestIdx]
	secondLabel := ""
	if numClasses >= 2 {
		secondLabel = g.labels[secondIdx]
	}

	// Compute LogSumExp and Free Energy across logits
	var maxLogit float32 = adjustedLogits[0]
	for i := 1; i < numClasses; i++ {
		if adjustedLogits[i] > maxLogit {
			maxLogit = adjustedLogits[i]
		}
	}
	var sumExp float64
	for i := 0; i < numClasses; i++ {
		sumExp += math.Exp(float64(adjustedLogits[i] - maxLogit))
	}
	logSumExp := float64(maxLogit) + math.Log(sumExp)
	freeEnergy := -logSumExp

	// Compute raw top-1 and top-2 logit margin and co-activation count
	var maxLogit1, maxLogit2 float32 = -math.MaxFloat32, -math.MaxFloat32
	var coActive uint8 = 0
	for i := 0; i < numClasses; i++ {
		l := adjustedLogits[i]
		if l >= DefaultActivationThreshold {
			coActive++
		}
		if l > maxLogit1 {
			maxLogit2 = maxLogit1
			maxLogit1 = l
		} else if l > maxLogit2 {
			maxLogit2 = l
		}
	}
	var logitMargin float32 = 0.0
	if numClasses >= 2 {
		logitMargin = maxLogit1 - maxLogit2
	}

	// -------------------------------------------------------------
	// [Head 1 & Head 3]: Unified OOD & Energy Guard
	// -------------------------------------------------------------
	var oodReason string

	effectiveMinEnergy := g.policy.MinLogSumExp
	if effectiveMinEnergy == 0 && model != nil && model.Header.CalibratedMinEnergy > 0 {
		effectiveMinEnergy = float64(model.Header.CalibratedMinEnergy)
	}
	effectiveMinCosine := g.minCosineSim
	if effectiveMinCosine == 0 && model != nil && model.Header.CalibratedMinCosine != 0 {
		effectiveMinCosine = model.Header.CalibratedMinCosine
	}

	if g.hasCentroid && effectiveMinCosine > 0 && cosineSim < effectiveMinCosine {
		isOOD = true
		oodReason = fmt.Sprintf("cosine similarity %.4f below domain threshold %.4f (OOD)", cosineSim, effectiveMinCosine)
	} else if len(tokens) >= 2 && g.policy.MaxSingleCharRatio > 0 && singleRatio >= g.policy.MaxSingleCharRatio {
		isOOD = true
		oodReason = fmt.Sprintf("unlearned vocabulary (single-char ratio %.2f >= %.2f)", singleRatio, g.policy.MaxSingleCharRatio)
	} else if effectiveMinEnergy > 0 && logSumExp < effectiveMinEnergy {
		isOOD = true
		oodReason = fmt.Sprintf("free energy %.4f (logSumExp %.4f) below in-distribution threshold %.4f (OOD)", freeEnergy, logSumExp, effectiveMinEnergy)
	} else if entropy > g.policy.MaxEntropy {
		isOOD = true
		oodReason = fmt.Sprintf("prediction entropy %.4f exceeds limit %.4f (OOD)", entropy, g.policy.MaxEntropy)
	} else if unkRatio >= 0.5 {
		isOOD = true
		oodReason = fmt.Sprintf("excessive unknown tokens (%.2f >= 0.50)", unkRatio)
	}

	trace := BranchTrace{
		InputText:          text,
		TokenIDs:           tokens,
		Subwords:           subwords,
		UnknownTokenRatio:  unkRatio,
		CosineSimilarity:   cosineSim,
		IsOOD:              isOOD,
		AnchorBitmask:      textBitmask,
		TriggeredAnchors:   triggeredAnchors,
		ClassProbabilities: probMap,
		PredictedLabel:     bestLabel,
		SecondaryLabel:     secondLabel,
		Confidence:         calibratedConfidence,
		Margin:             margin,
		LogitMargin:        logitMargin,
		CoActiveCount:      coActive,
		Entropy:            entropy,
		LogSumExp:          logSumExp,
		FreeEnergy:         freeEnergy,
		Threshold:          g.policy.HighThreshold,
		LatencyMicros:      time.Since(start).Microseconds(),
	}

	// Routing tier evaluation
	if isOOD {
		trace.IsOOD = true
		trace.IsFallback = true
		trace.FallbackReason = oodReason
	} else {
		// Multi-intent pipeline evaluation (Softmax probability, dual-anchor co-activation, or pre-softmax co-activation)
		if secondLabel != "" {
			var primaryAnchors, secondaryAnchors bool
			for _, rule := range g.anchorRules {
				if (textBitmask & rule.Mask) != 0 {
					if rule.ClassIndex == bestIdx {
						primaryAnchors = true
					}
					if rule.ClassIndex == secondIdx {
						secondaryAnchors = true
					}
				}
			}
			hasBothAnchors := primaryAnchors && secondaryAnchors
			isDualActivated := coActive >= 2 && adjustedLogits[secondIdx] >= DefaultActivationThreshold
			if calibratedSecond >= g.policy.PipelineThreshold || hasBothAnchors || isDualActivated {
				trace.IsPipeline = true
			}
		}

		effectiveMargin := g.policy.RawLogitMargin
		if effectiveMargin == 0 && model != nil && model.Header.CalibratedMargin > 0 {
			effectiveMargin = model.Header.CalibratedMargin
		}

		isAmbiguous := trace.Confidence < g.policy.HighThreshold || margin < g.policy.MarginCutoff || (effectiveMargin > 0.0 && logitMargin < effectiveMargin) || (coActive >= 2 && effectiveMargin > 0.0 && logitMargin < effectiveMargin*1.5)
		if isAmbiguous {
			trace.IsAmbiguous = true
		} else if trace.Confidence < g.policy.LowThreshold {
			trace.IsFallback = true
			trace.FallbackReason = fmt.Sprintf("confidence %.4f below low threshold %.4f", trace.Confidence, g.policy.LowThreshold)
		}
	}

	return trace
}
