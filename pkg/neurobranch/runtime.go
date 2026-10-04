package neurobranch

import (
	"errors"
	"fmt"
	"math"
	"sync"
	"unicode/utf8"
)

const (
	// MaxInputBytes limits raw input byte length to defend against algorithmic CPU exhaustion.
	MaxInputBytes = 512

	// MaxSequenceTokens limits token length to preserve sub-millisecond forward pass SLAs.
	MaxSequenceTokens = 128
)

var (
	ErrModelNotInitialized = errors.New("model not properly initialized")
	ErrClassIndexOutOfRange = errors.New("predicted class index exceeds label count")

	// Fail-Safe Sentinel Errors (Layer 1 & Layer 2)
	ErrUnlearnedVocabulary = errors.New("neurobranch: input dominated by unlearned subwords or OOV fragments")
	ErrUnlearnedPattern    = errors.New("neurobranch: unlearned random/hex pattern detected")
	ErrDegeneratedInput    = errors.New("neurobranch: degenerated repetitive token sequence detected")
	ErrLowConfidence       = errors.New("neurobranch: prediction confidence below safety threshold")
	ErrHighEntropy         = errors.New("neurobranch: prediction entropy exceeds uncertainty boundary")
	ErrOutOfDomain         = errors.New("neurobranch: request energy or representation is out of domain")
	ErrAmbiguousIntent     = errors.New("neurobranch: ambiguous intent between competing candidates")
)

// inferenceBuffer holds scratch memory slices to enable zero-allocation forward passes.
type inferenceBuffer struct {
	pooled []float32
	hidden []float32
	logits []float32
	probs  []float32
}

// InferenceModel represents an in-memory embedded classifier loaded from binary format.
type InferenceModel struct {
	Header      Header
	Labels      []string
	Vocab       []string
	MergeRules  []MergeRule
	Weights     Weights
	Temperature float32
	Tokenizer   *BPETokenizer

	bufPool sync.Pool
}

// NewInferenceModel constructs and prepares an InferenceModel with an internal scratch buffer pool.
func NewInferenceModel(
	header Header,
	labels []string,
	vocab []string,
	mergeRules []MergeRule,
	weights Weights,
) *InferenceModel {
	tok := NewBPETokenizer(vocab, mergeRules)

	model := &InferenceModel{
		Header:      header,
		Labels:      labels,
		Vocab:       vocab,
		MergeRules:  mergeRules,
		Weights:     weights,
		Temperature: 1.0,
		Tokenizer:   tok,
	}

	model.bufPool = sync.Pool{
		New: func() any {
			return &inferenceBuffer{
				pooled: make([]float32, header.EmbeddingDim),
				hidden: make([]float32, header.HiddenDim),
				logits: make([]float32, header.NumClasses),
				probs:  make([]float32, header.NumClasses),
			}
		},
	}

	return model
}

// MatchSlot captures a predicted class index and its normalized confidence score without heap allocations.
type MatchSlot struct {
	Index      int16
	Confidence float32
}

// StaticInferenceResult encapsulates top-2 ranked prediction slots and uncertainty entropy on the stack.
type StaticInferenceResult struct {
	Primary       MatchSlot
	Secondary     MatchSlot
	LogitMargin   float32
	Entropy       float32
	Energy        float32
	Total         uint8
	CoActiveCount uint8   // Number of classes breaching positive activation threshold
	Top1RawLogit  float32 // Unscaled raw logit of Primary candidate
	Top2RawLogit  float32 // Unscaled raw logit of Secondary candidate
}

// DefaultActivationThreshold defines the minimum raw logit value indicating significant class activation.
const DefaultActivationThreshold float32 = 0.0

// computeEntropy calculates Shannon entropy in bits with epsilon guards to prevent NaN/Inf underflows.
func computeEntropy(probs []float32) float32 {
	var entropy float64
	for _, p := range probs {
		if p > 1e-7 {
			entropy -= float64(p) * math.Log2(float64(p))
		}
	}
	return float32(entropy)
}

// forwardInternal executes the forward computation directly inside the provided scratch buffer without allocations.
func (m *InferenceModel) forwardInternal(tokenIDs []uint32, temperature float32, buf *inferenceBuffer) error {
	if len(tokenIDs) == 0 {
		return ErrEmptyInput
	}

	// 1. Mean Pooling with Positional Encoding: [SeqLen] -> [EmbeddingDim]
	if err := MeanPoolingWithPos(tokenIDs, m.Weights.Embedding, m.Weights.Positional, int(m.Header.EmbeddingDim), buf.pooled); err != nil {
		return fmt.Errorf("mean pooling failed: %w", err)
	}

	// 2. Layer 1 Linear: [EmbeddingDim] x [EmbeddingDim x HiddenDim] + [HiddenDim] -> [HiddenDim]
	if err := MatMulVecAdd(buf.pooled, m.Weights.W1, m.Weights.B1, int(m.Header.EmbeddingDim), int(m.Header.HiddenDim), buf.hidden); err != nil {
		return fmt.Errorf("layer 1 forward failed: %w", err)
	}

	// 3. GELU Non-Linear Activation In-Place
	GELUInPlace(buf.hidden)

	// 4. Layer 2 Linear: [HiddenDim] x [HiddenDim x NumClasses] + [NumClasses] -> [NumClasses]
	if err := MatMulVecAdd(buf.hidden, m.Weights.W2, m.Weights.B2, int(m.Header.HiddenDim), int(m.Header.NumClasses), buf.logits); err != nil {
		return fmt.Errorf("layer 2 forward failed: %w", err)
	}

	// 5. Softmax with Temperature Scaling
	temp := temperature
	if temp <= 0.0 {
		temp = m.Temperature
	}
	if err := Softmax(buf.logits, temp, buf.probs); err != nil {
		return fmt.Errorf("softmax failed: %w", err)
	}

	return nil
}

// Forward executes the 2-layer MLP inference over a slice of token IDs and returns newly allocated probabilities.
// Note: For zero-allocation hot paths, use PredictSlots or PredictTokens instead.
func (m *InferenceModel) Forward(tokenIDs []uint32, temperature float32) ([]float32, error) {
	buf := m.bufPool.Get().(*inferenceBuffer)
	defer m.bufPool.Put(buf)

	if err := m.forwardInternal(tokenIDs, temperature, buf); err != nil {
		return nil, err
	}

	result := make([]float32, m.Header.NumClasses)
	copy(result, buf.probs)
	return result, nil
}

// PredictSlots computes top-2 class predictions and entropy on the stack with ZERO heap allocation.
func (m *InferenceModel) PredictSlots(tokenIDs []uint32, temperature float32) (StaticInferenceResult, error) {
	buf := m.bufPool.Get().(*inferenceBuffer)
	defer m.bufPool.Put(buf)

	if err := m.forwardInternal(tokenIDs, temperature, buf); err != nil {
		return StaticInferenceResult{}, err
	}

	var top1Idx, top2Idx int16 = -1, -1
	var top1Prob, top2Prob float32 = -1.0, -1.0
	var top1Logit, top2Logit float32 = -math.MaxFloat32, -math.MaxFloat32
	var coActive uint8 = 0

	for i, p := range buf.probs {
		idx := int16(i)
		logit := buf.logits[i]

		if logit >= DefaultActivationThreshold {
			coActive++
		}

		if p > top1Prob {
			top2Prob = top1Prob
			top2Idx = top1Idx
			top2Logit = top1Logit

			top1Prob = p
			top1Idx = idx
			top1Logit = logit
		} else if p > top2Prob {
			top2Prob = p
			top2Idx = idx
			top2Logit = logit
		}
	}

	var res StaticInferenceResult
	if top1Idx >= 0 {
		res.Primary = MatchSlot{Index: top1Idx, Confidence: top1Prob}
		res.Total = 1
		res.Top1RawLogit = top1Logit
	}
	if top2Idx >= 0 {
		res.Secondary = MatchSlot{Index: top2Idx, Confidence: top2Prob}
		res.Total = 2
		res.Top2RawLogit = top2Logit
		res.LogitMargin = top1Logit - top2Logit
	} else if top1Idx >= 0 {
		res.LogitMargin = float32(math.MaxFloat32)
	}
	res.CoActiveCount = coActive
	res.Entropy = computeEntropy(buf.probs)
	res.Energy = LogSumExp(buf.logits)

	return res, nil
}

// PredictTokens computes class probabilities and returns the top label alongside its confidence score with zero allocations.
func (m *InferenceModel) PredictTokens(tokenIDs []uint32) (string, float64, error) {
	res, err := m.PredictSlots(tokenIDs, m.Temperature)
	if err != nil {
		return "", 0.0, err
	}

	bestIdx := int(res.Primary.Index)
	if bestIdx < 0 || bestIdx >= len(m.Labels) {
		return "", 0.0, ErrClassIndexOutOfRange
	}

	return m.Labels[bestIdx], float64(res.Primary.Confidence), nil
}

// PredictDetailed executes inference and returns top-2 slots, calibrated confidence, and OOV ratio with zero allocations.
func (m *InferenceModel) PredictDetailed(text string) (StaticInferenceResult, float64, error) {
	if !utf8.ValidString(text) {
		return StaticInferenceResult{}, 0.0, ErrEmptyInput
	}
	if ScanUnlearnedPatterns(text) {
		return StaticInferenceResult{}, 1.0, ErrUnlearnedPattern
	}
	if len(text) > MaxInputBytes {
		text = TruncateToRuneBoundary(text, MaxInputBytes)
	}

	tokenIDs := m.Tokenizer.Encode(text)
	if len(tokenIDs) == 0 {
		return StaticInferenceResult{}, 0.0, ErrEmptyInput
	}
	if len(tokenIDs) > MaxSequenceTokens {
		tokenIDs = tokenIDs[:MaxSequenceTokens]
	}

	res, err := m.PredictSlots(tokenIDs, m.Temperature)
	if err != nil {
		return StaticInferenceResult{}, 0.0, err
	}

	// Guard 3: Calculate UNK & single-character fallback ratio, then penalize confidence proportionally
	singleRatio, unkRatio := m.Tokenizer.AnalyzeUnlearnedRatio(tokenIDs)
	effectivePenalty := unkRatio
	if singleRatio > 0.5 {
		effectivePenalty = math.Max(unkRatio, (singleRatio-0.5)*2.0)
	}

	decay := float32(1.0 - effectivePenalty)
	if decay < 0.0 {
		decay = 0.0
	}
	res.Primary.Confidence *= decay
	res.Secondary.Confidence *= decay

	return res, effectivePenalty, nil
}


// TruncateToRuneBoundary truncates text to at most maxBytes without slicing multi-byte UTF-8 runes.
func TruncateToRuneBoundary(text string, maxBytes int) string {
	if len(text) <= maxBytes {
		return text
	}
	idx := maxBytes
	for idx > 0 && !utf8.RuneStart(text[idx]) {
		idx--
	}
	return text[:idx]
}

// Predict tokenizes raw text with subword BPE and returns predicted label and confidence score with safety guards.
func (m *InferenceModel) Predict(text string) (string, float64, error) {
	// Guard 1: Validate UTF-8 and truncate oversized input strings respecting rune boundaries
	if !utf8.ValidString(text) {
		return "", 0.0, ErrEmptyInput
	}
	if len(text) > MaxInputBytes {
		text = TruncateToRuneBoundary(text, MaxInputBytes)
	}

	tokenIDs := m.Tokenizer.Encode(text)
	if len(tokenIDs) == 0 {
		return "", 0.0, ErrEmptyInput
	}

	// Guard 2: Clamp sequence length to prevent excessive pooling latency
	if len(tokenIDs) > MaxSequenceTokens {
		tokenIDs = tokenIDs[:MaxSequenceTokens]
	}

	label, score, err := m.PredictTokens(tokenIDs)
	if err != nil {
		return "", 0.0, err
	}

	// Guard 3: Penalize confidence proportionally if input is dominated by out-of-vocabulary [UNK] tokens
	unkID, hasUnk := m.Tokenizer.VocabMap["[UNK]"]
	if hasUnk && len(tokenIDs) > 0 {
		unkCount := 0
		for _, id := range tokenIDs {
			if id == unkID {
				unkCount++
			}
		}
		unkRatio := float64(unkCount) / float64(len(tokenIDs))
		score = score * (1.0 - unkRatio)
	}

	return label, score, nil
}

// PredictFeatures fills the provided outPooled and outLogits slices with zero allocations.
func (m *InferenceModel) PredictFeatures(tokenIDs []uint32, outPooled []float32, outLogits []float32) error {
	buf := m.bufPool.Get().(*inferenceBuffer)
	defer m.bufPool.Put(buf)

	if err := m.forwardInternal(tokenIDs, m.Temperature, buf); err != nil {
		return err
	}

	if len(outPooled) >= len(buf.pooled) {
		copy(outPooled, buf.pooled)
	}
	if len(outLogits) >= len(buf.logits) {
		copy(outLogits, buf.logits)
	}
	return nil
}



