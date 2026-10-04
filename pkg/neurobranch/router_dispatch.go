package neurobranch

import (
	"context"
	"errors"
	"fmt"
	"math"
	"time"
	"unicode/utf8"
)

// RouteQuery executes 2-layer fail-safe evaluation and returns standard Go sentinel errors on failure.
func (r *Router) RouteQuery(ctx context.Context, text string) (RouteDecision, error) {
	select {
	case <-ctx.Done():
		return RouteDecision{}, ctx.Err()
	default:
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	if err := ctx.Err(); err != nil {
		return RouteDecision{}, err
	}

	model := r.model.Load()
	if model == nil {
		return RouteDecision{}, ErrModelNotInitialized
	}

	if !utf8.ValidString(text) {
		return RouteDecision{}, ErrEmptyInput
	}
	if len(text) > MaxInputBytes {
		text = TruncateToRuneBoundary(text, MaxInputBytes)
	}

	// Guard 0: Pre-inference pattern detection (< 1 μs)
	if r.policy.EnablePatternGuard && ScanUnlearnedPatterns(text) {
		return RouteDecision{}, ErrUnlearnedPattern
	}

	tokenIDs := model.Tokenizer.Encode(text)
	if len(tokenIDs) == 0 {
		return RouteDecision{}, ErrEmptyInput
	}
	if len(tokenIDs) > MaxSequenceTokens {
		tokenIDs = tokenIDs[:MaxSequenceTokens]
	}

	// -------------------------------------------------------------
	// [Layer 1 Guard] Tokenizer-level unlearned vocabulary cutoff (< 1 μs)
	// -------------------------------------------------------------
	singleRatio, unkRatio := model.Tokenizer.AnalyzeUnlearnedRatio(tokenIDs)
	uniqueRatio := CalculateUniqueTokenRatio(tokenIDs)

	if len(tokenIDs) >= 4 && r.policy.MinUniqueTokenRatio > 0.0 && uniqueRatio < r.policy.MinUniqueTokenRatio {
		return RouteDecision{
			SingleCharRatio:   singleRatio,
			UnknownTokenRatio: unkRatio,
			UniqueTokenRatio:  uniqueRatio,
		}, ErrDegeneratedInput
	}
	if len(tokenIDs) >= 2 && r.policy.MaxSingleCharRatio > 0.0 && singleRatio >= r.policy.MaxSingleCharRatio {
		return RouteDecision{
			SingleCharRatio:   singleRatio,
			UnknownTokenRatio: unkRatio,
			UniqueTokenRatio:  uniqueRatio,
		}, ErrUnlearnedVocabulary
	}
	if r.policy.MaxUnknownTokenRatio > 0.0 && unkRatio >= r.policy.MaxUnknownTokenRatio {
		return RouteDecision{
			SingleCharRatio:   singleRatio,
			UnknownTokenRatio: unkRatio,
			UniqueTokenRatio:  uniqueRatio,
		}, ErrUnlearnedVocabulary
	}

	// -------------------------------------------------------------
	// [Layer 2 Guard] Neural forward pass & metric-based cutoff (~29 μs)
	// -------------------------------------------------------------
	res, err := model.PredictSlots(tokenIDs, model.Temperature)
	if err != nil {
		return RouteDecision{}, err
	}
	if res.Total == 0 {
		return RouteDecision{}, ErrOutOfDomain
	}

	primaryIdx := int(res.Primary.Index)
	if primaryIdx < 0 || primaryIdx >= len(model.Labels) {
		return RouteDecision{}, ErrClassIndexOutOfRange
	}

	primaryLabel := model.Labels[primaryIdx]
	rawPrimaryConf := float64(res.Primary.Confidence)
	entropy := float64(res.Entropy)
	energy := float64(res.Energy)

	var secondaryLabel string
	var rawSecondaryConf float64
	if res.Total >= 2 {
		secIdx := int(res.Secondary.Index)
		if secIdx >= 0 && secIdx < len(model.Labels) {
			secondaryLabel = model.Labels[secIdx]
			rawSecondaryConf = float64(res.Secondary.Confidence)
		}
	}

	// Guard 3: Calculate single-character fragment & excessive UNK penalty
	unkPenalty := 0.0
	if unkRatio > 0.25 {
		unkPenalty = (unkRatio - 0.25) * 1.5
	}
	singlePenalty := 0.0
	if singleRatio > 0.5 {
		singlePenalty = (singleRatio - 0.5) * 2.0
	}
	effectivePenalty := math.Max(unkPenalty, singlePenalty)
	if effectivePenalty > 1.0 {
		effectivePenalty = 1.0
	}
	calibratedConfidence := rawPrimaryConf * (1.0 - effectivePenalty)
	calibratedSecond := rawSecondaryConf * (1.0 - effectivePenalty)
	margin := calibratedConfidence - calibratedSecond

	decision := RouteDecision{
		Intent:              primaryLabel,
		Confidence:          calibratedConfidence,
		Entropy:             entropy,
		Energy:              energy,
		Margin:              margin,
		LogitMargin:         res.LogitMargin,
		CoActiveCount:       res.CoActiveCount,
		SingleCharRatio:     singleRatio,
		UnknownTokenRatio:   unkRatio,
		UniqueTokenRatio:    uniqueRatio,
		SecondaryIntent:     secondaryLabel,
		SecondaryConfidence: calibratedSecond,
	}

	effectiveMinEnergy := r.policy.MinLogSumExp
	if effectiveMinEnergy == 0 && model != nil && model.Header.CalibratedMinEnergy > 0 {
		effectiveMinEnergy = float64(model.Header.CalibratedMinEnergy)
	}
	effectiveMargin := r.policy.RawLogitMargin
	if effectiveMargin == 0 && model != nil && model.Header.CalibratedMargin > 0 {
		effectiveMargin = model.Header.CalibratedMargin
	}

	// 1. Energy boundary (LogSumExp)
	if effectiveMinEnergy != 0.0 && energy < effectiveMinEnergy {
		return decision, ErrOutOfDomain
	}

	// 2. High entropy boundary
	if entropy > r.policy.MaxEntropy {
		return decision, ErrHighEntropy
	}

	// 3. Ambiguity margin boundary (both probability margin, raw logit gap, and co-activation conflict)
	isAmbiguous := calibratedConfidence < r.policy.HighThreshold || margin < r.policy.MarginCutoff || (effectiveMargin > 0.0 && res.LogitMargin < effectiveMargin) || (res.CoActiveCount >= 2 && effectiveMargin > 0.0 && res.LogitMargin < effectiveMargin*1.5)
	if isAmbiguous {
		return decision, ErrAmbiguousIntent
	}

	// 4. Low confidence boundary
	if calibratedConfidence < r.policy.LowThreshold {
		return decision, ErrLowConfidence
	}

	return decision, nil
}

// Inspect evaluates input text and generates a full diagnostic RouteTrace including 3-tier and entropy metrics.
func (r *Router) Inspect(text string) RouteTrace {
	start := time.Now()
	r.mu.RLock()
	defer r.mu.RUnlock()

	model := r.model.Load()
	if model == nil {
		return RouteTrace{
			InputText:      text,
			IsFallback:     true,
			FallbackReason: "model not loaded",
			Threshold:      r.threshold,
			LatencyMicros:  time.Since(start).Microseconds(),
		}
	}

	// Guard 0: Pre-inference pattern check
	if r.policy.EnablePatternGuard && ScanUnlearnedPatterns(text) {
		return RouteTrace{
			InputText:      text,
			IsFallback:     true,
			FallbackReason: "unlearned pattern detected",
			Threshold:      r.threshold,
			LatencyMicros:  time.Since(start).Microseconds(),
		}
	}

	// Guard 1: Truncate oversized input strings respecting UTF-8 rune boundaries
	if len(text) > MaxInputBytes {
		text = TruncateToRuneBoundary(text, MaxInputBytes)
	}

	tokens := model.Tokenizer.Encode(text)
	if len(tokens) == 0 {
		return RouteTrace{
			InputText:      text,
			IsFallback:     true,
			FallbackReason: "empty input tokens",
			Threshold:      r.threshold,
			LatencyMicros:  time.Since(start).Microseconds(),
		}
	}

	// Guard 2: Clamp sequence length
	if len(tokens) > MaxSequenceTokens {
		tokens = tokens[:MaxSequenceTokens]
	}

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

	singleRatio, unkRatio := model.Tokenizer.AnalyzeUnlearnedRatio(tokens)
	uniqueRatio := CalculateUniqueTokenRatio(tokens)

	probs, err := model.Forward(tokens, model.Temperature)
	probMap := make(map[string]float32, len(model.Labels))

	var bestLabel, secondLabel string
	var bestScore, secondScore float32 = -1.0, -1.0
	for i, p := range probs {
		if i < len(model.Labels) {
			lbl := model.Labels[i]
			probMap[lbl] = p
			if p > bestScore {
				secondScore = bestScore
				secondLabel = bestLabel
				bestScore = p
				bestLabel = lbl
			} else if p > secondScore {
				secondScore = p
				secondLabel = lbl
			}
		}
	}

	// Guard 3: Apply calibrated confidence penalty
	effectivePenalty := unkRatio
	if singleRatio > 0.5 {
		effectivePenalty = math.Max(unkRatio, (singleRatio-0.5)*2.0)
	}
	calibratedConfidence := float64(bestScore) * (1.0 - effectivePenalty)
	calibratedSecond := float64(secondScore) * (1.0 - effectivePenalty)
	margin := calibratedConfidence - calibratedSecond
	entropy := float64(computeEntropy(probs))

	slotRes, _ := model.PredictSlots(tokens, model.Temperature)
	energy := float64(slotRes.Energy)

	effectiveMinEnergy := r.policy.MinLogSumExp
	if effectiveMinEnergy == 0 && model.Header.CalibratedMinEnergy > 0 {
		effectiveMinEnergy = float64(model.Header.CalibratedMinEnergy)
	}
	effectiveMargin := r.policy.RawLogitMargin
	if effectiveMargin == 0 && model.Header.CalibratedMargin > 0 {
		effectiveMargin = model.Header.CalibratedMargin
	}

	trace := RouteTrace{
		InputText:          text,
		TokenIDs:           tokens,
		Subwords:           subwords,
		SingleCharRatio:    singleRatio,
		UnknownTokenRatio:  unkRatio,
		UniqueTokenRatio:   uniqueRatio,
		ClassProbabilities: probMap,
		PredictedLabel:     bestLabel,
		SecondaryLabel:     secondLabel,
		Confidence:         calibratedConfidence,
		Margin:             margin,
		LogitMargin:        slotRes.LogitMargin,
		CoActiveCount:      slotRes.CoActiveCount,
		Entropy:            entropy,
		Energy:             energy,
		Threshold:          r.policy.HighThreshold,
		LatencyMicros:      time.Since(start).Microseconds(),
	}

	if err != nil {
		trace.IsFallback = true
		trace.FallbackReason = fmt.Sprintf("inference error: %v", err)
	} else if len(tokens) >= 4 && r.policy.MinUniqueTokenRatio > 0.0 && uniqueRatio < r.policy.MinUniqueTokenRatio {
		trace.IsFallback = true
		trace.FallbackReason = fmt.Sprintf("repetitive pattern detected (unique token ratio %.2f < %.2f)", uniqueRatio, r.policy.MinUniqueTokenRatio)
	} else if len(tokens) >= 2 && r.policy.MaxSingleCharRatio > 0.0 && singleRatio >= r.policy.MaxSingleCharRatio {
		trace.IsFallback = true
		trace.FallbackReason = fmt.Sprintf("unlearned vocabulary (single-char ratio %.2f >= %.2f)", singleRatio, r.policy.MaxSingleCharRatio)
	} else if unkRatio >= 0.5 {
		trace.IsFallback = true
		trace.FallbackReason = fmt.Sprintf("excessive unknown tokens (%.2f >= 0.50)", unkRatio)
	} else if effectiveMinEnergy != 0.0 && energy < effectiveMinEnergy {
		trace.IsFallback = true
		trace.FallbackReason = fmt.Sprintf("energy %.4f below minimum threshold %.4f (OOD)", energy, effectiveMinEnergy)
	} else if entropy > r.policy.MaxEntropy {
		trace.IsFallback = true
		trace.FallbackReason = fmt.Sprintf("prediction entropy %.4f exceeds limit %.4f (OOD)", entropy, r.policy.MaxEntropy)
	} else if trace.Confidence < r.policy.LowThreshold {
		trace.IsFallback = true
		trace.FallbackReason = fmt.Sprintf("confidence %.4f below low threshold %.4f", trace.Confidence, r.policy.LowThreshold)
	} else {
		if secondLabel != "" && calibratedSecond >= r.policy.PipelineThreshold {
			trace.IsPipeline = true
		}
		if trace.Confidence < r.policy.HighThreshold || margin < r.policy.MarginCutoff || (effectiveMargin > 0.0 && trace.LogitMargin < effectiveMargin) || (trace.CoActiveCount >= 2 && effectiveMargin > 0.0 && trace.LogitMargin < effectiveMargin*1.5) {
			trace.IsAmbiguous = true
		}
	}

	return trace
}

// -------------------------------------------------------------
// Native Language Branching Primitives (Zero Learning Curve)
// -------------------------------------------------------------

// Select returns the verified intent label for direct usage in native Go switch-case statements.
// If the input fails safety guards (OOD, high entropy, low margin) or query errors occur,
// it returns an empty string (""), cleanly falling through to the native default: branch.
func (r *Router) Select(text string) string {
	return r.SelectCtx(context.Background(), text)
}

// SelectCtx returns the verified intent label with explicit context propagation for native switch-case statements.
func (r *Router) SelectCtx(ctx context.Context, text string) string {
	decision, err := r.RouteQuery(ctx, text)
	if err != nil {
		return ""
	}

	effectiveMinEnergy := r.policy.MinLogSumExp
	model := r.model.Load()
	if effectiveMinEnergy == 0 && model != nil && model.Header.CalibratedMinEnergy > 0 {
		effectiveMinEnergy = float64(model.Header.CalibratedMinEnergy)
	}

	// Reinforce safety cutoffs: LogSumExp Free Energy and Shannon entropy
	if effectiveMinEnergy > 0.0 && decision.Energy < effectiveMinEnergy {
		return ""
	}
	if decision.Entropy > r.policy.MaxEntropy {
		return ""
	}

	return decision.Intent
}

// Is returns whether the input text matches the targetLabel with confidence exceeding high threshold (or explicit threshold).
func (r *Router) Is(text string, targetLabel string, threshold ...float64) bool {
	return r.IsCtx(context.Background(), text, targetLabel, threshold...)
}

// If evaluates whether the input text matches targetLabel, providing an intuitive conditional keyword for if statements.
func (r *Router) If(text string, targetLabel string, threshold ...float64) bool {
	return r.Is(text, targetLabel, threshold...)
}

// IfCtx evaluates whether the input text matches targetLabel with context cancellation support.
func (r *Router) IfCtx(ctx context.Context, text string, targetLabel string, threshold ...float64) bool {
	return r.IsCtx(ctx, text, targetLabel, threshold...)
}

// IsCtx evaluates whether the input text matches targetLabel with context cancellation support.
func (r *Router) IsCtx(ctx context.Context, text string, targetLabel string, threshold ...float64) bool {
	decision, err := r.RouteQuery(ctx, text)
	if err != nil {
		if errors.Is(err, ErrAmbiguousIntent) || errors.Is(err, ErrLowConfidence) {
			if len(threshold) > 0 && threshold[0] > 0.0 {
				return decision.Intent == targetLabel && decision.Confidence >= threshold[0]
			}
		}
		return false
	}
	if decision.Intent != targetLabel {
		return false
	}
	if len(threshold) > 0 && threshold[0] > 0.0 {
		return decision.Confidence >= threshold[0]
	}
	return true
}

// Match provides Go's native comma-ok idiom for intent evaluation.
// Returns (intent, true) on confident match; (intent, false) on ambiguous match; ("", false) on OOD/error.
func (r *Router) Match(text string, threshold ...float64) (string, bool) {
	return r.MatchCtx(context.Background(), text, threshold...)
}

// MatchCtx provides Go's native comma-ok idiom with context propagation.
func (r *Router) MatchCtx(ctx context.Context, text string, threshold ...float64) (string, bool) {
	decision, err := r.RouteQuery(ctx, text)
	if err != nil {
		if errors.Is(err, ErrAmbiguousIntent) || errors.Is(err, ErrLowConfidence) {
			if len(threshold) > 0 && threshold[0] > 0.0 && decision.Confidence >= threshold[0] {
				return decision.Intent, true
			}
			return decision.Intent, false
		}
		return "", false
	}
	if len(threshold) > 0 && threshold[0] > 0.0 {
		return decision.Intent, decision.Confidence >= threshold[0]
	}
	return decision.Intent, true
}

// Assert checks if the query strictly matches targetLabel, returning sentinel errors on failure.
func (r *Router) Assert(text string, targetLabel string) error {
	return r.AssertCtx(context.Background(), text, targetLabel)
}

// AssertCtx checks if the query strictly matches targetLabel with context cancellation support.
func (r *Router) AssertCtx(ctx context.Context, text string, targetLabel string) error {
	decision, err := r.RouteQuery(ctx, text)
	if err != nil {
		return err
	}
	if decision.Intent != targetLabel {
		return ErrOutOfDomain
	}
	return nil
}

