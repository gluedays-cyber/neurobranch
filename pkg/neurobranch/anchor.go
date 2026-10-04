package neurobranch

import (
	"strings"
)

const (
	// DefaultMaxAnchorBoost defines the default maximum cumulative logit boost per class (prevents logit explosion).
	DefaultMaxAnchorBoost float32 = 3.0
)

// AnchorRule defines a symbolic soft-bias injected into a specific class logit upon bitmask match.
type AnchorRule struct {
	ClassIndex     int
	Mask           uint64
	Weight         float32
	InhibitClasses []int
	Penalty        float32
	Keywords       []string
}

// BranchRouteBuilder provides fluent API chaining for binding routes and anchor soft biases.
type BranchRouteBuilder struct {
	branch     *NeuroBranch
	classIndex int
	label      string
}

// GateRouteBuilder is an alias for BranchRouteBuilder for backward compatibility.
type GateRouteBuilder = BranchRouteBuilder

// WithAnchor registers anchor keywords that inject a soft additive bias into this class's logit.
func (b *BranchRouteBuilder) WithAnchor(weight float32, keywords ...string) *BranchRouteBuilder {
	b.branch.mu.Lock()
	defer b.branch.mu.Unlock()

	var mask uint64 = 0
	var cleanKeywords []string
	for _, kw := range keywords {
		kw = strings.TrimSpace(strings.ToLower(kw))
		if kw == "" {
			continue
		}
		cleanKeywords = append(cleanKeywords, kw)
		m, exists := b.branch.anchorDict[kw]
		if !exists {
			if len(b.branch.anchorDict) < 64 {
				m = 1 << uint64(len(b.branch.anchorDict))
				b.branch.anchorDict[kw] = m
			}
		}
		mask |= m
	}

	model := b.branch.model.Load()
	if model != nil && model.Tokenizer != nil {
		for _, kw := range cleanKeywords {
			toks := model.Tokenizer.Encode(kw)
			for _, tid := range toks {
				b.branch.anchorTokenMap[tid] |= mask
			}
		}
	}

	b.branch.anchorRules = append(b.branch.anchorRules, AnchorRule{
		ClassIndex: b.classIndex,
		Mask:       mask,
		Weight:     weight,
		Keywords:   cleanKeywords,
	})
	return b
}

// Inhibit registers competing class labels to penalize when this anchor triggers.
func (b *BranchRouteBuilder) Inhibit(penalty float32, competingLabels ...string) *BranchRouteBuilder {
	b.branch.mu.Lock()
	defer b.branch.mu.Unlock()

	if len(b.branch.anchorRules) == 0 {
		return b
	}
	lastIdx := len(b.branch.anchorRules) - 1
	rule := &b.branch.anchorRules[lastIdx]

	for _, lbl := range competingLabels {
		if cIdx, exists := b.branch.labelToIndex[lbl]; exists {
			rule.InhibitClasses = append(rule.InhibitClasses, cIdx)
		}
	}
	rule.Penalty = penalty
	return b
}
