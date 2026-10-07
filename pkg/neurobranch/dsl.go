package neurobranch

import (
	"context"
	"errors"
	"fmt"
)

// Sentinel errors for declarative DSL routing
var (
	ErrUnhandledIntent = errors.New("neurobranch: no matching case and no fallback specified")
	ErrConfirmationReq = errors.New("neurobranch: confirmation required for ambiguous intent")
)

// IntentHandler defines a standard execution callback for matched branches.
type IntentHandler func(ctx context.Context) error

// ConfirmationHandler handles interactive or deferred human-in-the-loop confirmation.
type ConfirmationHandler func(ctx context.Context, prompt string) error

// CaseClause captures declarative branching criteria and thresholds for a specific intent label.
type CaseClause struct {
	label      string
	autoFn     IntentHandler
	confirmMsg string
	confirmFn  ConfirmationHandler
	minConf    float64
}

// SwitchBuilder constructs and evaluates declarative semantic switch blocks.
type SwitchBuilder struct {
	router      *Router
	ctx         context.Context
	text        string
	cases       map[string]*CaseClause
	currentCase *CaseClause
	fallbackFn  IntentHandler
}

// Switch initiates a declarative switch builder bound to a Router instance.
func (r *Router) Switch(text string) *SwitchBuilder {
	return r.SwitchCtx(context.Background(), text)
}

// SwitchCtx initiates a declarative switch builder with an explicit context.
func (r *Router) SwitchCtx(ctx context.Context, text string) *SwitchBuilder {
	return &SwitchBuilder{
		router: r,
		ctx:    ctx,
		text:   text,
		cases:  make(map[string]*CaseClause),
	}
}

// Case declares a target semantic label branch.
func (sb *SwitchBuilder) Case(label string) *SwitchBuilder {
	clause := &CaseClause{
		label:   label,
		minConf: sb.router.policy.HighThreshold,
	}
	sb.cases[label] = clause
	sb.currentCase = clause
	return sb
}

// AtLeast sets an explicit confidence threshold for the preceding Case clause.
func (sb *SwitchBuilder) AtLeast(threshold float64) *SwitchBuilder {
	if sb.currentCase != nil && threshold > 0.0 {
		sb.currentCase.minConf = threshold
	}
	return sb
}

// Auto attaches the decisive execution handler executed when confidence meets or exceeds criteria.
func (sb *SwitchBuilder) Auto(fn IntentHandler) *SwitchBuilder {
	if sb.currentCase != nil {
		sb.currentCase.autoFn = fn
	}
	return sb
}

// Confirm registers an interactive confirmation prompt and optional confirmation handler for borderline states.
func (sb *SwitchBuilder) Confirm(prompt string, fn ...ConfirmationHandler) *SwitchBuilder {
	if sb.currentCase != nil {
		sb.currentCase.confirmMsg = prompt
		if len(fn) > 0 {
			sb.currentCase.confirmFn = fn[0]
		}
	}
	return sb
}

// Default registers the catch-all fallback handler for OOD, low confidence, or unhandled intents.
func (sb *SwitchBuilder) Default(fn IntentHandler) *SwitchBuilder {
	sb.fallbackFn = fn
	return sb
}

// Evaluate performs neural routing and dispatches execution according to matched Case rules.
// If an explicit context is provided, it overrides any context set during SwitchCtx.
func (sb *SwitchBuilder) Evaluate(ctx ...context.Context) error {
	evalCtx := sb.ctx
	if len(ctx) > 0 && ctx[0] != nil {
		evalCtx = ctx[0]
	}

	decision, err := sb.router.RouteQuery(evalCtx, sb.text)
	if err != nil {
		// Borderline / Ambiguous intent check
		if errors.Is(err, ErrAmbiguousIntent) || errors.Is(err, ErrLowConfidence) {
			if clause, exists := sb.cases[decision.Intent]; exists {
				if clause.confirmFn != nil || clause.confirmMsg != "" {
					if clause.confirmFn != nil {
						return clause.confirmFn(evalCtx, clause.confirmMsg)
					}
					return fmt.Errorf("%w: %s", ErrConfirmationReq, clause.confirmMsg)
				}
			}
		}

		if sb.fallbackFn != nil {
			return sb.fallbackFn(evalCtx)
		}
		return err
	}

	clause, exists := sb.cases[decision.Intent]
	if !exists {
		if sb.fallbackFn != nil {
			return sb.fallbackFn(evalCtx)
		}
		return ErrUnhandledIntent
	}

	// 1. Confident Auto Branch
	if decision.Confidence >= clause.minConf {
		if clause.autoFn != nil {
			return clause.autoFn(evalCtx)
		}
		// If Auto handler is omitted but Confirm is declared, invoke confirmation
		if clause.confirmFn != nil || clause.confirmMsg != "" {
			if clause.confirmFn != nil {
				return clause.confirmFn(evalCtx, clause.confirmMsg)
			}
			return fmt.Errorf("%w: %s", ErrConfirmationReq, clause.confirmMsg)
		}
		// If neither handler is defined, fall back to Default
		if sb.fallbackFn != nil {
			return sb.fallbackFn(evalCtx)
		}
		return ErrUnhandledIntent
	}

	// 2. Ambiguous / Threshold-unmet Confirmation Branch
	if clause.confirmFn != nil || clause.confirmMsg != "" {
		if clause.confirmFn != nil {
			return clause.confirmFn(evalCtx, clause.confirmMsg)
		}
		return fmt.Errorf("%w: %s", ErrConfirmationReq, clause.confirmMsg)
	}

	// 3. Fallback when below threshold and no confirmation provided
	if sb.fallbackFn != nil {
		return sb.fallbackFn(evalCtx)
	}

	return ErrLowConfidence
}
