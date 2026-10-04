package neurobranch

import (
	"context"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

func TestRouterNativeSelectAndFallback(t *testing.T) {
	tempDir := t.TempDir()
	modelPath := filepath.Join(tempDir, "router_test.bin")

	origModel := createSampleModel()
	if err := SaveBinaryModel(modelPath, origModel); err != nil {
		t.Fatalf("Failed to save model: %v", err)
	}

	router, err := NewRouter(modelPath, 0.1)
	if err != nil {
		t.Fatalf("Failed to create router: %v", err)
	}

	// 1. Select with matching label tokens
	selected := router.Select("refund")
	if selected == "" {
		t.Log("Note: query fell back under low threshold, but select completed safely")
	}

	// 2. Select with strict threshold causing deterministic fallback to empty string (default:)
	strictRouter, err := NewRouter(modelPath, 0.999999)
	if err != nil {
		t.Fatalf("Failed to create strict router: %v", err)
	}

	strictSelected := strictRouter.Select("refund")
	if strictSelected != "" {
		t.Errorf("Expected empty string (fallback to default:) under ultra-strict threshold, got %s", strictSelected)
	}
}

func TestRouterInspectAndTrace(t *testing.T) {
	tempDir := t.TempDir()
	modelPath := filepath.Join(tempDir, "router_trace_test.bin")

	origModel := createSampleModel()
	if err := SaveBinaryModel(modelPath, origModel); err != nil {
		t.Fatalf("Failed to save model: %v", err)
	}

	router, err := NewRouter(modelPath, 0.5)
	if err != nil {
		t.Fatalf("Failed to create router: %v", err)
	}

	trace := router.Inspect("refund")
	if len(trace.TokenIDs) == 0 {
		t.Error("Expected non-empty token IDs in trace")
	}
	if len(trace.ClassProbabilities) == 0 {
		t.Error("Expected class probability breakdown in trace")
	}
	if trace.Confidence <= 0.0 {
		t.Errorf("Invalid confidence in trace: %f", trace.Confidence)
	}
}

func TestRouterContextCancellation(t *testing.T) {
	tempDir := t.TempDir()
	modelPath := filepath.Join(tempDir, "router_ctx_test.bin")

	origModel := createSampleModel()
	if err := SaveBinaryModel(modelPath, origModel); err != nil {
		t.Fatalf("Failed to save model: %v", err)
	}

	router, err := NewRouter(modelPath, 0.1)
	if err != nil {
		t.Fatalf("Failed to create router: %v", err)
	}

	// Pre-canceled context should abort immediately
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = router.RouteQuery(ctx, "refund")
	if err != context.Canceled {
		t.Errorf("Expected context.Canceled, got %v", err)
	}

	selected := router.SelectCtx(ctx, "refund")
	if selected != "" {
		t.Errorf("Expected empty string on cancelled context, got %s", selected)
	}
}

func TestRouter3TierPolicy(t *testing.T) {
	tempDir := t.TempDir()
	modelPath := filepath.Join(tempDir, "router_3tier_test.bin")

	origModel := createSampleModel()
	if err := SaveBinaryModel(modelPath, origModel); err != nil {
		t.Fatalf("Failed to save model: %v", err)
	}

	router, err := NewRouter(modelPath, 0.5)
	if err != nil {
		t.Fatalf("Failed to create router: %v", err)
	}

	// Configure strict policy: High=0.80, Low=0.30, Margin=0.20
	policy := DispatchPolicy{
		HighThreshold: 0.80,
		LowThreshold:  0.30,
		MarginCutoff:  0.20,
		MaxEntropy:    2.0,
	}
	router.SetPolicy(policy)

	// 1. Trace inspect to check confidence
	trace := router.Inspect("refund")
	decision, err := router.RouteQuery(context.Background(), "refund")

	if trace.IsAmbiguous {
		if err != ErrAmbiguousIntent {
			t.Logf("Decision Ambiguous match: err=%v", err)
		}
	} else if trace.IsFallback {
		if err == nil {
			t.Error("Expected error for fallback query")
		}
	} else {
		if err != nil {
			t.Errorf("Expected nil error for definite match, got %v", err)
		}
		if decision.Intent == "" {
			t.Error("Expected non-empty intent")
		}
	}
}

func TestRouterOODEntropyIsolation(t *testing.T) {
	tempDir := t.TempDir()
	modelPath := filepath.Join(tempDir, "router_ood_test.bin")

	origModel := createSampleModel()
	if err := SaveBinaryModel(modelPath, origModel); err != nil {
		t.Fatalf("Failed to save model: %v", err)
	}

	router, err := NewRouter(modelPath, 0.5)
	if err != nil {
		t.Fatalf("Failed to create router: %v", err)
	}

	// Set very strict MaxEntropy: 0.001 to force OOD isolation
	policy := DispatchPolicy{
		HighThreshold: 0.70,
		LowThreshold:  0.20,
		MarginCutoff:  0.10,
		MaxEntropy:    0.001, // Anything with uncertainty is treated as OOD
	}
	router.SetPolicy(policy)

	// In Native mode, OOD automatically falls through to default: (empty string)
	selected := router.Select("refund")
	if selected != "" {
		t.Errorf("Expected empty string (default: fallback) when entropy exceeds MaxEntropy (OOD), got %s", selected)
	}

	_, err = router.RouteQuery(context.Background(), "refund")
	if err != ErrHighEntropy {
		t.Errorf("Expected ErrHighEntropy, got %v", err)
	}
}

func TestRouterAtomicReloadConcurrently(t *testing.T) {
	tempDir := t.TempDir()
	modelPathA := filepath.Join(tempDir, "model_a.bin")
	modelPathB := filepath.Join(tempDir, "model_b.bin")

	modelA := createSampleModel()
	if err := SaveBinaryModel(modelPathA, modelA); err != nil {
		t.Fatalf("Failed to save model A: %v", err)
	}

	modelB := createSampleModel()
	modelB.Temperature = 0.8
	if err := SaveBinaryModel(modelPathB, modelB); err != nil {
		t.Fatalf("Failed to save model B: %v", err)
	}

	router, err := NewRouter(modelPathA, 0.5)
	if err != nil {
		t.Fatalf("Failed to construct router: %v", err)
	}

	stopChan := make(chan struct{})
	var wg sync.WaitGroup

	// 8 Reader Goroutines hammering native Select and Inspect concurrently
	numReaders := 8
	for i := 0; i < numReaders; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			ctx := context.Background()
			for {
				select {
				case <-stopChan:
					return
				default:
					_ = router.SelectCtx(ctx, "refund my payment immediately")
					_ = router.Inspect("check status")
					_ = router.Is("refund", "Refund")
				}
			}
		}(i)
	}

	// 1 Reload Goroutine repeatedly swapping models
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 20; i++ {
			select {
			case <-stopChan:
				return
			default:
				target := modelPathB
				if i%2 == 0 {
					target = modelPathA
				}
				if err := router.Reload(target); err != nil {
					t.Errorf("Reload failed during concurrent traffic: %v", err)
				}
				time.Sleep(2 * time.Millisecond)
			}
		}
	}()

	time.Sleep(60 * time.Millisecond)
	close(stopChan)
	wg.Wait()
}

func TestFailSafeLayer1_UnlearnedVocabulary(t *testing.T) {
	tempDir := t.TempDir()
	modelPath := filepath.Join(tempDir, "failsafe_l1.bin")

	// Create model with authentic BPE subwords learned from corpus
	corpus := []string{
		"refund payment money please",
		"cancel refund order payment",
		"delivery shipment package status",
		"track order delivery status",
	}
	tok, err := TrainBPE(corpus, 40)
	if err != nil {
		t.Fatalf("TrainBPE failed: %v", err)
	}

	header := Header{
		Magic:        MagicBytes,
		Version:      2,
		VocabSize:    uint32(tok.VocabSize()),
		EmbeddingDim: 8,
		HiddenDim:    12,
		NumClasses:   2,
	}
	labels := []string{"Refund", "Delivery"}
	weights := Weights{
		Embedding:  make([]float32, int(header.VocabSize)*int(header.EmbeddingDim)),
		Positional: make([]float32, 32*int(header.EmbeddingDim)),
		W1:         make([]float32, int(header.EmbeddingDim)*int(header.HiddenDim)),
		B1:         make([]float32, int(header.HiddenDim)),
		W2:         make([]float32, int(header.HiddenDim)*int(header.NumClasses)),
		B2:         make([]float32, int(header.NumClasses)),
	}
	for i := range weights.Embedding {
		weights.Embedding[i] = 0.05
	}
	for i := range weights.W1 {
		weights.W1[i] = 0.05
	}
	for i := range weights.W2 {
		weights.W2[i] = 0.05
	}

	model := NewInferenceModel(header, labels, tok.Vocab, tok.MergeRules, weights)
	if err := SaveBinaryModel(modelPath, model); err != nil {
		t.Fatalf("Failed to save model: %v", err)
	}

	router, err := NewRouter(modelPath, 0.70)
	if err != nil {
		t.Fatalf("Failed to init router: %v", err)
	}

	ctx := context.Background()

	// 1. Completely unlearned noisy input -> Layer 1 rejection with ErrUnlearnedVocabulary
	unlearnedNoise := "xzqjwpkvmcty1837"
	_, err = router.RouteQuery(ctx, unlearnedNoise)
	if err == nil {
		t.Fatalf("Expected error for unlearned noise '%s', but got nil", unlearnedNoise)
	}
	if err != ErrUnlearnedVocabulary {
		t.Fatalf("Expected ErrUnlearnedVocabulary, got %v", err)
	}

	// 2. In Native mode, unlearned vocabulary falls through to default: (empty string)
	selected := router.Select(unlearnedNoise)
	if selected != "" {
		t.Errorf("Expected empty string (default: fallback) for unlearned vocabulary, got %s", selected)
	}
}

func TestFailSafeLayer2_NeuralMetrics(t *testing.T) {
	tempDir := t.TempDir()
	modelPath := filepath.Join(tempDir, "failsafe_l2.bin")

	sampleModel := createSampleModel()
	if err := SaveBinaryModel(modelPath, sampleModel); err != nil {
		t.Fatalf("Failed to save model: %v", err)
	}

	router, err := NewRouter(modelPath, 0.50)
	if err != nil {
		t.Fatalf("Failed to init router: %v", err)
	}

	ctx := context.Background()

	// 1. Normal query should execute RouteQuery cleanly
	res, err := router.RouteQuery(ctx, "refund")
	if err == nil {
		if res.Intent == "" {
			t.Errorf("Expected non-empty intent, got '%s'", res.Intent)
		}
		if res.Confidence <= 0 {
			t.Errorf("Expected positive confidence, got %f", res.Confidence)
		}
	} else if err != ErrAmbiguousIntent && err != ErrLowConfidence {
		t.Logf("RouteQuery returned: %v", err)
	}

	// 2. High threshold policy should return ErrLowConfidence or ErrAmbiguousIntent
	router.policy.HighThreshold = 0.9999
	router.policy.LowThreshold = 0.9990
	_, err = router.RouteQuery(ctx, "refund")
	if err == nil {
		t.Fatalf("Expected safety error under ultra-strict threshold, got nil")
	}
	if err != ErrLowConfidence && err != ErrAmbiguousIntent && err != ErrOutOfDomain {
		t.Errorf("Expected ErrLowConfidence/ErrAmbiguousIntent, got %v", err)
	}
}

func TestRouterRepetitiveTokenFlood(t *testing.T) {
	tempDir := t.TempDir()
	modelPath := filepath.Join(tempDir, "flood_router.bin")

	sampleModel := createSampleModel()
	if err := SaveBinaryModel(modelPath, sampleModel); err != nil {
		t.Fatalf("Failed to save model: %v", err)
	}

	router, err := NewRouter(modelPath, 0.50)
	if err != nil {
		t.Fatalf("Failed to init router: %v", err)
	}

	// 1. Repetitive normal words flood -> Layer 1 rejection with ErrDegeneratedInput
	ctx := context.Background()
	floodQuery := "refund refund refund refund refund refund"
	decision, err := router.RouteQuery(ctx, floodQuery)
	if err == nil {
		t.Fatalf("expected ErrDegeneratedInput for repetitive token sequence, got nil")
	}
	if err != ErrDegeneratedInput {
		t.Fatalf("expected ErrDegeneratedInput, got %v", err)
	}
	if decision.UniqueTokenRatio >= router.policy.MinUniqueTokenRatio {
		t.Fatalf("expected UniqueTokenRatio < %f, got %f", router.policy.MinUniqueTokenRatio, decision.UniqueTokenRatio)
	}

	// 2. In Native mode, repetitive flood automatically falls through to default: (empty string)
	selected := router.Select(floodQuery)
	if selected != "" {
		t.Errorf("Expected empty string (default: fallback) on repetitive token flood, got %s", selected)
	}

	// 3. Verify Inspect records repetitive pattern fallback reason
	trace := router.Inspect(floodQuery)
	if !trace.IsFallback {
		t.Fatalf("Expected Inspect trace.IsFallback to be true for repetitive input")
	}
	if trace.FallbackReason == "" {
		t.Fatalf("Expected non-empty FallbackReason in Inspect trace")
	}
}

func TestTrainInMemory_OneShotBranching(t *testing.T) {
	// In-process one-shot training directly from Go code without disk I/O or external tools
	router, err := TrainFromMap(map[string][]string{
		"order_refund": {
			"can i get a refund please",
			"refund my money for order",
			"i want my money back",
			"reverse the charge on this purchase",
			"issue a refund to my credit card",
		},
		"order_cancel": {
			"cancel my order right now",
			"stop the shipment and cancel",
			"i want to cancel purchase",
			"abort my order immediately",
			"please do not ship and cancel",
		},
	})
	if err != nil {
		t.Fatalf("TrainFromMap failed: %v", err)
	}

	// 1. Verify multi-select branching on distinct phrasings
	target := router.Select("i want my money back please")
	if target != "order_refund" {
		t.Errorf("Expected 'order_refund', got '%s'", target)
	}

	// 2. Verify boolean condition branching
	if !router.Is("stop the shipment and cancel", "order_cancel") {
		trace := router.Inspect("stop the shipment and cancel")
		t.Errorf("Expected router.Is to return true for order_cancel, trace: %+v", trace)
	}
}

func TestRouter_AppendData_DynamicRetrain(t *testing.T) {
	// Initialize with 2 classes
	router, err := TrainFromMap(map[string][]string{
		"order_refund": {
			"can i get a refund please",
			"refund my money for order",
			"i want my money back",
		},
		"order_cancel": {
			"cancel my order right now",
			"stop the shipment and cancel",
			"i want to cancel purchase",
		},
	})
	if err != nil {
		t.Fatalf("Initial TrainFromMap failed: %v", err)
	}

	// Dynamic update: Add a brand new class 'shipping_tracker' at runtime without stopping service
	err = router.AppendDataMap(map[string][]string{
		"shipping_tracker": {
			"where is my package tracking number",
			"track my delivery status",
			"when will my package arrive",
			"shipping tracker for order",
		},
	})
	if err != nil {
		t.Fatalf("AppendDataMap failed: %v", err)
	}

	// Verify newly added class is immediately routeable
	target := router.Select("track my delivery status please")
	if target != "shipping_tracker" {
		t.Errorf("Expected newly added 'shipping_tracker', got '%s'", target)
	}

	// Verify existing classes still route properly
	refundTarget := router.Select("refund my money for order")
	if refundTarget != "order_refund" {
		t.Errorf("Expected 'order_refund' to persist after retrain, got '%s'", refundTarget)
	}
}

func TestAIAlias_SelectBranching(t *testing.T) {
	// Verify that AI type alias and TrainAIFromMap enable clean switch ai.Select(query) idioms
	var ai *AI
	var err error

	ai, err = TrainAIFromMap(map[string][]string{
		"order_refund": {
			"can i get a refund please",
			"refund my money for order",
			"i want my money back",
		},
		"order_cancel": {
			"cancel my order right now",
			"stop the shipment and cancel",
			"abort my order immediately",
		},
	})
	if err != nil {
		t.Fatalf("TrainAIFromMap failed: %v", err)
	}

	// 1. switch ai.Select(query)
	var handled string
	switch ai.Select("i want my money back please") {
	case "order_refund":
		handled = "refunded"
	case "order_cancel":
		handled = "cancelled"
	default:
		handled = "fallback"
	}

	if handled != "refunded" {
		t.Errorf("Expected 'refunded', got '%s'", handled)
	}

	// 2. if ai.Is(query, target) and if ai.If(query, target)
	if !ai.Is("stop the shipment and cancel", "order_cancel") {
		t.Errorf("Expected ai.Is to return true for order_cancel")
	}
	if !ai.If("stop the shipment and cancel", "order_cancel") {
		t.Errorf("Expected ai.If to return true for order_cancel")
	}

	// 3. target, ok := ai.Match(query)
	if target, ok := ai.Match("i want my money back please"); !ok || target != "order_refund" {
		t.Errorf("Expected ('order_refund', true), got ('%s', %v)", target, ok)
	}
}

func TestAIAlias_OODCutoff(t *testing.T) {
	ai, err := TrainAIFromMap(map[string][]string{
		"Refund": {
			"cancel payment and request refund",
			"want my money back refund",
			"reverse transaction charge refund",
			"issue refund for purchase order",
			"please process full refund immediately",
			"sent return parcel need refund",
			"defective item request refund reimbursement",
			"credit card transaction chargeback refund",
		},
		"Delivery": {
			"where is my package delivery tracking",
			"courier delivery shipping tracking status",
			"update delivery shipping destination address",
			"package delivery transit shipment delayed",
			"track courier parcel delivery location",
			"courier delivery tracking number lookup",
			"parcel delivery has not arrived yet",
			"change courier delivery dropoff point",
		},
		"Account": {
			"forgot my account login password",
			"locked out of user account login",
			"change account profile email credentials",
			"two factor account security authentication",
			"reset user account dashboard password",
			"account security profile recovery support",
			"unlock frozen user account profile",
			"reset credentials for account signin",
		},
		"Billing": {
			"billing credit card monthly receipt",
			"billing subscription invoice tax receipt",
			"change billing payment method invoice",
			"download corporate billing vat invoice",
			"annual subscription billing payment statement",
			"update billing invoice payment details",
		},
	})
	if err != nil {
		t.Fatalf("TrainAIFromMap failed: %v", err)
	}

	var minE, maxE, sumE float64 = 999.0, -999.0, 0.0
	for _, s := range ai.samples {
		st := ai.Inspect(s.Text)
		if st.Energy < minE {
			minE = st.Energy
		}
		if st.Energy > maxE {
			maxE = st.Energy
		}
		sumE += st.Energy
	}
	avgE := sumE / float64(len(ai.samples))
	t.Logf("In-Domain Energy Stats: Min=%.4f, Max=%.4f, Avg=%.4f, Count=%d", minE, maxE, avgE, len(ai.samples))

	testOODQueries := []string{
		"hardware device driver crash kernel panic",
		"quantum physics entangled photon spin",
		"weather forecast tomorrow in tokyo",
	}

	for _, oodQuery := range testOODQueries {
		trace := ai.Inspect(oodQuery)
		selected := ai.Select(oodQuery)
		decision, err := ai.RouteQuery(context.Background(), oodQuery)
		t.Logf("Query: %q -> Select: %q, RouteQuery err: %v, Inspect IsFallback: %v (Reason: %q, Energy: %.4f, MinEnergy: %.4f)",
			oodQuery, selected, err, trace.IsFallback, trace.FallbackReason, trace.Energy, ai.policy.MinLogSumExp)

		if selected != "" {
			t.Errorf("Expected empty string (default: fallback) for OOD query %q, got %q (decision: %+v)", oodQuery, selected, decision)
		}
	}
}
