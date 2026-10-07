package neurobranch

import (
	"context"
	"strings"
	"testing"
)

// TestImprovement1_EnergyBasedOODCutoff verifies that general knowledge and out-of-domain queries
// are strictly blocked by Free Energy (LogSumExp) threshold instead of falsely leaking into business branches.
func TestImprovement1_EnergyBasedOODCutoff(t *testing.T) {
	ai, err := TrainAIFromMap(map[string][]string{
		"Delivery": {
			"where is my package delivery tracking",
			"courier delivery shipping tracking status",
			"package delivery transit shipment delayed",
			"track courier parcel delivery location",
			"courier delivery tracking number lookup",
			"parcel delivery has not arrived yet",
		},
		"Refund": {
			"cancel payment and request refund",
			"want my money back refund",
			"reverse transaction charge refund",
			"issue refund for purchase order",
			"defective item request refund reimbursement",
		},
	})
	if err != nil {
		t.Fatalf("TrainAIFromMap failed: %v", err)
	}

	// Set strict EnergyThreshold (cutoff at 3.0)
	ai.SetEnergyThreshold(3.0)

	leakQueries := []string{
		"tell me the capital city of france",
		"the weather is nice today in paris",
		"what is the distance between earth and mars",
	}

	for _, query := range leakQueries {
		branch := ai.Select(query)
		trace := ai.Inspect(query)
		t.Logf("OOD Query: %q -> Branch: %q, Energy: %.4f, Threshold: %.4f, FallbackReason: %q",
			query, branch, trace.Energy, ai.policy.EnergyThreshold, trace.FallbackReason)

		if branch != "" {
			t.Errorf("Expected OOD query %q to be blocked (empty branch), but leaked to: %q (energy: %.4f)",
				query, branch, trace.Energy)
		}

		_, routeErr := ai.RouteQuery(context.Background(), query)
		if routeErr == nil {
			t.Errorf("Expected RouteQuery to return error for OOD query %q, got nil", query)
		}
	}
}

// TestImprovement2_DSLChainSilentDropFix verifies that when .Auto() is omitted and only .Confirm() is declared,
// the Switch chain never silently drops the query and executes Confirm or Default properly.
func TestImprovement2_DSLChainSilentDropFix(t *testing.T) {
	router := setupTestRouter(t)
	ctx := context.Background()

	// Scenario: Case "Refund" has Confirm handler but NO Auto handler
	var confirmCalled bool
	var defaultCalled bool

	err := router.Switch("refund my money please").
		Case("Refund").
		Confirm("Would you like to process a refund?", func(ctx context.Context, prompt string) error {
			confirmCalled = true
			if !strings.Contains(prompt, "refund") {
				t.Errorf("unexpected prompt: %s", prompt)
			}
			return nil
		}).
		Default(func(ctx context.Context) error {
			defaultCalled = true
			return nil
		}).
		Evaluate(ctx)

	if err != nil {
		t.Fatalf("unexpected evaluate error: %v", err)
	}
	if !confirmCalled {
		t.Errorf("Expected Confirm callback to be triggered when Auto is omitted, but was silently dropped")
	}
	if defaultCalled {
		t.Errorf("Default handler should not have been called when Confirm is handled")
	}

	// Scenario 2: Unhandled intent falls back to Default
	confirmCalled = false
	defaultCalled = false

	err = router.Switch("what is the weather today").
		Case("Refund").
		Confirm("Confirm refund?", func(ctx context.Context, prompt string) error {
			confirmCalled = true
			return nil
		}).
		Default(func(ctx context.Context) error {
			defaultCalled = true
			return nil
		}).
		Evaluate(ctx)

	if err != nil {
		t.Fatalf("unexpected evaluate error: %v", err)
	}
	if !defaultCalled {
		t.Errorf("Expected Default handler to be called for OOD query")
	}
}

// TestImprovement3_ClassImbalanceAutoCorrection verifies that adding a tiny minority class
// during runtime hot-swap (AppendDataMap) is automatically oversampled to maintain high confidence and accuracy.
func TestImprovement3_ClassImbalanceAutoCorrection(t *testing.T) {
	// Initialize with substantial base classes
	initialData := map[string][]string{
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
	}

	ai, err := TrainAIFromMap(initialData)
	if err != nil {
		t.Fatalf("initial TrainAIFromMap failed: %v", err)
	}

	// Hot-swap append tiny minority class (TechSupport) with only 3 samples
	minorityData := map[string][]string{
		"TechSupport": {
			"blue screen system crash error",
			"device driver hardware failure",
			"operating system kernel panic reboot",
		},
	}

	cfg := DefaultTrainConfig()
	cfg.AutoBalance = true
	cfg.Epochs = 60

	if err := ai.AppendDataMap(minorityData, cfg); err != nil {
		t.Fatalf("AppendDataMap failed: %v", err)
	}

	// Test query for newly appended minority class
	testQuery := "system kernel panic crash error"
	selected := ai.Select(testQuery)
	trace := ai.Inspect(testQuery)
	t.Logf("Minority Class Query: %q -> Selected: %q, Conf: %.2f, Reason: %q",
		testQuery, selected, trace.Confidence, trace.FallbackReason)

	if selected != "TechSupport" {
		t.Errorf("Expected minority class 'TechSupport', got %q (confidence: %.2f, reason: %s)",
			selected, trace.Confidence, trace.FallbackReason)
	}
	if trace.Confidence < 0.60 {
		t.Errorf("Expected confidence >= 0.60 due to AutoBalance oversampling, got %.2f", trace.Confidence)
	}
}

// TestImprovement4_PreloadedBaseVocab_ZeroUNK verifies that pre-loaded base vocabulary
// prevents [UNK] tokens on arbitrary character noise and prevents subword fragmentation.
func TestImprovement4_PreloadedBaseVocab_ZeroUNK(t *testing.T) {
	// 1. Verify DefaultBaseVocab and DefaultBaseCorpus resources exist and are non-empty
	baseVocab := DefaultBaseVocab()
	if len(baseVocab) < 20 {
		t.Fatalf("expected DefaultBaseVocab to contain at least 20 words, got %d", len(baseVocab))
	}
	baseCorpus := DefaultBaseCorpus()
	if len(baseCorpus) < 5 {
		t.Fatalf("expected DefaultBaseCorpus to contain at least 5 sentences, got %d", len(baseCorpus))
	}

	// 2. Train BPE with Base Vocabulary support
	smallCorpus := []string{
		"request return parcel",
		"cancel my order shipment",
	}

	tokenizer, err := TrainBPEWithBaseVocab(smallCorpus, 80, baseVocab)
	if err != nil {
		t.Fatalf("TrainBPEWithBaseVocab failed: %v", err)
	}

	// 3. Verify zero [UNK] on arbitrary English noise
	noiseQuery := "asdfghjklqwertyuiopzxcvbnm"
	noiseTokens := tokenizer.Encode(noiseQuery)
	singleRatio, unkRatio := tokenizer.AnalyzeUnlearnedRatio(noiseTokens)
	t.Logf("Noise Query: %q -> Tokens: %v, UnkRatio: %.2f, SingleRatio: %.2f",
		noiseQuery, noiseTokens, unkRatio, singleRatio)

	if unkRatio > 0.0 {
		t.Errorf("Expected zero UNK tokens for lowercase alphabet noise with pre-loaded base vocab, got unkRatio: %.2f", unkRatio)
	}

	// 4. Verify unfragmented encoding of common words
	commonWordTokens := tokenizer.Encode("return")
	if len(commonWordTokens) == 0 {
		t.Fatalf("expected valid tokens for 'return'")
	}
	// Decode back
	decoded := tokenizer.Decode(commonWordTokens)
	if decoded != "return" {
		t.Errorf("expected decoded text 'return', got %q", decoded)
	}
}

// TestFalseNegativeRecallAndOODIsolation verifies that typo/slang variations of valid domain queries
// are correctly routed (high recall) while general trivia and OOD queries remain strictly blocked (high precision).
func TestFalseNegativeRecallAndOODIsolation(t *testing.T) {
	dataPath := "c:/Users/sezzi/programming/neurobranch-demo/data/train.csv"
	samples, err := LoadCSVDataset(dataPath)
	if err != nil {
		t.Skipf("cannot load demo dataset: %v", err)
	}

	cfg := DefaultTrainConfig()
	cfg.Seed = 42
	cfg.Epochs = 100

	ai, err := TrainInMemory(samples, cfg)
	if err != nil {
		t.Fatalf("TrainInMemory failed: %v", err)
	}

	// 1. In-domain typo/slang queries must route to their proper domain branch
	domainQueries := []struct {
		Query    string
		Expected string
	}{
		{"courier left note but no box at door", "Delivery"},
		{"2fa auth code not sending to phone", "Account"},
		{"send recipt to my email plz", "Billing"},
		{"money back request for damaged shipment", "Refund"},
		{"traxking parcel not moving", "Delivery"},
		{"reset two factor authentication credentials", "Account"},
	}

	for _, tc := range domainQueries {
		branch := ai.Select(tc.Query)
		if branch != tc.Expected {
			trace := ai.Inspect(tc.Query)
			t.Errorf("Query %q: expected branch %q, got %q (trace: conf=%.2f, energy=%.2f, fallback=%t, reason=%s)",
				tc.Query, tc.Expected, branch, trace.Confidence, trace.Energy, trace.IsFallback, trace.FallbackReason)
		}
	}

	// Comma-ok Match validation
	if intent, confident := ai.Match("reset two factor authentication credentials"); intent != "Account" || !confident {
		t.Errorf("Expected Match to return ('Account', true), got (%q, %t)", intent, confident)
	}

	// 2. Out-of-Domain queries must fall back to empty string
	oodQueries := []string{
		"tell me the capital city of France",
		"the weather is nice today",
		"quantum physics entangled photon spin",
		"how to bake sourdough bread with yeast",
	}

	for _, q := range oodQueries {
		trace := ai.Inspect(q)
		branch := ai.Select(q)
		t.Logf("OOD Query %q: branch=%q, conf=%.4f, energy=%.4f, singleRatio=%.2f, subwords=%v, fallback=%t (%s)",
			q, branch, trace.Confidence, trace.Energy, trace.SingleCharRatio, trace.Subwords, trace.IsFallback, trace.FallbackReason)
		if branch != "" {
			t.Errorf("OOD Query %q: expected empty branch, got %q (trace: conf=%.2f, energy=%.2f)",
				q, branch, trace.Confidence, trace.Energy)
		}
	}

	// 3. DSL Confirm dispatch for boundary queries
	ctx := context.Background()
	confirmTriggered := false
	defaultTriggered := false

	err = ai.Switch("money back request for damaged shipment").
		Case("Refund").
		Confirm("Confirm refund authorization of this order?", func(ctx context.Context, prompt string) error {
			confirmTriggered = true
			return nil
		}).
		Default(func(ctx context.Context) error {
			defaultTriggered = true
			return nil
		}).
		Evaluate(ctx)

	if err != nil {
		t.Fatalf("Switch.Evaluate failed: %v", err)
	}
	if !confirmTriggered {
		t.Errorf("Expected DSL Confirm callback to trigger for boundary query 'money back request for damaged shipment'")
	}
	if defaultTriggered {
		t.Errorf("DSL Default callback should not trigger when Confirm is handled")
	}
}
