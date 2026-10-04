package neurobranch

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func setupTestRouter(t *testing.T) *Router {
	t.Helper()
	tempDir := t.TempDir()
	csvPath := filepath.Join(tempDir, "dataset.csv")
	modelPath := filepath.Join(tempDir, "model.bin")

	csvContent := `text,label
refund my money please,Refund
cancel payment and refund,Refund
where is my package shipment,Delivery
track order delivery status,Delivery
reset my account password,Account
cannot login to account,Account
`
	if err := os.WriteFile(csvPath, []byte(csvContent), 0644); err != nil {
		t.Fatalf("failed to write test csv: %v", err)
	}

	cfg := DefaultTrainConfig()
	cfg.Epochs = 50
	cfg.TargetVocabSize = 128
	if err := Train(csvPath, modelPath, cfg); err != nil {
		t.Fatalf("Train failed: %v", err)
	}

	router, err := Open(modelPath, 0.40)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	return router
}

func TestNativeBranching_Select_Is_Match_Assert(t *testing.T) {
	router := setupTestRouter(t)

	// 1. Select: Native switch compatibility
	query := "cancel payment and refund please"
	selected := router.Select(query)
	if selected != "Refund" {
		t.Errorf("expected 'Refund', got '%s'", selected)
	}

	// 2. Select with OOD (should fall through to default with empty string)
	oodQuery := "quantum thermodynamics entropy equation"
	oodSelected := router.Select(oodQuery)
	if oodSelected != "" {
		t.Errorf("expected empty string for OOD query, got '%s'", oodSelected)
	}

	// 3. Is: Native if guard clause
	if !router.Is(query, "Refund") {
		t.Errorf("expected router.Is(query, 'Refund') to be true")
	}
	if router.Is(query, "Delivery") {
		t.Errorf("expected router.Is(query, 'Delivery') to be false")
	}
	if router.Is(oodQuery, "Refund") {
		t.Errorf("expected router.Is(oodQuery, 'Refund') to be false")
	}

	// 4. Match: Native comma-ok idiom
	intent, ok := router.Match(query)
	if !ok || intent != "Refund" {
		t.Errorf("expected ('Refund', true), got ('%s', %v)", intent, ok)
	}

	_, oodOk := router.Match(oodQuery)
	if oodOk {
		t.Errorf("expected Match ok to be false for OOD query")
	}

	// 5. Assert: Error-based verification
	if err := router.Assert(query, "Refund"); err != nil {
		t.Errorf("expected nil error for valid match, got %v", err)
	}
	if err := router.Assert(query, "Delivery"); !errors.Is(err, ErrOutOfDomain) && !errors.Is(err, ErrAmbiguousIntent) {
		t.Errorf("expected ErrOutOfDomain or mismatch error, got %v", err)
	}
}

func TestDeclarativeSwitchBuilder(t *testing.T) {
	router := setupTestRouter(t)
	ctx := context.Background()

	// 1. Confident Auto Branch
	trace := router.Inspect("where is my package shipment")
	t.Logf("Inspect Trace: %+v", trace)

	var executedBranch string
	err := router.Switch("where is my package shipment").
		Case("Refund").
			Auto(func(ctx context.Context) error {
				executedBranch = "Refund"
				return nil
			}).
		Case("Delivery").
			Auto(func(ctx context.Context) error {
				executedBranch = "Delivery"
				return nil
			}).
		Default(func(ctx context.Context) error {
			executedBranch = "Default"
			return nil
		}).
		Evaluate(ctx)

	if err != nil {
		t.Fatalf("unexpected evaluate error: %v", err)
	}
	if executedBranch != "Delivery" {
		t.Errorf("expected 'Delivery', got '%s'", executedBranch)
	}

	// 2. Default Fallback on OOD
	executedBranch = ""
	err = router.Switch("alien spacecraft propulsion mechanics").
		Case("Refund").
			Auto(func(ctx context.Context) error {
				executedBranch = "Refund"
				return nil
			}).
		Default(func(ctx context.Context) error {
			executedBranch = "Default"
			return nil
		}).
		Evaluate(ctx)

	if err != nil {
		t.Fatalf("unexpected evaluate error: %v", err)
	}
	if executedBranch != "Default" {
		t.Errorf("expected 'Default', got '%s'", executedBranch)
	}

	// 3. Confirm Branch when threshold is not met
	var confirmTriggered bool
	err = router.Switch("refund my money please").
		Case("Refund").
			AtLeast(0.9999). // Intentionally unattainable threshold to force confirmation
			Confirm("Would you like to refund?", func(ctx context.Context, prompt string) error {
				confirmTriggered = true
				if prompt != "Would you like to refund?" {
					t.Errorf("unexpected prompt: %s", prompt)
				}
				return nil
			}).
		Evaluate(ctx)

	if err != nil {
		t.Fatalf("unexpected evaluate error: %v", err)
	}
	if !confirmTriggered {
		t.Errorf("expected confirm handler to be executed")
	}
}

func BenchmarkNativeSelectAllocations(b *testing.B) {
	tempDir := b.TempDir()
	csvPath := filepath.Join(tempDir, "dataset.csv")
	modelPath := filepath.Join(tempDir, "model.bin")

	csvContent := `text,label
refund my money please,Refund
cancel payment and refund,Refund
where is my package shipment,Delivery
track order delivery status,Delivery
reset my account password,Account
cannot login to account,Account
`
	if err := os.WriteFile(csvPath, []byte(csvContent), 0644); err != nil {
		b.Fatalf("failed to write test csv: %v", err)
	}

	cfg := DefaultTrainConfig()
	cfg.Epochs = 20
	cfg.TargetVocabSize = 50
	if err := Train(csvPath, modelPath, cfg); err != nil {
		b.Fatalf("Train failed: %v", err)
	}

	router, err := Open(modelPath, 0.40)
	if err != nil {
		b.Fatalf("Open failed: %v", err)
	}

	query := "cancel payment and refund please"

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = router.Select(query)
	}
}

func BenchmarkNativeIsAllocations(b *testing.B) {
	tempDir := b.TempDir()
	csvPath := filepath.Join(tempDir, "dataset.csv")
	modelPath := filepath.Join(tempDir, "model.bin")

	csvContent := `text,label
refund my money please,Refund
cancel payment and refund,Refund
where is my package shipment,Delivery
track order delivery status,Delivery
reset my account password,Account
cannot login to account,Account
`
	if err := os.WriteFile(csvPath, []byte(csvContent), 0644); err != nil {
		b.Fatalf("failed to write test csv: %v", err)
	}

	cfg := DefaultTrainConfig()
	cfg.Epochs = 20
	cfg.TargetVocabSize = 50
	if err := Train(csvPath, modelPath, cfg); err != nil {
		b.Fatalf("Train failed: %v", err)
	}

	router, err := Open(modelPath, 0.40)
	if err != nil {
		b.Fatalf("Open failed: %v", err)
	}

	query := "cancel payment and refund please"

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = router.Is(query, "Refund")
	}
}
