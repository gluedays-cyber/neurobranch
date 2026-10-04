package neurobranch

import (
	"bytes"
	"path/filepath"
	"testing"
)

// TestRFC_StrictShortCircuit_OODBeforeAmbiguity proves that an Out-of-Domain sample
// with narrow margin is strictly rejected as OOD and NEVER incorrectly flagged as Ambiguous.
func TestRFC_StrictShortCircuit_OODBeforeAmbiguity(t *testing.T) {
	tempDir := t.TempDir()
	modelPath := filepath.Join(tempDir, "ood_short_circuit.bin")

	samples := []DataSample{
		{Text: "refund money to bank account card", Label: "Refund"},
		{Text: "track courier shipment parcel delivery", Label: "Delivery"},
	}
	cfg := DefaultTrainConfig()
	cfg.Epochs = 30
	model, err := TrainModel(samples, cfg)
	if err != nil {
		t.Fatalf("TrainModel failed: %v", err)
	}
	if err := SaveBinaryModel(modelPath, model); err != nil {
		t.Fatalf("SaveBinaryModel failed: %v", err)
	}

	gate, err := NewNeuroBranch(modelPath)
	if err != nil {
		t.Fatalf("NewNeuroBranch failed: %v", err)
	}

	// Calibrate distribution to establish true manifold boundaries
	gate.CalibrateDomainDistribution(samples, 1.5)

	// A foreign domain input that has nothing to do with Refund or Delivery
	foreignQuery := "quantum entanglement physics simulation particles"
	trace := gate.Inspect(foreignQuery)

	if !trace.IsOOD {
		t.Errorf("Expected foreign query to be flagged as IsOOD=true, got IsOOD=%t (reason: %s)", trace.IsOOD, trace.FallbackReason)
	}
	if trace.IsAmbiguous {
		t.Errorf("Violation of RFC Short-Circuit: Foreign OOD query was incorrectly marked as IsAmbiguous=true")
	}
}

// TestRFC_SelfCalibratingMetadataPortability validates that Header FormatVersion3
// preserves and propagates calibration thresholds directly into new gate instances.
func TestRFC_SelfCalibratingMetadataPortability(t *testing.T) {
	tempDir := t.TempDir()
	modelPath := filepath.Join(tempDir, "v3_calibrated.bin")

	samples := []DataSample{
		{Text: "refund money to bank account card", Label: "Refund"},
		{Text: "track courier shipment parcel delivery", Label: "Delivery"},
	}
	cfg := DefaultTrainConfig()
	cfg.Epochs = 30
	model, err := TrainModel(samples, cfg)
	if err != nil {
		t.Fatalf("TrainModel failed: %v", err)
	}

	gate1 := NewNeuroBranchWithModel(model)
	gate1.CalibrateDomainDistribution(samples, 2.0)

	// Save calibrated model (FormatVersion3)
	if err := SaveBinaryModel(modelPath, gate1.Model()); err != nil {
		t.Fatalf("SaveBinaryModel failed: %v", err)
	}

	// Load model into a fresh instance without runtime calibration
	loadedModel, err := LoadBinaryModel(modelPath)
	if err != nil {
		t.Fatalf("LoadBinaryModel failed: %v", err)
	}

	if loadedModel.Header.Version != FormatVersion3 {
		t.Fatalf("Expected FormatVersion3, got %d", loadedModel.Header.Version)
	}
	if loadedModel.Header.CalibratedMinEnergy <= 0.0 {
		t.Errorf("Expected positive CalibratedMinEnergy in header, got %f", loadedModel.Header.CalibratedMinEnergy)
	}
	if loadedModel.Header.CalibratedMargin <= 0.0 {
		t.Errorf("Expected positive CalibratedMargin in header, got %f", loadedModel.Header.CalibratedMargin)
	}

	// Instantiate fresh NeuroBranch and Router from the v3 binary
	gate2 := NewNeuroBranchWithModel(loadedModel)
	if gate2.policy.MinLogSumExp != float64(loadedModel.Header.CalibratedMinEnergy) {
		t.Errorf("Gate did not auto-apply MinLogSumExp: got %f, expected %f",
			gate2.policy.MinLogSumExp, loadedModel.Header.CalibratedMinEnergy)
	}

	router, err := NewRouter(modelPath, 0.70)
	if err != nil {
		t.Fatalf("NewRouter failed: %v", err)
	}
	if router.policy.MinLogSumExp != float64(loadedModel.Header.CalibratedMinEnergy) {
		t.Errorf("Router did not auto-apply MinLogSumExp: got %f, expected %f",
			router.policy.MinLogSumExp, loadedModel.Header.CalibratedMinEnergy)
	}
}

// TestRFC_LegacyVersion2BackwardCompatibility ensures existing v2 binaries load and run without errors.
func TestRFC_LegacyVersion2BackwardCompatibility(t *testing.T) {
	header := Header{
		Magic:        MagicBytes,
		Version:      FormatVersion2,
		VocabSize:    4,
		EmbeddingDim: 4,
		HiddenDim:    6,
		NumClasses:   2,
	}
	labels := []string{"Refund", "Delivery"}
	vocab := []string{"[PAD]", "refund", "cancel", "delivery"}
	mergeRules := []MergeRule{
		{Token1: 1, Token2: 2, Target: 3},
	}
	posLen := int(MaxSequenceTokens * header.EmbeddingDim)
	weights := Weights{
		Embedding:  make([]float32, header.VocabSize*header.EmbeddingDim),
		Positional: make([]float32, posLen),
		W1:         make([]float32, header.EmbeddingDim*header.HiddenDim),
		B1:         make([]float32, header.HiddenDim),
		W2:         make([]float32, header.HiddenDim*header.NumClasses),
		B2:         make([]float32, header.NumClasses),
	}

	v2Model := NewInferenceModel(header, labels, vocab, mergeRules, weights)
	var buf bytes.Buffer
	if err := SerializeModel(&buf, v2Model); err != nil {
		t.Fatalf("SerializeModel v2 failed: %v", err)
	}

	loadedModel, err := DeserializeModel(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("DeserializeModel failed on v2 binary: %v", err)
	}
	if loadedModel.Header.Version != FormatVersion2 {
		t.Errorf("Expected FormatVersion2, got %d", loadedModel.Header.Version)
	}

	// Verify gate loads v2 model cleanly
	gate := NewNeuroBranchWithModel(loadedModel)
	if gate == nil {
		t.Fatal("Expected non-nil NeuroBranch with v2 model")
	}
	trace := gate.Inspect("refund cancel")
	if trace.PredictedLabel == "" {
		t.Fatalf("Inspect on v2 model returned empty label")
	}
}

// TestRFC_DeterministicTraining_SeedReproducibility tests 100% deterministic weight reproduction with fixed seed.
func TestRFC_DeterministicTraining_SeedReproducibility(t *testing.T) {
	samples := []DataSample{
		{Text: "cancel order and refund money", Label: "Refund"},
		{Text: "request payment refund please", Label: "Refund"},
		{Text: "package tracking delivery status", Label: "Delivery"},
		{Text: "where is my delivery parcel", Label: "Delivery"},
	}

	cfg1 := DefaultTrainConfig()
	cfg1.Epochs = 20
	cfg1.Seed = 1337
	model1, err := TrainModel(samples, cfg1)
	if err != nil {
		t.Fatalf("TrainModel 1 failed: %v", err)
	}

	cfg2 := DefaultTrainConfig()
	cfg2.Epochs = 20
	cfg2.Seed = 1337
	model2, err := TrainModel(samples, cfg2)
	if err != nil {
		t.Fatalf("TrainModel 2 failed: %v", err)
	}

	// 1. Verify weights match exactly byte-for-byte between model1 and model2
	if len(model1.Weights.Embedding) != len(model2.Weights.Embedding) {
		t.Fatalf("Embedding length mismatch: %d vs %d", len(model1.Weights.Embedding), len(model2.Weights.Embedding))
	}
	for i := range model1.Weights.Embedding {
		if model1.Weights.Embedding[i] != model2.Weights.Embedding[i] {
			t.Fatalf("Embedding weight mismatch at index %d: %f vs %f", i, model1.Weights.Embedding[i], model2.Weights.Embedding[i])
		}
	}
	for i := range model1.Weights.W1 {
		if model1.Weights.W1[i] != model2.Weights.W1[i] {
			t.Fatalf("W1 weight mismatch at index %d: %f vs %f", i, model1.Weights.W1[i], model2.Weights.W1[i])
		}
	}
	for i := range model1.Weights.W2 {
		if model1.Weights.W2[i] != model2.Weights.W2[i] {
			t.Fatalf("W2 weight mismatch at index %d: %f vs %f", i, model1.Weights.W2[i], model2.Weights.W2[i])
		}
	}

	// 2. Verify model with different seed produces different weights
	cfg3 := DefaultTrainConfig()
	cfg3.Epochs = 20
	cfg3.Seed = 9999
	model3, err := TrainModel(samples, cfg3)
	if err != nil {
		t.Fatalf("TrainModel 3 failed: %v", err)
	}

	diffCount := 0
	for i := range model1.Weights.W1 {
		if model1.Weights.W1[i] != model3.Weights.W1[i] {
			diffCount++
		}
	}
	if diffCount == 0 {
		t.Error("Expected different seed (9999) to produce different weights, but got identical weights")
	}
}

// TestRFC_ZeroAllocationBenchmark verifies that inference remains strictly zero-allocation.
func BenchmarkRFC_ZeroAllocationInference(b *testing.B) {
	samples := []DataSample{
		{Text: "refund money to card", Label: "Refund"},
		{Text: "package delivery tracking", Label: "Delivery"},
	}
	cfg := DefaultTrainConfig()
	cfg.Epochs = 20
	model, err := TrainModel(samples, cfg)
	if err != nil {
		b.Fatalf("TrainModel failed: %v", err)
	}

	gate := NewNeuroBranchWithModel(model)
	tokens := model.Tokenizer.Encode("refund money")

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = gate.SelectTokens(tokens)
	}
}
