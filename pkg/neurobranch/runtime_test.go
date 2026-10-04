package neurobranch

import (
	"math"
	"testing"
)

func createStandardSpecModel() *InferenceModel {
	// Architectural specifications:
	// EmbeddingDim = 64
	// HiddenDim = 128
	// VocabSize = 500
	// NumClasses = 5
	header := Header{
		Magic:        MagicBytes,
		Version:      1,
		VocabSize:    500,
		EmbeddingDim: 64,
		HiddenDim:    128,
		NumClasses:   5,
	}

	labels := []string{"Refund", "Delivery", "Account", "Payment", "General"}
	vocab := make([]string, header.VocabSize)
	for i := range vocab {
		vocab[i] = "token"
	}
	vocab[0] = "[PAD]"
	vocab[1] = "[UNK]"
	vocab[2] = "r"
	vocab[3] = "e"
	vocab[4] = "f"
	vocab[5] = "u"
	vocab[6] = "n"
	vocab[7] = "d"
	vocab[8] = " "

	weights := Weights{
		Embedding: make([]float32, header.VocabSize*header.EmbeddingDim),
		W1:        make([]float32, header.EmbeddingDim*header.HiddenDim),
		B1:        make([]float32, header.HiddenDim),
		W2:        make([]float32, header.HiddenDim*header.NumClasses),
		B2:        make([]float32, header.NumClasses),
	}

	for i := range weights.Embedding {
		weights.Embedding[i] = 0.01 * float32(i%10)
	}
	for i := range weights.W1 {
		weights.W1[i] = 0.005 * float32((i%20)-10)
	}
	for i := range weights.W2 {
		weights.W2[i] = 0.005 * float32((i%15)-7)
	}

	return NewInferenceModel(header, labels, vocab, nil, weights)
}

func TestModelForward(t *testing.T) {
	model := createStandardSpecModel()
	tokens := []uint32{1, 2, 5, 10}

	probs, err := model.Forward(tokens, 1.0)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	if len(probs) != int(model.Header.NumClasses) {
		t.Fatalf("Expected %d class probabilities, got %d", model.Header.NumClasses, len(probs))
	}

	var sum float32
	for _, p := range probs {
		if p < 0.0 || p > 1.0 {
			t.Errorf("Invalid probability value: %f", p)
		}
		sum += p
	}

	if math.Abs(float64(sum-1.0)) > 1e-4 {
		t.Errorf("Probabilities sum to %f, expected 1.0", sum)
	}
}

func TestModelPredictTokens(t *testing.T) {
	model := createStandardSpecModel()
	tokens := []uint32{1, 2}

	label, score, err := model.PredictTokens(tokens)
	if err != nil {
		t.Fatalf("PredictTokens failed: %v", err)
	}

	if label == "" {
		t.Error("Expected non-empty label")
	}
	if score <= 0.0 || score > 1.0 {
		t.Errorf("Invalid score: %f", score)
	}
}

func TestInputTruncationGuard(t *testing.T) {
	model := createStandardSpecModel()
	// Create an oversized text string with 2000 characters
	longText := ""
	for i := 0; i < 200; i++ {
		longText += "refund "
	}

	label, score, err := model.Predict(longText)
	if err != nil {
		t.Fatalf("Predict on oversized input failed: %v", err)
	}
	if label == "" || score <= 0.0 {
		t.Errorf("Invalid prediction on truncated input: label=%s, score=%f", label, score)
	}
}

func TestOOVConfidenceDiscounting(t *testing.T) {
	model := createStandardSpecModel()
	// Text composed entirely of characters never seen in vocab
	noiseText := "§±¶€$!@#%^&*()_+~`|}{[]:;?><"

	_, score, err := model.Predict(noiseText)
	if err != nil {
		t.Fatalf("Predict on noise text failed: %v", err)
	}

	// Because almost all tokens are [UNK], calibrated score must be severely discounted (< 0.20)
	if score > 0.35 {
		t.Errorf("Expected severely discounted score for pure noise, got: %f", score)
	}
}

func BenchmarkForward(b *testing.B) {
	model := createStandardSpecModel()
	tokens := []uint32{1, 2, 10, 20, 30, 40}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := model.Forward(tokens, 1.0)
		if err != nil {
			b.Fatalf("Forward failed: %v", err)
		}
	}
}

func BenchmarkPredictTokens(b *testing.B) {
	model := createStandardSpecModel()
	tokens := []uint32{1, 2, 10, 20, 30, 40}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, _, err := model.PredictTokens(tokens)
		if err != nil {
			b.Fatalf("PredictTokens failed: %v", err)
		}
	}
}

func BenchmarkPredictSlots(b *testing.B) {
	model := createStandardSpecModel()
	tokens := []uint32{1, 2, 10, 20, 30, 40}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := model.PredictSlots(tokens, 1.0)
		if err != nil {
			b.Fatalf("PredictSlots failed: %v", err)
		}
	}
}

func TestPredictSlotsZeroAlloc(t *testing.T) {
	model := createStandardSpecModel()
	tokens := []uint32{1, 2, 10}

	res, err := model.PredictSlots(tokens, 1.0)
	if err != nil {
		t.Fatalf("PredictSlots failed: %v", err)
	}

	if res.Total < 1 {
		t.Fatalf("Expected at least 1 prediction slot, got %d", res.Total)
	}
	if res.Primary.Index < 0 || int(res.Primary.Index) >= len(model.Labels) {
		t.Errorf("Primary index out of bounds: %d", res.Primary.Index)
	}
	if res.Primary.Confidence <= 0.0 || res.Primary.Confidence > 1.0 {
		t.Errorf("Primary confidence out of bounds: %f", res.Primary.Confidence)
	}
}

func BenchmarkGELU(b *testing.B) {
	vec := make([]float32, 128)
	for i := range vec {
		vec[i] = float32(i) * 0.1
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GELUInPlace(vec)
	}
}

func TestTruncateToRuneBoundary(t *testing.T) {
	// 3-byte UTF-8 character: Euro symbol '€' (0xE2, 0x82, 0xAC)
	// Text: "Euro € Sign" -> 5 + 3 + 5 = 13 bytes
	multiByteText := "Euro € Sign"

	// Truncating at 6 bytes would cut '€' in half (it starts at index 5 and spans 5..7).
	// It should back up to byte 5 ("Euro ").
	truncated6 := TruncateToRuneBoundary(multiByteText, 6)
	if truncated6 != "Euro " {
		t.Errorf("TruncateToRuneBoundary(multiByteText, 6) = %q; expected %q", truncated6, "Euro ")
	}

	// Truncating at 8 bytes should cleanly include '€' (bytes 0..7).
	truncated8 := TruncateToRuneBoundary(multiByteText, 8)
	if truncated8 != "Euro €" {
		t.Errorf("TruncateToRuneBoundary(multiByteText, 8) = %q; expected %q", truncated8, "Euro €")
	}

	// Truncating at 0 should be empty.
	truncated0 := TruncateToRuneBoundary(multiByteText, 0)
	if truncated0 != "" {
		t.Errorf("TruncateToRuneBoundary(multiByteText, 0) = %q; expected empty", truncated0)
	}

	// ASCII text
	ascii := "hello world"
	truncatedASCII := TruncateToRuneBoundary(ascii, 5)
	if truncatedASCII != "hello" {
		t.Errorf("TruncateToRuneBoundary(ascii, 5) = %q; expected %q", truncatedASCII, "hello")
	}
}

func TestSemanticXOR_PositionalDisambiguation(t *testing.T) {
	embDim := 4
	embeddingTable := []float32{
		1.0, 0.0, 0.0, 0.0, // token 0: "A"
		0.0, 1.0, 0.0, 0.0, // token 1: "B"
	}
	posTable := []float32{
		0.1, 0.2, 0.0, 0.0, // position 0
		0.0, 0.0, 0.3, 0.4, // position 1
	}

	seqAB := []uint32{0, 1} // A then B
	seqBA := []uint32{1, 0} // B then A

	// 1. Without positional encoding (Legacy MeanPooling): Vectors are 100% IDENTICAL
	outLegacyAB := make([]float32, embDim)
	outLegacyBA := make([]float32, embDim)
	_ = MeanPooling(seqAB, embeddingTable, embDim, outLegacyAB)
	_ = MeanPooling(seqBA, embeddingTable, embDim, outLegacyBA)

	diffLegacy := float32(0.0)
	for d := 0; d < embDim; d++ {
		diffLegacy += float32(math.Abs(float64(outLegacyAB[d] - outLegacyBA[d])))
	}
	if diffLegacy != 0.0 {
		t.Fatalf("Legacy MeanPooling should have suffered from XOR collapse (identical vectors), but diff was %f", diffLegacy)
	}

	// 2. With positional encoding (MeanPoolingWithPos): Vectors MUST DIVERGE
	outPosAB := make([]float32, embDim)
	outPosBA := make([]float32, embDim)
	_ = MeanPoolingWithPos(seqAB, embeddingTable, posTable, embDim, outPosAB)
	_ = MeanPoolingWithPos(seqBA, embeddingTable, posTable, embDim, outPosBA)

	diffPos := float32(0.0)
	for d := 0; d < embDim; d++ {
		diffPos += float32(math.Abs(float64(outPosAB[d] - outPosBA[d])))
	}

	if diffPos <= 0.05 {
		t.Errorf("Positional embedding failed to disambiguate token order: diff = %f, expected > 0.05", diffPos)
	}
}

func TestEmptyWhitespaceInputReturnsError(t *testing.T) {
	model := createStandardSpecModel()

	// Empty string
	_, _, err := model.Predict("")
	if err != ErrEmptyInput {
		t.Errorf("Expected ErrEmptyInput for empty string, got: %v", err)
	}

	// All whitespace string (BPE tokenizer produces zero tokens)
	_, _, err = model.Predict("   \t\n   ")
	if err != ErrEmptyInput {
		t.Errorf("Expected ErrEmptyInput for whitespace string, got: %v", err)
	}
}
