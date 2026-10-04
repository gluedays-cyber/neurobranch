package neurobranch

import (
	"math"
	"testing"
)

func TestGELU(t *testing.T) {
	// Reference values for GELU tanh approximation:
	// GELU(0.0) == 0.0
	// GELU(1.0) == 0.841192
	// GELU(-1.0) == -0.158808
	testCases := []struct {
		input    float32
		expected float32
		epsilon  float32
	}{
		{0.0, 0.0, 1e-5},
		{1.0, 0.841192, 1e-4},
		{-1.0, -0.158808, 1e-4},
		{2.0, 1.9545, 1e-3},
		{-2.0, -0.0454, 1e-3},
	}

	for _, tc := range testCases {
		res := GELU(tc.input)
		diff := float32(math.Abs(float64(res - tc.expected)))
		if diff > tc.epsilon {
			t.Errorf("GELU(%f) = %f; expected %f (diff: %f)", tc.input, res, tc.expected, diff)
		}
	}
}

func TestGELUInPlace(t *testing.T) {
	inputs := []float32{0.0, 1.0, -1.0}
	expected := []float32{0.0, 0.841192, -0.158808}

	GELUInPlace(inputs)

	for i := range inputs {
		diff := float32(math.Abs(float64(inputs[i] - expected[i])))
		if diff > 1e-4 {
			t.Errorf("Index %d: got %f, expected %f", i, inputs[i], expected[i])
		}
	}
}

func TestSoftmax(t *testing.T) {
	logits := []float32{2.0, 1.0, 0.1}
	probs := make([]float32, len(logits))

	if err := Softmax(logits, 1.0, probs); err != nil {
		t.Fatalf("Softmax failed: %v", err)
	}

	var sum float32
	for _, p := range probs {
		if p < 0.0 || p > 1.0 {
			t.Errorf("Probability out of bounds: %f", p)
		}
		sum += p
	}

	if math.Abs(float64(sum-1.0)) > 1e-5 {
		t.Errorf("Softmax probabilities sum to %f; expected 1.0", sum)
	}

	if probs[0] <= probs[1] || probs[1] <= probs[2] {
		t.Errorf("Softmax monotonicity broken: %v", probs)
	}
}

func TestMeanPooling(t *testing.T) {
	embDim := 4
	embeddingTable := []float32{
		1.0, 2.0, 3.0, 4.0, // token 0
		5.0, 6.0, 7.0, 8.0, // token 1
	}

	tokenIDs := []uint32{0, 1}
	out := make([]float32, embDim)

	if err := MeanPooling(tokenIDs, embeddingTable, embDim, out); err != nil {
		t.Fatalf("MeanPooling failed: %v", err)
	}

	expected := []float32{3.0, 4.0, 5.0, 6.0}
	for i := 0; i < embDim; i++ {
		if out[i] != expected[i] {
			t.Errorf("MeanPooling dim %d: got %f, expected %f", i, out[i], expected[i])
		}
	}
}

func TestMatMulVecAdd(t *testing.T) {
	inDim := 2
	outDim := 3

	vec := []float32{1.0, 2.0}
	weights := []float32{
		1.0, 0.5, 0.0, // row 0
		0.0, 1.0, 2.0, // row 1
	}
	bias := []float32{0.1, 0.2, 0.3}
	out := make([]float32, outDim)

	if err := MatMulVecAdd(vec, weights, bias, inDim, outDim, out); err != nil {
		t.Fatalf("MatMulVecAdd failed: %v", err)
	}

	// Calculation:
	// out[0] = 1*1.0 + 2*0.0 + 0.1 = 1.1
	// out[1] = 1*0.5 + 2*1.0 + 0.2 = 2.7
	// out[2] = 1*0.0 + 2*2.0 + 0.3 = 4.3
	expected := []float32{1.1, 2.7, 4.3}
	for i := 0; i < outDim; i++ {
		diff := float32(math.Abs(float64(out[i] - expected[i])))
		if diff > 1e-5 {
			t.Errorf("MatMulVecAdd out[%d]: got %f, expected %f", i, out[i], expected[i])
		}
	}
}

func TestOps_NumericalHardening_NaN_Inf_Clamping(t *testing.T) {
	// 1. SafeClamp validation
	nanVal := float32(math.NaN())
	infVal := float32(math.Inf(1))
	negInfVal := float32(math.Inf(-1))

	if SafeClamp(nanVal, 100.0) != 0.0 {
		t.Errorf("SafeClamp(NaN) expected 0.0, got %f", SafeClamp(nanVal, 100.0))
	}
	if SafeClamp(infVal, 100.0) != 0.0 {
		t.Errorf("SafeClamp(+Inf) expected 0.0, got %f", SafeClamp(infVal, 100.0))
	}
	if SafeClamp(negInfVal, 100.0) != 0.0 {
		t.Errorf("SafeClamp(-Inf) expected 0.0, got %f", SafeClamp(negInfVal, 100.0))
	}
	if SafeClamp(250.0, 100.0) != 100.0 {
		t.Errorf("SafeClamp(250.0) expected 100.0, got %f", SafeClamp(250.0, 100.0))
	}
	if SafeClamp(-250.0, 100.0) != -100.0 {
		t.Errorf("SafeClamp(-250.0) expected -100.0, got %f", SafeClamp(-250.0, 100.0))
	}

	// 2. GELU with extreme inputs
	extremeGELU := GELU(1e20)
	if math.IsNaN(float64(extremeGELU)) || math.IsInf(float64(extremeGELU), 0) || extremeGELU > 100.0 {
		t.Errorf("GELU with extreme input breached clamping: %f", extremeGELU)
	}

	nanGELU := GELU(nanVal)
	if nanGELU != 0.0 {
		t.Errorf("GELU(NaN) expected 0.0, got %f", nanGELU)
	}

	// 3. MatMulVecAdd with NaN and Inf weights
	corruptedVec := []float32{nanVal, infVal}
	corruptedWeights := []float32{1.0, 2.0, 3.0, 4.0}
	corruptedBias := []float32{nanVal, 10.0}
	corruptedOut := make([]float32, 2)

	if err := MatMulVecAdd(corruptedVec, corruptedWeights, corruptedBias, 2, 2, corruptedOut); err != nil {
		t.Fatalf("MatMulVecAdd failed on corrupted input: %v", err)
	}
	for i, v := range corruptedOut {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("MatMulVecAdd leaked NaN/Inf to out[%d]: %f", i, v)
		}
	}

	// 4. Softmax with degenerated logits
	degeneratedLogits := []float32{nanVal, infVal, -infVal}
	degeneratedProbs := make([]float32, 3)
	if err := Softmax(degeneratedLogits, 1.0, degeneratedProbs); err != nil {
		t.Fatalf("Softmax failed on degenerated logits: %v", err)
	}
	for i, p := range degeneratedProbs {
		if math.IsNaN(float64(p)) || math.IsInf(float64(p), 0) {
			t.Errorf("Softmax leaked NaN/Inf to prob[%d]: %f", i, p)
		}
	}
}
