package neurobranch

import (
	"errors"
	"math"
)

const (
	// Sqrt2OverPi is sqrt(2 / pi) for the GELU tanh approximation.
	Sqrt2OverPi float32 = 0.7978845608

	// GeluCoeff is the polynomial coefficient for GELU tanh approximation.
	GeluCoeff float32 = 0.044715

	// NumericalClampLimit protects float32 values from overflowing into Inf/NaN.
	NumericalClampLimit float32 = 100.0
)

var (
	ErrZeroLengthTokens  = errors.New("cannot pool over zero tokens")
	ErrDimensionMismatch = errors.New("tensor dimension mismatch during linear operation")
)

// SafeClamp restricts a float32 within [-limit, limit] and replaces NaN/Inf with 0.0.
func SafeClamp(val float32, limit float32) float32 {
	if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
		return 0.0
	}
	if val < -limit {
		return -limit
	}
	if val > limit {
		return limit
	}
	return val
}

// GELU calculates the Gaussian Error Linear Unit activation using the standard tanh approximation.
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
func GELU(x float32) float32 {
	x = SafeClamp(x, NumericalClampLimit)
	cube := x * x * x
	inner := Sqrt2OverPi * (x + GeluCoeff*cube)
	inner = SafeClamp(inner, NumericalClampLimit)
	tanhVal := float32(math.Tanh(float64(inner)))
	return SafeClamp(0.5*x*(1.0+tanhVal), NumericalClampLimit)
}

// GELUInPlace applies the GELU non-linear activation function across a float32 slice in-place.
func GELUInPlace(vec []float32) {
	for i := 0; i < len(vec); i++ {
		vec[i] = GELU(vec[i])
	}
}

// MeanPoolingWithPos computes the average embedding vector across the given token IDs with learned positional embeddings.
// out must have a length of at least embDim.
func MeanPoolingWithPos(tokenIDs []uint32, embeddingTable []float32, posTable []float32, embDim int, out []float32) error {
	seqLen := len(tokenIDs)
	if seqLen == 0 {
		return ErrZeroLengthTokens
	}

	// Zero out target buffer
	for i := 0; i < embDim; i++ {
		out[i] = 0.0
	}

	maxSeq := 0
	if embDim > 0 && len(posTable) > 0 {
		maxSeq = len(posTable) / embDim
	}

	// Accumulate embeddings with positional encoding and numerical clamp protection
	for pos, id := range tokenIDs {
		tokOffset := int(id) * embDim
		if tokOffset+embDim > len(embeddingTable) {
			return errors.New("token ID exceeds embedding table bounds")
		}

		var posOffset int
		hasPos := false
		if maxSeq > 0 && pos < maxSeq {
			posOffset = pos * embDim
			hasPos = true
		}

		for d := 0; d < embDim; d++ {
			val := SafeClamp(embeddingTable[tokOffset+d], NumericalClampLimit)
			if hasPos {
				// Non-linear GELU projection breaks the commutative property of summation:
				// GELU(A + P0) + GELU(B + P1) != GELU(B + P0) + GELU(A + P1)
				val = GELU(val + posTable[posOffset+d])
			}
			out[d] = SafeClamp(out[d]+val, NumericalClampLimit)
		}
	}

	// Scale by 1 / L
	invLen := 1.0 / float32(seqLen)
	for d := 0; d < embDim; d++ {
		out[d] = SafeClamp(out[d]*invLen, NumericalClampLimit)
	}

	return nil
}

// MeanPooling computes the average embedding vector across the given token IDs without positional encoding.
func MeanPooling(tokenIDs []uint32, embeddingTable []float32, embDim int, out []float32) error {
	return MeanPoolingWithPos(tokenIDs, embeddingTable, nil, embDim, out)
}

// MatMulVecAdd computes out = vec * weights + bias where vec is [1 x inDim], weights is [inDim x outDim],
// bias is [outDim], and out is [outDim].
func MatMulVecAdd(vec []float32, weights []float32, bias []float32, inDim int, outDim int, out []float32) error {
	if len(vec) < inDim || len(bias) < outDim || len(out) < outDim || len(weights) < inDim*outDim {
		return ErrDimensionMismatch
	}

	// Initialize with clamped bias values
	for j := 0; j < outDim; j++ {
		out[j] = SafeClamp(bias[j], NumericalClampLimit)
	}

	// Perform vector-matrix product with cache-efficient layout: weights is inDim x outDim row-major
	for i := 0; i < inDim; i++ {
		v := SafeClamp(vec[i], NumericalClampLimit)
		if v == 0.0 {
			continue
		}
		rowOffset := i * outDim
		for j := 0; j < outDim; j++ {
			w := SafeClamp(weights[rowOffset+j], NumericalClampLimit)
			out[j] = SafeClamp(out[j]+v*w, NumericalClampLimit)
		}
	}

	return nil
}

// Softmax computes the numerically stable softmax probabilities over logits with temperature scaling.
// out must have at least len(logits).
func Softmax(logits []float32, temperature float32, out []float32) error {
	n := len(logits)
	if n == 0 {
		return errors.New("empty logits")
	}
	if temperature <= 0.0 || math.IsNaN(float64(temperature)) || math.IsInf(float64(temperature), 0) {
		temperature = 1.0
	}

	invTemp := 1.0 / temperature

	// Find max logit for numerical stability
	maxLogit := SafeClamp(logits[0], NumericalClampLimit) * invTemp
	for i := 1; i < n; i++ {
		scaled := SafeClamp(logits[i], NumericalClampLimit) * invTemp
		if scaled > maxLogit {
			maxLogit = scaled
		}
	}

	// Compute exp and sum
	var sumExp float32
	for i := 0; i < n; i++ {
		val := SafeClamp(logits[i], NumericalClampLimit)*invTemp - maxLogit
		e := float32(math.Exp(float64(val)))
		if math.IsNaN(float64(e)) || math.IsInf(float64(e), 0) {
			e = 0.0
		}
		out[i] = e
		sumExp += e
	}

	// Fallback to uniform distribution if sumExp is degenerated
	if sumExp <= 0.0 || math.IsNaN(float64(sumExp)) || math.IsInf(float64(sumExp), 0) {
		uniform := 1.0 / float32(n)
		for i := 0; i < n; i++ {
			out[i] = uniform
		}
		return nil
	}

	// Normalize
	invSum := 1.0 / sumExp
	for i := 0; i < n; i++ {
		out[i] *= invSum
	}

	return nil
}

// L2Normalize computes out = vec / ||vec||2 with numerical safety.
// Returns the original Euclidean norm.
func L2Normalize(vec []float32, out []float32) float32 {
	var sumSq float64
	for _, v := range vec {
		sumSq += float64(v * v)
	}
	norm := float32(math.Sqrt(sumSq))
	if norm < 1e-7 || math.IsNaN(float64(norm)) || math.IsInf(float64(norm), 0) {
		for i := range out {
			out[i] = 0
		}
		return 0
	}
	invNorm := 1.0 / norm
	for i, v := range vec {
		out[i] = v * invNorm
	}
	return norm
}

// DotProduct computes the dot product between two float32 slices without allocations.
func DotProduct(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var sum float32
	for i := 0; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// LogSumExp calculates ln(sum(exp(x_i))) in a numerically stable manner with zero heap allocations.
func LogSumExp(logits []float32) float32 {
	n := len(logits)
	if n == 0 {
		return 0.0
	}

	maxLogit := SafeClamp(logits[0], NumericalClampLimit)
	for i := 1; i < n; i++ {
		val := SafeClamp(logits[i], NumericalClampLimit)
		if val > maxLogit {
			maxLogit = val
		}
	}

	var sumExp float64
	for i := 0; i < n; i++ {
		val := SafeClamp(logits[i], NumericalClampLimit) - maxLogit
		e := math.Exp(float64(val))
		if !math.IsNaN(e) && !math.IsInf(e, 0) {
			sumExp += e
		}
	}

	if sumExp <= 0.0 || math.IsNaN(sumExp) || math.IsInf(sumExp, 0) {
		return maxLogit
	}

	return maxLogit + float32(math.Log(sumExp))
}

// CosineSimilarity calculates the cosine similarity between two float32 vectors.
func CosineSimilarity(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if n == 0 {
		return 0.0
	}

	var dot, normA, normB float64
	for i := 0; i < n; i++ {
		va := float64(a[i])
		vb := float64(b[i])
		dot += va * vb
		normA += va * va
		normB += vb * vb
	}

	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom < 1e-7 || math.IsNaN(denom) || math.IsInf(denom, 0) {
		return 0.0
	}

	sim := float32(dot / denom)
	if sim > 1.0 {
		return 1.0
	}
	if sim < -1.0 {
		return -1.0
	}
	return sim
}

