package neurobranch

import (
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"strings"
)

// TrainConfig defines the hyperparameter specification for offline model training.
type TrainConfig struct {
	EmbeddingDim    int
	HiddenDim       int
	TargetVocabSize int
	LearningRate    float32
	WeightDecay     float32
	Beta1           float32
	Beta2           float32
	Epsilon         float32
	Epochs          int
	BatchSize       int
	Patience        int
	Seed            int64
}

// DefaultTrainConfig provides production-ready default hyperparameters.
func DefaultTrainConfig() TrainConfig {
	return TrainConfig{
		EmbeddingDim:    64,
		HiddenDim:       128,
		TargetVocabSize: 200,
		LearningRate:    0.001,
		WeightDecay:     0.01,
		Beta1:           0.9,
		Beta2:           0.999,
		Epsilon:         1e-8,
		Epochs:          100,
		BatchSize:       32,
		Patience:        5,
		Seed:            42,
	}
}

// DataSample represents a paired text and target label entry.
type DataSample struct {
	Text  string
	Label string
}

// LoadCSVDataset reads training data from a two-column (text,label) CSV file.
func LoadCSVDataset(filePath string) ([]DataSample, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open dataset file: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	// Header row
	header, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("failed to read CSV header: %w", err)
	}
	if len(header) < 2 {
		return nil, fmt.Errorf("CSV must contain at least 2 columns (text, label)")
	}

	var samples []DataSample
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("error reading record: %w", err)
		}
		if len(record) < 2 {
			continue
		}
		text := strings.TrimSpace(record[0])
		label := strings.TrimSpace(record[1])
		if text != "" && label != "" {
			samples = append(samples, DataSample{Text: text, Label: label})
		}
	}

	if len(samples) == 0 {
		return nil, fmt.Errorf("dataset is empty")
	}

	return samples, nil
}

// AdamWState maintains first and second momentum vectors for AdamW parameter updates.
type AdamWState struct {
	m []float32
	v []float32
	t int
}

func newAdamWState(size int) *AdamWState {
	return &AdamWState{
		m: make([]float32, size),
		v: make([]float32, size),
		t: 0,
	}
}

func (opt *AdamWState) Step(param []float32, grad []float32, lr float32, weightDecay float32, beta1 float32, beta2 float32, eps float32) {
	opt.t++
	tFloat := float64(opt.t)
	bc1 := float32(1.0 - math.Pow(float64(beta1), tFloat))
	bc2 := float32(1.0 - math.Pow(float64(beta2), tFloat))

	for i := 0; i < len(param); i++ {
		g := grad[i]

		// Update biased 1st and 2nd momentum
		opt.m[i] = beta1*opt.m[i] + (1.0-beta1)*g
		opt.v[i] = beta2*opt.v[i] + (1.0-beta2)*(g*g)

		// Bias-corrected estimates
		mHat := opt.m[i] / bc1
		vHat := opt.v[i] / bc2

		// Weight decay update & Adam step
		param[i] -= lr * (mHat/(float32(math.Sqrt(float64(vHat)))+eps) + weightDecay*param[i])
	}
}

// geluDerivative calculates d(GELU(x))/dx for backpropagation.
func geluDerivative(x float32) float32 {
	cube := x * x * x
	u := Sqrt2OverPi * (x + GeluCoeff*cube)
	tanhU := float32(math.Tanh(float64(u)))
	du := Sqrt2OverPi * (1.0 + 3.0*GeluCoeff*x*x)
	sech2 := 1.0 - tanhU*tanhU
	return 0.5*(1.0+tanhU) + 0.5*x*sech2*du
}

// TrainModel executes the complete training pipeline including BPE, AdamW optimization, and Early Stopping.
func TrainModel(samples []DataSample, cfg TrainConfig) (*InferenceModel, error) {
	seed := cfg.Seed
	if seed == 0 {
		seed = 42
	}
	rng := rand.New(rand.NewSource(seed))

	// 1. Collect unique labels
	labelMap := make(map[string]uint32)
	var labels []string
	for _, s := range samples {
		if _, exists := labelMap[s.Label]; !exists {
			labelMap[s.Label] = uint32(len(labels))
			labels = append(labels, s.Label)
		}
	}
	numClasses := len(labels)
	if numClasses < 2 {
		return nil, fmt.Errorf("dataset must contain at least 2 distinct classes, found %d", numClasses)
	}

	// 2. Train Pure Go BPE Tokenizer
	corpus := make([]string, len(samples))
	for i, s := range samples {
		corpus[i] = s.Text
	}
	tokenizer, err := TrainBPE(corpus, cfg.TargetVocabSize)
	if err != nil {
		return nil, fmt.Errorf("failed to train BPE: %w", err)
	}
	vocabSize := tokenizer.VocabSize()

	// 3. Prepare Encoded Dataset
	type EncodedSample struct {
		tokens  []uint32
		classID uint32
	}
	classBuckets := make(map[uint32][]EncodedSample)
	for _, s := range samples {
		tokens := tokenizer.Encode(s.Text)
		if len(tokens) == 0 {
			tokens = []uint32{0}
		}
		cID := labelMap[s.Label]
		classBuckets[cID] = append(classBuckets[cID], EncodedSample{
			tokens:  tokens,
			classID: cID,
		})
	}

	var trainSet []EncodedSample
	var valSet []EncodedSample

	// Stratified split: ensure every class has representation in both train and validation
	for cID := uint32(0); cID < uint32(numClasses); cID++ {
		b := classBuckets[cID]
		rng.Shuffle(len(b), func(i, j int) {
			b[i], b[j] = b[j], b[i]
		})
		if len(b) <= 2 {
			// Very small class: keep in train, evaluate on same for sanity
			trainSet = append(trainSet, b...)
			valSet = append(valSet, b...)
		} else {
			valCnt := int(float64(len(b)) * 0.15)
			if valCnt < 1 {
				valCnt = 1
			}
			trainSet = append(trainSet, b[valCnt:]...)
			valSet = append(valSet, b[:valCnt]...)
		}
	}

	// 4. Initialize Network Weights using Xavier / He uniform initialization
	header := Header{
		Magic:        MagicBytes,
		Version:      CurrentFormatVersion,
		VocabSize:    uint32(vocabSize),
		EmbeddingDim: uint32(cfg.EmbeddingDim),
		HiddenDim:    uint32(cfg.HiddenDim),
		NumClasses:   uint32(numClasses),
	}

	weights := Weights{
		Embedding:  make([]float32, vocabSize*cfg.EmbeddingDim),
		Positional: make([]float32, MaxSequenceTokens*cfg.EmbeddingDim),
		W1:         make([]float32, cfg.EmbeddingDim*cfg.HiddenDim),
		B1:         make([]float32, cfg.HiddenDim),
		W2:         make([]float32, cfg.HiddenDim*numClasses),
		B2:         make([]float32, numClasses),
	}

	// Initialize weights
	embScale := float32(math.Sqrt(1.0 / float64(cfg.EmbeddingDim)))
	for i := range weights.Embedding {
		weights.Embedding[i] = (rng.Float32()*2.0 - 1.0) * embScale
	}
	for i := range weights.Positional {
		weights.Positional[i] = (rng.Float32()*2.0 - 1.0) * embScale
	}
	w1Scale := float32(math.Sqrt(2.0 / float64(cfg.EmbeddingDim)))
	for i := range weights.W1 {
		weights.W1[i] = (rng.Float32()*2.0 - 1.0) * w1Scale
	}
	w2Scale := float32(math.Sqrt(2.0 / float64(cfg.HiddenDim)))
	for i := range weights.W2 {
		weights.W2[i] = (rng.Float32()*2.0 - 1.0) * w2Scale
	}

	// Initialize AdamW Optimizers
	optEmb := newAdamWState(len(weights.Embedding))
	optPos := newAdamWState(len(weights.Positional))
	optW1 := newAdamWState(len(weights.W1))
	optB1 := newAdamWState(len(weights.B1))
	optW2 := newAdamWState(len(weights.W2))
	optB2 := newAdamWState(len(weights.B2))

	// Gradients Accumulator Buffers
	gradEmb := make([]float32, len(weights.Embedding))
	gradPos := make([]float32, len(weights.Positional))
	gradW1 := make([]float32, len(weights.W1))
	gradB1 := make([]float32, len(weights.B1))
	gradW2 := make([]float32, len(weights.W2))
	gradB2 := make([]float32, len(weights.B2))

	// Scratch buffers for forward-backward
	pooled := make([]float32, cfg.EmbeddingDim)
	z1 := make([]float32, cfg.HiddenDim)
	a1 := make([]float32, cfg.HiddenDim)
	z2 := make([]float32, numClasses)
	probs := make([]float32, numClasses)

	dZ2 := make([]float32, numClasses)
	dA1 := make([]float32, cfg.HiddenDim)
	dZ1 := make([]float32, cfg.HiddenDim)
	dMean := make([]float32, cfg.EmbeddingDim)

	bestValLoss := float32(math.MaxFloat32)
	bestTrainLoss := float32(math.MaxFloat32)
	patienceCounter := 0
	var bestWeights Weights

	batchSize := cfg.BatchSize
	if len(trainSet) <= 32 {
		batchSize = 4
	} else if batchSize > len(trainSet) {
		batchSize = len(trainSet)
	}
	if batchSize <= 0 {
		batchSize = 1
	}

	// 5. Training Loop
	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		// Shuffle trainSet
		rng.Shuffle(len(trainSet), func(i, j int) {
			trainSet[i], trainSet[j] = trainSet[j], trainSet[i]
		})

		// Reset gradients
		clearSlice(gradEmb)
		clearSlice(gradPos)
		clearSlice(gradW1)
		clearSlice(gradB1)
		clearSlice(gradW2)
		clearSlice(gradB2)

		accumCount := 0

		for idx, sample := range trainSet {
			// Forward Pass with Positional Encoding
			_ = MeanPoolingWithPos(sample.tokens, weights.Embedding, weights.Positional, cfg.EmbeddingDim, pooled)
			_ = MatMulVecAdd(pooled, weights.W1, weights.B1, cfg.EmbeddingDim, cfg.HiddenDim, z1)
			for i := 0; i < cfg.HiddenDim; i++ {
				a1[i] = GELU(z1[i])
			}
			_ = MatMulVecAdd(a1, weights.W2, weights.B2, cfg.HiddenDim, numClasses, z2)
			_ = Softmax(z2, 1.0, probs)

			// Backward Pass
			// 1. Loss gradient wrt z2: dZ2 = probs - y_onehot
			for c := 0; c < numClasses; c++ {
				dZ2[c] = probs[c]
			}
			dZ2[sample.classID] -= 1.0

			// 2. Gradients for Layer 2: gradW2 += a1^T * dZ2, gradB2 += dZ2
			for h := 0; h < cfg.HiddenDim; h++ {
				ah := a1[h]
				rowOffset := h * numClasses
				for c := 0; c < numClasses; c++ {
					gradW2[rowOffset+c] += ah * dZ2[c]
				}
			}
			for c := 0; c < numClasses; c++ {
				gradB2[c] += dZ2[c]
			}

			// 3. Backprop through Layer 2 to a1: dA1 = dZ2 * W2^T
			for h := 0; h < cfg.HiddenDim; h++ {
				var sum float32
				rowOffset := h * numClasses
				for c := 0; c < numClasses; c++ {
					sum += dZ2[c] * weights.W2[rowOffset+c]
				}
				dA1[h] = sum
			}

			// 4. Backprop through GELU activation: dZ1 = dA1 * GELU'(z1)
			for h := 0; h < cfg.HiddenDim; h++ {
				dZ1[h] = dA1[h] * geluDerivative(z1[h])
			}

			// 5. Gradients for Layer 1: gradW1 += pooled^T * dZ1, gradB1 += dZ1
			for e := 0; e < cfg.EmbeddingDim; e++ {
				pe := pooled[e]
				rowOffset := e * cfg.HiddenDim
				for h := 0; h < cfg.HiddenDim; h++ {
					gradW1[rowOffset+h] += pe * dZ1[h]
				}
			}
			for h := 0; h < cfg.HiddenDim; h++ {
				gradB1[h] += dZ1[h]
			}

			// 6. Backprop through Layer 1 to pooled embedding: dMean = dZ1 * W1^T
			for e := 0; e < cfg.EmbeddingDim; e++ {
				var sum float32
				rowOffset := e * cfg.HiddenDim
				for h := 0; h < cfg.HiddenDim; h++ {
					sum += dZ1[h] * weights.W1[rowOffset+h]
				}
				dMean[e] = sum
			}

			// 7. Backprop through Non-Linear Positional Mean Pooling
			invLen := 1.0 / float32(len(sample.tokens))
			for pos, tok := range sample.tokens {
				tokOffset := int(tok) * cfg.EmbeddingDim
				posOffset := pos * cfg.EmbeddingDim
				for e := 0; e < cfg.EmbeddingDim; e++ {
					sumVal := weights.Embedding[tokOffset+e] + weights.Positional[posOffset+e]
					g := dMean[e] * invLen * geluDerivative(sumVal)
					gradEmb[tokOffset+e] += g
					if posOffset+cfg.EmbeddingDim <= len(gradPos) {
						gradPos[posOffset+e] += g
					}
				}
			}

			accumCount++

			// Step AdamW on batch boundary or dataset end
			if accumCount%batchSize == 0 || idx == len(trainSet)-1 {
				scale := 1.0 / float32(accumCount)
				scaleSlice(gradEmb, scale)
				scaleSlice(gradPos, scale)
				scaleSlice(gradW1, scale)
				scaleSlice(gradB1, scale)
				scaleSlice(gradW2, scale)
				scaleSlice(gradB2, scale)

				optEmb.Step(weights.Embedding, gradEmb, cfg.LearningRate, 0.0, cfg.Beta1, cfg.Beta2, cfg.Epsilon)
				optPos.Step(weights.Positional, gradPos, cfg.LearningRate, 0.0, cfg.Beta1, cfg.Beta2, cfg.Epsilon)
				optW1.Step(weights.W1, gradW1, cfg.LearningRate, cfg.WeightDecay, cfg.Beta1, cfg.Beta2, cfg.Epsilon)
				optB1.Step(weights.B1, gradB1, cfg.LearningRate, 0.0, cfg.Beta1, cfg.Beta2, cfg.Epsilon)
				optW2.Step(weights.W2, gradW2, cfg.LearningRate, cfg.WeightDecay, cfg.Beta1, cfg.Beta2, cfg.Epsilon)
				optB2.Step(weights.B2, gradB2, cfg.LearningRate, 0.0, cfg.Beta1, cfg.Beta2, cfg.Epsilon)

				clearSlice(gradEmb)
				clearSlice(gradPos)
				clearSlice(gradW1)
				clearSlice(gradB1)
				clearSlice(gradW2)
				clearSlice(gradB2)
				accumCount = 0
			}
		}

		// Validation Evaluation
		var trainLoss float32
		var trainCorrect int
		for _, sample := range trainSet {
			_ = MeanPoolingWithPos(sample.tokens, weights.Embedding, weights.Positional, cfg.EmbeddingDim, pooled)
			_ = MatMulVecAdd(pooled, weights.W1, weights.B1, cfg.EmbeddingDim, cfg.HiddenDim, z1)
			for i := 0; i < cfg.HiddenDim; i++ {
				a1[i] = GELU(z1[i])
			}
			_ = MatMulVecAdd(a1, weights.W2, weights.B2, cfg.HiddenDim, numClasses, z2)
			_ = Softmax(z2, 1.0, probs)

			p := probs[sample.classID]
			if p < 1e-7 {
				p = 1e-7
			}
			trainLoss -= float32(math.Log(float64(p)))

			var pred uint32
			var maxP float32 = -1.0
			for c := 0; c < numClasses; c++ {
				if probs[c] > maxP {
					maxP = probs[c]
					pred = uint32(c)
				}
			}
			if pred == sample.classID {
				trainCorrect++
			}
		}
		trainLoss /= float32(len(trainSet))
		trainAcc := float32(trainCorrect) / float32(len(trainSet))

		var valLoss float32
		var valCorrect int
		for _, sample := range valSet {
			_ = MeanPoolingWithPos(sample.tokens, weights.Embedding, weights.Positional, cfg.EmbeddingDim, pooled)
			_ = MatMulVecAdd(pooled, weights.W1, weights.B1, cfg.EmbeddingDim, cfg.HiddenDim, z1)
			for i := 0; i < cfg.HiddenDim; i++ {
				a1[i] = GELU(z1[i])
			}
			_ = MatMulVecAdd(a1, weights.W2, weights.B2, cfg.HiddenDim, numClasses, z2)
			_ = Softmax(z2, 1.0, probs)

			// Cross entropy
			p := probs[sample.classID]
			if p < 1e-7 {
				p = 1e-7
			}
			valLoss -= float32(math.Log(float64(p)))

			// Predict argmax
			var pred uint32
			var maxP float32 = -1.0
			for c := 0; c < numClasses; c++ {
				if probs[c] > maxP {
					maxP = probs[c]
					pred = uint32(c)
				}
			}
			if pred == sample.classID {
				valCorrect++
			}
		}
		valLoss /= float32(len(valSet))
		valAcc := float32(valCorrect) / float32(len(valSet))

		// Check Best Model and Early Stopping
		isBetter := false
		if len(samples) < 50 {
			if trainLoss < bestTrainLoss {
				bestTrainLoss = trainLoss
				isBetter = true
			}
		} else {
			if valLoss < bestValLoss {
				bestValLoss = valLoss
				isBetter = true
			}
		}

		if isBetter {
			bestValLoss = valLoss
			patienceCounter = 0
			bestWeights = cloneWeights(weights)
		} else {
			patienceCounter++
			if patienceCounter >= cfg.Patience && epoch >= 30 {
				fmt.Printf("[Early Stopping] Triggered at epoch %d (Train Loss: %.4f, Val Loss: %.4f)\n", epoch, trainLoss, valLoss)
				break
			}
		}

		if epoch%10 == 0 || epoch == cfg.Epochs {
			fmt.Printf("Epoch %3d/%3d - Train Loss: %.4f (Acc: %.1f%%) | Val Loss: %.4f (Acc: %.1f%%)\n", epoch, cfg.Epochs, trainLoss, trainAcc*100.0, valLoss, valAcc*100.0)
		}
	}

	if bestWeights.Embedding == nil {
		bestWeights = weights
	}

	return NewInferenceModel(header, labels, tokenizer.Vocab, tokenizer.MergeRules, bestWeights), nil
}

func clearSlice(s []float32) {
	for i := range s {
		s[i] = 0.0
	}
}

func scaleSlice(s []float32, factor float32) {
	for i := range s {
		s[i] *= factor
	}
}

func cloneWeights(w Weights) Weights {
	cp := Weights{
		Embedding:  make([]float32, len(w.Embedding)),
		Positional: make([]float32, len(w.Positional)),
		W1:         make([]float32, len(w.W1)),
		B1:         make([]float32, len(w.B1)),
		W2:         make([]float32, len(w.W2)),
		B2:         make([]float32, len(w.B2)),
	}
	copy(cp.Embedding, w.Embedding)
	copy(cp.Positional, w.Positional)
	copy(cp.W1, w.W1)
	copy(cp.B1, w.B1)
	copy(cp.W2, w.W2)
	copy(cp.B2, w.B2)
	return cp
}
