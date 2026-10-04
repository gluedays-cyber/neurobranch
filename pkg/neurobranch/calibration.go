package neurobranch

import (
	"math"
)

// computeBaselineCentroid derives an initial normalized centroid from the model's vocabulary embeddings.
func (g *NeuroBranch) computeBaselineCentroid(model *InferenceModel) {
	embDim := int(model.Header.EmbeddingDim)
	if embDim > MaxGateEmbDim || embDim == 0 || len(model.Weights.Embedding) == 0 {
		return
	}

	var sum [MaxGateEmbDim]float64
	validTokens := 0
	for tokID, word := range model.Vocab {
		if word == "[PAD]" || word == "[UNK]" {
			continue
		}
		offset := tokID * embDim
		if offset+embDim <= len(model.Weights.Embedding) {
			for d := 0; d < embDim; d++ {
				sum[d] += float64(model.Weights.Embedding[offset+d])
			}
			validTokens++
		}
	}

	if validTokens > 0 {
		inv := 1.0 / float64(validTokens)
		var raw [MaxGateEmbDim]float32
		for d := 0; d < embDim; d++ {
			raw[d] = float32(sum[d] * inv)
		}
		L2Normalize(raw[:embDim], g.domainCentroid[:embDim])
		g.hasCentroid = true
	}
}

// CalibrateDomainCentroid calculates the true manifold center from sample dataset sentences.
func (g *NeuroBranch) CalibrateDomainCentroid(samples []DataSample) *NeuroBranch {
	g.mu.Lock()
	defer g.mu.Unlock()

	model := g.model.Load()
	if model == nil || len(samples) == 0 {
		return g
	}

	embDim := int(model.Header.EmbeddingDim)
	if embDim > MaxGateEmbDim {
		embDim = MaxGateEmbDim
	}

	var sum [MaxGateEmbDim]float64
	validCount := 0
	var pooled [MaxGateEmbDim]float32

	var dummyLogits [MaxGateClasses]float32
	for _, s := range samples {
		tokens := model.Tokenizer.Encode(s.Text)
		if len(tokens) == 0 {
			continue
		}
		if err := model.PredictFeatures(tokens, pooled[:embDim], dummyLogits[:g.classCount]); err == nil {
			for d := 0; d < embDim; d++ {
				sum[d] += float64(pooled[d])
			}
			validCount++
		}
	}

	if validCount > 0 {
		inv := 1.0 / float64(validCount)
		var raw [MaxGateEmbDim]float32
		for d := 0; d < embDim; d++ {
			raw[d] = float32(sum[d] * inv)
		}
		L2Normalize(raw[:embDim], g.domainCentroid[:embDim])
		g.hasCentroid = true
	}
	return g
}

// CalibrateDomainDistribution calculates the manifold center and dynamically computes
// the standard deviation of cosine similarities across sample embeddings to configure an adaptive OOD threshold:
// minCosine = mean - (k * stdDev).
func (g *NeuroBranch) CalibrateDomainDistribution(samples []DataSample, k float32) *NeuroBranch {
	g.mu.Lock()
	defer g.mu.Unlock()

	model := g.model.Load()
	if model == nil || len(samples) == 0 {
		return g
	}

	embDim := int(model.Header.EmbeddingDim)
	if embDim > MaxGateEmbDim {
		embDim = MaxGateEmbDim
	}

	var sum [MaxGateEmbDim]float64
	validCount := 0
	var pooled [MaxGateEmbDim]float32
	var dummyLogits [MaxGateClasses]float32

	for _, s := range samples {
		tokens := model.Tokenizer.Encode(s.Text)
		if len(tokens) == 0 {
			continue
		}
		if err := model.PredictFeatures(tokens, pooled[:embDim], dummyLogits[:g.classCount]); err == nil {
			for d := 0; d < embDim; d++ {
				sum[d] += float64(pooled[d])
			}
			validCount++
		}
	}

	if validCount == 0 {
		return g
	}

	inv := 1.0 / float64(validCount)
	var raw [MaxGateEmbDim]float32
	for d := 0; d < embDim; d++ {
		raw[d] = float32(sum[d] * inv)
	}
	L2Normalize(raw[:embDim], g.domainCentroid[:embDim])
	g.hasCentroid = true

	// Pass 2: Calculate distribution variance and std dev of cosine similarities and LogSumExp energies
	var sumSim, sumSqSim float64
	var sumEnergy, sumSqEnergy float64
	evalCount := 0
	var normPooled [MaxGateEmbDim]float32
	for _, s := range samples {
		tokens := model.Tokenizer.Encode(s.Text)
		if len(tokens) == 0 {
			continue
		}
		if err := model.PredictFeatures(tokens, pooled[:embDim], dummyLogits[:g.classCount]); err == nil {
			L2Normalize(pooled[:embDim], normPooled[:embDim])
			sim := DotProduct(normPooled[:embDim], g.domainCentroid[:embDim])
			sumSim += float64(sim)
			sumSqSim += float64(sim * sim)

			energy := LogSumExp(dummyLogits[:g.classCount])
			sumEnergy += float64(energy)
			sumSqEnergy += float64(energy * energy)
			evalCount++
		}
	}

	if evalCount > 0 {
		n := float64(evalCount)
		mean := float32(sumSim / n)
		variance := float32((sumSqSim / n) - float64(mean*mean))
		if variance < 0 {
			variance = 0
		}
		stdDev := float32(math.Sqrt(float64(variance)))
		g.domainMeanSim = mean
		g.domainStdDev = stdDev

		adaptiveMin := mean - (k * stdDev)
		if adaptiveMin < -1.0 {
			adaptiveMin = -1.0
		}
		g.minCosineSim = adaptiveMin

		// Compute adaptive LogSumExp energy boundary
		meanEnergy := float32(sumEnergy / n)
		varianceEnergy := float32((sumSqEnergy / n) - float64(meanEnergy*meanEnergy))
		if varianceEnergy < 0 {
			varianceEnergy = 0
		}
		stdDevEnergy := float32(math.Sqrt(float64(varianceEnergy)))
		adaptiveEnergy := meanEnergy - (k * stdDevEnergy)
		if adaptiveEnergy > 0 {
			g.policy.MinLogSumExp = float64(adaptiveEnergy)
		}
	}

	// Pass 3: Calculate Inter-Class Boundary Distance and Adaptive RawLogitMargin
	classSums := make([][MaxGateEmbDim]float64, g.classCount)
	classCounts := make([]int, g.classCount)
	for _, s := range samples {
		cIdx, exists := g.labelToIndex[s.Label]
		if !exists || cIdx >= g.classCount {
			continue
		}
		tokens := model.Tokenizer.Encode(s.Text)
		if len(tokens) == 0 {
			continue
		}
		if err := model.PredictFeatures(tokens, pooled[:embDim], dummyLogits[:g.classCount]); err == nil {
			for d := 0; d < embDim; d++ {
				classSums[cIdx][d] += float64(pooled[d])
			}
			classCounts[cIdx]++
		}
	}

	classCentroids := make([][MaxGateEmbDim]float32, g.classCount)
	for c := 0; c < g.classCount; c++ {
		if classCounts[c] > 0 {
			invC := 1.0 / float64(classCounts[c])
			var rawC [MaxGateEmbDim]float32
			for d := 0; d < embDim; d++ {
				rawC[d] = float32(classSums[c][d] * invC)
			}
			L2Normalize(rawC[:embDim], classCentroids[c][:embDim])
		}
	}

	minDist := float32(2.0)
	validPairs := 0
	for i := 0; i < g.classCount; i++ {
		if classCounts[i] == 0 {
			continue
		}
		for j := i + 1; j < g.classCount; j++ {
			if classCounts[j] == 0 {
				continue
			}
			sim := DotProduct(classCentroids[i][:embDim], classCentroids[j][:embDim])
			dist := 1.0 - sim
			if dist < minDist {
				minDist = dist
			}
			validPairs++
		}
	}

	if validPairs > 0 {
		// Closely clustered classes require a stricter logit margin to disambiguate
		adaptiveMargin := float32(0.50) * (2.0 - minDist)
		if adaptiveMargin < 0.25 {
			adaptiveMargin = 0.25
		}
		if adaptiveMargin > 1.20 {
			adaptiveMargin = 1.20
		}
		g.policy.RawLogitMargin = adaptiveMargin
	}

	// Persist self-calibration metadata to model Header for zero-configuration portability
	if model != nil {
		model.Header.Version = FormatVersion3
		model.Header.CalibratedMinEnergy = float32(g.policy.MinLogSumExp)
		model.Header.CalibratedMargin = g.policy.RawLogitMargin
		model.Header.CalibratedMinCosine = g.minCosineSim
	}

	return g
}

// DomainStats returns the calibrated manifold distribution metrics.
func (g *NeuroBranch) DomainStats() (hasCentroid bool, meanSim float32, stdDev float32, minCosine float32) {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.hasCentroid, g.domainMeanSim, g.domainStdDev, g.minCosineSim
}
