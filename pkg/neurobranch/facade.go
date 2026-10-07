package neurobranch

import (
	"fmt"
	"os"
	"path/filepath"
)

// Train executes the complete pipeline: dataset loading, BPE tokenization,
// AdamW neural training, directory creation, and Little-Endian binary model serialization.
func Train(csvPath string, outputPath string, configs ...TrainConfig) error {
	samples, err := LoadCSVDataset(csvPath)
	if err != nil {
		return fmt.Errorf("failed to load dataset from %s: %w", csvPath, err)
	}

	cfg := DefaultTrainConfig()
	if len(configs) > 0 {
		cfg = configs[0]
	}

	model, err := TrainModel(samples, cfg)
	if err != nil {
		return fmt.Errorf("model training failed: %w", err)
	}

	outDir := filepath.Dir(outputPath)
	if outDir != "" && outDir != "." {
		if err := os.MkdirAll(outDir, 0755); err != nil {
			return fmt.Errorf("failed to create output directory %s: %w", outDir, err)
		}
	}

	if err := SaveBinaryModel(outputPath, model); err != nil {
		return fmt.Errorf("failed to save binary model to %s: %w", outputPath, err)
	}

	return nil
}

// EnsureModel guarantees the binary model exists at modelPath.
// If it does not exist, it trains a new model from csvPath and serializes it.
func EnsureModel(csvPath string, modelPath string, configs ...TrainConfig) error {
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return Train(csvPath, modelPath, configs...)
	}
	return nil
}

// Open loads an existing compiled binary model file into memory and creates an in-memory Router.
// If threshold is omitted, the default calibrated threshold of 0.60 is applied.
func Open(modelPath string, defaultThreshold ...float64) (*Router, error) {
	threshold := 0.60
	if len(defaultThreshold) > 0 && defaultThreshold[0] > 0.0 {
		threshold = defaultThreshold[0]
	}
	return NewRouter(modelPath, threshold)
}

// OpenOrTrain ensures the binary model is trained from csvPath if missing,
// then loads it directly into an active in-memory Router for intelligent branching.
// It also attaches the training dataset to enable live AppendData/AppendDataMap retraining.
func OpenOrTrain(csvPath string, modelPath string, defaultThreshold ...float64) (*Router, error) {
	if err := EnsureModel(csvPath, modelPath); err != nil {
		return nil, fmt.Errorf("failed to ensure model: %w", err)
	}
	router, err := Open(modelPath, defaultThreshold...)
	if err != nil {
		return nil, err
	}
	if samples, err := LoadCSVDataset(csvPath); err == nil {
		router.samples = samples
	}
	return router, nil
}

// TrainInMemory executes the complete training pipeline entirely in memory without writing any files to disk,
// returning an active, fully initialized Router ready for zero-latency branching.
func TrainInMemory(samples []DataSample, configs ...TrainConfig) (*Router, error) {
	cfg := DefaultTrainConfig()
	if len(configs) > 0 {
		cfg = configs[0]
	}

	model, err := TrainModel(samples, cfg)
	if err != nil {
		return nil, fmt.Errorf("in-memory model training failed: %w", err)
	}

	router := NewRouterFromModel(model)
	router.samples = append([]DataSample(nil), samples...)
	return router, nil
}

// TrainFromMap creates an active Router directly from a label-to-phrases map with zero file dependencies.
// Example:
//
//	router, err := neurobranch.TrainFromMap(map[string][]string{
//	    "order_refund": {"refund my money", "need refund", "give cash back"},
//	    "order_cancel": {"cancel order", "stop shipment", "abort purchase"},
//	})
func TrainFromMap(data map[string][]string, configs ...TrainConfig) (*Router, error) {
	var samples []DataSample
	for label, phrases := range data {
		for _, phrase := range phrases {
			samples = append(samples, DataSample{Text: phrase, Label: label})
		}
	}
	return TrainInMemory(samples, configs...)
}

// TrainWithOptions compiles an in-memory Router with initial training config and functional options applied.
func TrainWithOptions(samples []DataSample, cfg TrainConfig, opts ...Option) (*Router, error) {
	router, err := TrainInMemory(samples, cfg)
	if err != nil {
		return nil, err
	}
	router.ApplyOptions(opts...)
	return router, nil
}

// TrainAIWithOptions is an alias for TrainWithOptions for domain AI compilation.
var TrainAIWithOptions = TrainWithOptions

// OpenWithOptions loads a binary model and applies functional options.
func OpenWithOptions(modelPath string, opts ...Option) (*Router, error) {
	router, err := Open(modelPath)
	if err != nil {
		return nil, err
	}
	router.ApplyOptions(opts...)
	return router, nil
}

// OpenAIWithOptions is an alias for OpenWithOptions.
var OpenAIWithOptions = OpenWithOptions

// TrainAI is an alias for TrainInMemory, emphasizing that a domain artificial intelligence engine is compiled.
var TrainAI = TrainInMemory

// TrainAIFromMap compiles a domain AI engine directly from a label-to-phrases map in memory.
// Example:
//
//	ai, err := neurobranch.TrainAIFromMap(map[string][]string{
//	    "order_refund": {"refund my money", "need refund"},
//	})
//	switch ai.Select(query) {
//	case "order_refund": ...
//	}
var TrainAIFromMap = TrainFromMap

// OpenAI is an alias for Open, emphasizing that the loaded binary is an in-memory AI engine.
var OpenAI = Open
