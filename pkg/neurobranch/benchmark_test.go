package neurobranch_test

import (
	_ "embed"
	"encoding/csv"
	"strings"
	"testing"

	"github.com/gluedays-cyber/neurobranch/pkg/neurobranch"
)

//go:embed testdata/enterprise_train.csv
var embeddedEnterpriseCSV string

func loadSamplesFromEmbeddedCSV(t testing.TB) []neurobranch.DataSample {
	reader := csv.NewReader(strings.NewReader(embeddedEnterpriseCSV))
	records, err := reader.ReadAll()
	if err != nil {
		t.Fatalf("failed to read embedded csv: %v", err)
	}
	if len(records) < 2 {
		t.Fatalf("csv contains insufficient rows")
	}

	var samples []neurobranch.DataSample
	for i := 1; i < len(records); i++ {
		row := records[i]
		if len(row) >= 2 {
			samples = append(samples, neurobranch.DataSample{
				Text:  strings.TrimSpace(row[0]),
				Label: strings.TrimSpace(row[1]),
			})
		}
	}
	return samples
}

// BenchmarkRoutingThroughput evaluates parallel query routing throughput on diverse query types.
func BenchmarkRoutingThroughput(b *testing.B) {
	samples := loadSamplesFromEmbeddedCSV(b)

	cfg := neurobranch.DefaultTrainConfig()
	cfg.Epochs = 40
	cfg.Seed = 42

	ai, err := neurobranch.TrainInMemory(samples, cfg)
	if err != nil {
		b.Fatalf("TrainInMemory failed: %v", err)
	}

	queries := []string{
		"courier left note but no box at door",
		"2fa auth code not sending to phone",
		"send recipt to my email plz",
		"money back request for damaged shipment",
		"tell me the capital city of France",
		"quantum physics entangled photon spin",
	}

	b.ResetTimer()
	b.ReportAllocs()

	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			q := queries[i%len(queries)]
			_ = ai.Select(q)
			i++
		}
	})
}

// BenchmarkSelectLatency measures individual sequential Select latency.
func BenchmarkSelectLatency(b *testing.B) {
	samples := loadSamplesFromEmbeddedCSV(b)

	cfg := neurobranch.DefaultTrainConfig()
	cfg.Epochs = 40
	cfg.Seed = 42

	ai, err := neurobranch.TrainInMemory(samples, cfg)
	if err != nil {
		b.Fatalf("TrainInMemory failed: %v", err)
	}

	query := "where is my package delivery tracking"

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = ai.Select(query)
	}
}

// BenchmarkOptionConfiguration verifies runtime overhead of applying functional options.
func BenchmarkOptionConfiguration(b *testing.B) {
	samples := loadSamplesFromEmbeddedCSV(b)

	cfg := neurobranch.DefaultTrainConfig()
	cfg.Epochs = 40
	cfg.Seed = 42

	ai, err := neurobranch.TrainWithOptions(
		samples,
		cfg,
		neurobranch.WithEnergyThreshold(3.0),
		neurobranch.WithConfidenceThreshold(0.75, 0.40),
		neurobranch.WithMarginCutoff(0.15),
	)
	if err != nil {
		b.Fatalf("TrainWithOptions failed: %v", err)
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		ai.SetEnergyThreshold(3.0 + float64(i%5)*0.1)
	}
}
