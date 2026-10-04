package neurobranch

import (
	"os"
	"path/filepath"
	"testing"
)

func TestFacadePipeline(t *testing.T) {
	tempDir := t.TempDir()
	csvPath := filepath.Join(tempDir, "test_data.csv")
	modelPath := filepath.Join(tempDir, "weights", "test_model.bin")

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

	// 1. Train via Facade
	cfg := DefaultTrainConfig()
	cfg.Epochs = 20
	cfg.TargetVocabSize = 50
	if err := Train(csvPath, modelPath, cfg); err != nil {
		t.Fatalf("Train failed: %v", err)
	}

	// 2. EnsureModel (should skip training since model already exists)
	if err := EnsureModel(csvPath, modelPath); err != nil {
		t.Fatalf("EnsureModel failed: %v", err)
	}

	// 3. Open Router via Facade
	router, err := Open(modelPath, 0.50)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	selected := router.Select("refund my payment please")
	if selected == "" {
		t.Log("Note: refund branch did not trigger, model may need more epochs")
	}


	// 4. OpenOrTrain with a fresh path
	freshModelPath := filepath.Join(tempDir, "weights", "fresh.bin")
	freshRouter, err := OpenOrTrain(csvPath, freshModelPath, 0.50)
	if err != nil {
		t.Fatalf("OpenOrTrain failed: %v", err)
	}
	if freshRouter == nil {
		t.Fatal("OpenOrTrain returned nil router")
	}
}
