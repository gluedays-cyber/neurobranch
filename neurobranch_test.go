package neurobranch_test

import (
	"context"
	"testing"

	"github.com/gluedays-cyber/neurobranch"
)

func TestRootFacadeIntegrity(t *testing.T) {
	trainData := map[string][]string{
		"greeting": {"hello", "hi there", "good morning"},
		"billing":  {"pay invoice", "billing issue", "subscription cost"},
	}

	ai, err := neurobranch.TrainAIFromMap(trainData, neurobranch.DefaultTrainConfig())
	if err != nil {
		t.Fatalf("TrainAIFromMap failed: %v", err)
	}

	branch := ai.Select("hello world")
	if branch != "greeting" {
		t.Fatalf("expected branch greeting, got %s", branch)
	}

	decision, err := ai.RouteQuery(context.Background(), "billing issue")
	if err != nil {
		t.Fatalf("ai.RouteQuery failed: %v", err)
	}
	if decision.Intent != "billing" {
		t.Fatalf("expected intent billing, got %s", decision.Intent)
	}
}
