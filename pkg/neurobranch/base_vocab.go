package neurobranch

import (
	_ "embed"
	"strings"
	"sync"
)

//go:embed vocab/base_vocab.txt
var rawBaseVocab string

var (
	baseVocabOnce  sync.Once
	baseVocabList  []string
	baseCorpusList []string
)

// DefaultBaseVocab returns the pre-loaded general base vocabulary to avoid fragmentation and UNKs.
func DefaultBaseVocab() []string {
	baseVocabOnce.Do(initBaseResources)
	res := make([]string, len(baseVocabList))
	copy(res, baseVocabList)
	return res
}

// DefaultBaseCorpus returns standard foundational sentences containing core domain vocabulary for BPE pre-training.
func DefaultBaseCorpus() []string {
	baseVocabOnce.Do(initBaseResources)
	res := make([]string, len(baseCorpusList))
	copy(res, baseCorpusList)
	return res
}

func initBaseResources() {
	lines := strings.Split(strings.ReplaceAll(rawBaseVocab, "\r\n", "\n"), "\n")
	seen := make(map[string]bool, len(lines))
	for _, line := range lines {
		line = strings.TrimSpace(strings.ToLower(line))
		if line == "" || seen[line] {
			continue
		}
		seen[line] = true
		baseVocabList = append(baseVocabList, line)
	}

	baseCorpusList = []string{
		"order refund and cancel return request for purchase item",
		"where is my package tracking number and delivery status",
		"track courier parcel shipment location and arrival time",
		"customer tech support service and account profile help",
		"reset user account login password and credentials",
		"payment credit card billing monthly invoice tax receipt",
		"change delivery address and confirm details please",
		"i want my money back please issue refund right now",
		"stop the shipment and do not deliver cancel order",
		"tell me the capital city of france paris weather today is nice",
		"what when where why how who which can could would should",
	}
}
