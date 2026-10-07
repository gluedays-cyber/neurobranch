package neurobranch

import (
	"strings"
	"unicode/utf8"
)

// BPETokenizer implements subword segmentation via iterative byte-pair merge rules.
type BPETokenizer struct {
	Vocab      []string
	VocabMap   map[string]uint32
	MergeRules []MergeRule
	ruleLookup map[uint64]uint32 // (token1 << 32 | token2) -> targetToken
}

// NewBPETokenizer constructs a tokenizer from serialized vocabulary and merge rules.
func NewBPETokenizer(vocab []string, rules []MergeRule) *BPETokenizer {
	vocabMap := make(map[string]uint32, len(vocab))
	for idx, tok := range vocab {
		vocabMap[tok] = uint32(idx)
	}

	ruleLookup := make(map[uint64]uint32, len(rules))
	for _, r := range rules {
		key := (uint64(r.Token1) << 32) | uint64(r.Token2)
		ruleLookup[key] = r.Target
	}

	return &BPETokenizer{
		Vocab:      vocab,
		VocabMap:   vocabMap,
		MergeRules: rules,
		ruleLookup: ruleLookup,
	}
}

// TrainBPE learns a BPE vocabulary and merge rules from a raw text corpus.
func TrainBPE(corpus []string, targetVocabSize int) (*BPETokenizer, error) {
	return TrainBPEWithBaseVocab(corpus, targetVocabSize, nil)
}

// TrainBPEWithBaseVocab learns a BPE vocabulary with base vocabulary support to eliminate UNKs and prevent fragmentation.
func TrainBPEWithBaseVocab(corpus []string, targetVocabSize int, baseVocab []string) (*BPETokenizer, error) {
	if targetVocabSize < 10 {
		targetVocabSize = 10
	}

	vocabMap := make(map[string]uint32)
	var vocab []string

	// Special tokens
	specialTokens := []string{"[PAD]", "[UNK]"}
	for _, st := range specialTokens {
		vocabMap[st] = uint32(len(vocab))
		vocab = append(vocab, st)
	}

	// 1. Collect initial unique characters from corpus
	for _, text := range corpus {
		text = strings.TrimSpace(strings.ToLower(text))
		for _, r := range text {
			ch := string(r)
			if _, exists := vocabMap[ch]; !exists {
				vocabMap[ch] = uint32(len(vocab))
				vocab = append(vocab, ch)
			}
		}
	}

	// 2. Add base English alphabet characters (a-z, space) only if baseVocab is enabled
	// This guarantees zero UNKs when base vocab is activated while preserving exact small-corpus dynamics
	if len(baseVocab) > 0 {
		baseAlphabet := "abcdefghijklmnopqrstuvwxyz "
		for _, r := range baseAlphabet {
			ch := string(r)
			if _, exists := vocabMap[ch]; !exists {
				vocabMap[ch] = uint32(len(vocab))
				vocab = append(vocab, ch)
			}
		}
	}

	// 3. Tokenize corpus into character token ID sequences
	var tokenizedCorpus [][]uint32
	for _, text := range corpus {
		text = strings.TrimSpace(strings.ToLower(text))
		if len(text) == 0 {
			continue
		}
		var seq []uint32
		for _, r := range text {
			ch := string(r)
			if id, exists := vocabMap[ch]; exists {
				seq = append(seq, id)
			}
		}
		if len(seq) > 0 {
			tokenizedCorpus = append(tokenizedCorpus, seq)
		}
	}

	var mergeRules []MergeRule
	ruleLookup := make(map[uint64]uint32)

	// Build lookup for base vocabulary if provided
	baseVocabSet := make(map[string]bool)
	if len(baseVocab) > 0 {
		for _, w := range baseVocab {
			w = strings.TrimSpace(strings.ToLower(w))
			if len(w) > 1 {
				baseVocabSet[w] = true
			}
		}
	}

	// 4. Iteratively learn adjacent pair merges deterministically
	for len(vocab) < targetVocabSize {
		pairCounts := make(map[uint64]int)
		for _, seq := range tokenizedCorpus {
			for i := 0; i < len(seq)-1; i++ {
				key := (uint64(seq[i]) << 32) | uint64(seq[i+1])
				pairCounts[key]++
			}
		}
		if len(pairCounts) == 0 {
			break
		}

		var bestKey uint64
		var maxFreq int
		for key, freq := range pairCounts {
			score := freq
			if len(baseVocabSet) > 0 {
				t1 := uint32(key >> 32)
				t2 := uint32(key & 0xFFFFFFFF)
				candStr := vocab[t1] + vocab[t2]
				if baseVocabSet[candStr] {
					score += 50
				}
			}

			if score > maxFreq || (score == maxFreq && (bestKey == 0 || key < bestKey)) {
				maxFreq = score
				bestKey = key
			}
		}

		if maxFreq < 2 && len(vocab) >= targetVocabSize/2 {
			break
		}

		t1 := uint32(bestKey >> 32)
		t2 := uint32(bestKey & 0xFFFFFFFF)
		mergedStr := vocab[t1] + vocab[t2]

		newID := uint32(len(vocab))
		vocabMap[mergedStr] = newID
		vocab = append(vocab, mergedStr)

		rule := MergeRule{Token1: t1, Token2: t2, Target: newID}
		mergeRules = append(mergeRules, rule)
		ruleLookup[bestKey] = newID

		// Apply merge in-place across tokenized corpus
		for sIdx, seq := range tokenizedCorpus {
			var newSeq []uint32
			i := 0
			for i < len(seq) {
				if i < len(seq)-1 && seq[i] == t1 && seq[i+1] == t2 {
					newSeq = append(newSeq, newID)
					i += 2
				} else {
					newSeq = append(newSeq, seq[i])
					i++
				}
			}
			tokenizedCorpus[sIdx] = newSeq
		}
	}

	return &BPETokenizer{
		Vocab:      vocab,
		VocabMap:   vocabMap,
		MergeRules: mergeRules,
		ruleLookup: ruleLookup,
	}, nil
}

// Encode converts input text into subword token IDs using the learned merge rules.
func (t *BPETokenizer) Encode(text string) []uint32 {
	text = strings.TrimSpace(strings.ToLower(text))
	if len(text) == 0 {
		return nil
	}

	// Fast path: if exact text exists directly in vocabulary (e.g., atomic words or sample models)
	if id, exists := t.VocabMap[text]; exists {
		return []uint32{id}
	}

	// 1. Initial character split
	var tokens []uint32
	unkID, hasUnk := t.VocabMap["[UNK]"]
	for _, r := range text {
		ch := string(r)
		if id, exists := t.VocabMap[ch]; exists {
			tokens = append(tokens, id)
		} else if hasUnk {
			tokens = append(tokens, unkID)
		} else {
			// Fallback to index 0 ([PAD]) if [UNK] is not defined in model header
			tokens = append(tokens, 0)
		}
	}

	if len(tokens) <= 1 {
		return tokens
	}

	// 2. Iteratively apply merge rules until no pairs can be merged
	for {
		merged := false
		var nextTokens []uint32
		i := 0
		for i < len(tokens) {
			if i < len(tokens)-1 {
				key := (uint64(tokens[i]) << 32) | uint64(tokens[i+1])
				if target, exists := t.ruleLookup[key]; exists {
					nextTokens = append(nextTokens, target)
					i += 2
					merged = true
					continue
				}
			}
			nextTokens = append(nextTokens, tokens[i])
			i++
		}
		tokens = nextTokens
		if !merged {
			break
		}
	}

	return tokens
}

// Decode transforms a sequence of token IDs back into readable text.
func (t *BPETokenizer) Decode(tokens []uint32) string {
	var sb strings.Builder
	for _, tok := range tokens {
		if int(tok) < len(t.Vocab) {
			str := t.Vocab[tok]
			if str != "[PAD]" && str != "[UNK]" {
				sb.WriteString(str)
			}
		}
	}
	return sb.String()
}

// IsValidUtf8 checks string validity.
func IsValidUtf8(s string) bool {
	return utf8.ValidString(s)
}

// VocabSize returns the count of registered subwords.
func (t *BPETokenizer) VocabSize() int {
	return len(t.Vocab)
}

// AnalyzeUnlearnedRatio computes the ratio of single-character fallback tokens and unknown tokens without allocations.
func (t *BPETokenizer) AnalyzeUnlearnedRatio(tokens []uint32) (singleCharRatio float64, unkRatio float64) {
	n := len(tokens)
	if n == 0 {
		return 1.0, 1.0
	}

	unkID, hasUnk := t.VocabMap["[UNK]"]
	singleCharCount := 0
	unkCount := 0

	for _, id := range tokens {
		if hasUnk && id == unkID {
			unkCount++
			continue
		}
		if int(id) < len(t.Vocab) {
			str := t.Vocab[id]
			if str != "[PAD]" && utf8.RuneCountInString(str) == 1 {
				singleCharCount++
			}
		}
	}

	total := float64(n)
	return float64(singleCharCount) / total, float64(unkCount) / total
}

// IsUnlearned determines if token sequence is dominated by unlearned single-character fragments or UNKs.
func (t *BPETokenizer) IsUnlearned(tokens []uint32, maxSingleRatio float64, maxUnkRatio float64) bool {
	if len(tokens) == 0 {
		return true
	}
	singleRatio, unkRatio := t.AnalyzeUnlearnedRatio(tokens)
	if maxSingleRatio > 0.0 && singleRatio >= maxSingleRatio {
		return true
	}
	if maxUnkRatio > 0.0 && unkRatio >= maxUnkRatio {
		return true
	}
	return false
}

// ScanUnlearnedPatterns scans text with zero allocations for non-semantic random hex/base64 sequences.
func ScanUnlearnedPatterns(text string) bool {
	n := len(text)
	if n < 6 {
		return false
	}

	for i := 0; i < n; i++ {
		// 1. Hex sequence detection: 0x or 0X followed by >= 4 hex characters
		if i+1 < n && text[i] == '0' && (text[i+1] == 'x' || text[i+1] == 'X') {
			hexCount := 0
			for j := i + 2; j < n; j++ {
				c := text[j]
				if (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F') {
					hexCount++
				} else {
					break
				}
			}
			if hexCount >= 4 {
				return true
			}
		}

		// 2. Continuous uppercase hex block (e.g. DEADBEEF, A1B2C3D4 >= 8 chars)
		c := text[i]
		if (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') {
			hexCount := 1
			hasAlpha := (c >= 'A' && c <= 'F')
			j := i + 1
			for ; j < n; j++ {
				cj := text[j]
				if (cj >= '0' && cj <= '9') || (cj >= 'A' && cj <= 'F') {
					hexCount++
					if cj >= 'A' && cj <= 'F' {
						hasAlpha = true
					}
				} else {
					break
				}
			}
			if hexCount >= 8 && hasAlpha {
				return true
			}
			i = j
		}
	}
	return false
}
