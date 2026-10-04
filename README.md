# NeuroBranch

<p align="center">
  <img src="assets/neurobranch-hero.jpg" width="100%" alt="NeuroBranch vs Retro Branching — Electric Hyperbike vs Rusty Bicycle">
</p>
<p align="center">
  <strong>Directly Creates and Runs Its Own Neural AI in Pure Go</strong><br>
  <em>Stop borrowing third-party AIs. This engine creates its own domain artificial intelligence from scratch in 1.5 seconds, routing execution flow in ~30 μs with 0 B/op (Zero Allocations), Zero Downloads, and Zero CGO.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Release-v2.8.0_Modular-purple.svg" alt="Release v2.8.0">
  <a href="#benchmarks"><img src="https://img.shields.io/badge/Latency-~30_μs-brightgreen.svg" alt="Latency"></a>
  <a href="#benchmarks"><img src="https://img.shields.io/badge/Allocs-0_B/op_(0_allocs)-blue.svg" alt="Allocations"></a>
  <img src="https://img.shields.io/badge/Wire_Format-v3_Self--Calibrating-orange.svg" alt="Format v3">
  <img src="https://img.shields.io/badge/CGO-Zero_Disabled-success.svg" alt="CGO Zero">
  <img src="https://img.shields.io/badge/Go-1.21+-00ADD8.svg" alt="Go Version">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey.svg" alt="License">
</p>

<p align="center">
  <a href="docs/MANUAL.md"><strong>📖 Read the Full Developer Manual & Production Tutorial →</strong></a>
</p>

---

## What is NeuroBranch?

**NeuroBranch is built upon two uncompromising mandates:**
1. **Overcome the Inherent Failures of Classical Logical Branching**: Traditional `if/switch` logic relies on brittle string equality or regex patterns, inevitably collapsing under real-world typos, synonyms, colloquial phrasing, and unseen semantic noise. NeuroBranch replaces fragile pattern matching with in-memory neural inference while preserving mathematical determinism and fail-safe cutoffs.
2. **Zero Learning Curve for Logical Branching Veterans**: Developers already familiar with standard control flow do not need to learn esoteric AI frameworks or complex routing abstractions. NeuroBranch directly maps intelligent neural decisions into native Go idioms (`switch`, `if`, `comma-ok`, and declarative chaining).

Instead of relying on brittle regex matching or calling bloated external LLMs, it **manufactures a domain-specific lightweight neural network directly from your dataset in under 2 seconds**. It maps typos, slang, inverted syntax, and colloquial phrasing into a continuous latent vector space—routing execution flow directly through standard Go branching syntax in **microseconds (~30 μs) with strictly 0 B/op heap allocation**.

```
Incoming Request ("bruh can u refund order #49281")
                     │
                     ▼
       [ In-Memory BPE Tokenizer ]
                     │
       ┌─────────────┴────────────────────────────────┐
       │ [Layer 1 Guard]: Fast Pre-Neural Fail-Safe   │ ──> SingleCharRatio ≥ 0.70 (Unlearned OOV)
       │ (< 1 μs, Zero Forward Math Computation)      │     OR UniqueTokenRatio < 0.25 (Flood Abuse)
       │                                              │     --> Immediate ErrUnlearned / ErrDegeneratedInput
       └─────────────┬────────────────────────────────┘
                     │ (Learned Subwords Validated)
                     ▼
  [ Dense (D=64) + Positional (P=32x64) ]
                     │
                     ▼
  [ Non-Linear GELU Mean Pooling (D=64) ]
                     │
                     ▼
      [ Hidden Projection (D=128) ]
                     │
                     ▼
  [ Free Energy (LogSumExp) + Softmax Entropy Guard ]
                     │
       ┌─────────────┴────────────────────────────────┐
       │ [Layer 2 Guard]: Multi-Metric Neural Cutoff  │ ──> Low Energy / High Entropy?
       │ Confidence, Entropy, Margin (~29 μs)         │     --> ErrOutOfDomain / ErrLowConfidence
       └─────────────┬────────────────────────────────┘
                     │
     ┌───────────────┼───────────────┬────────────────┐
     ▼               ▼               ▼                ▼
(Score ≥ 0.75)  (Score ≥ 0.30)  (Margin < 0.15)  (Low Energy / OOD)
[Select: target] [Match: ok]     [ErrAmbiguous]   [ErrOutOfDomain]
     │               │               │                │
     ▼               ▼               ▼                ▼
switch target {  if ok { ... }   fall-through     default clause
```

---

## Why NeuroBranch? (Beyond Retro Branching, Cloud LLMs, and Bloated Local Models)

Modern backends face an architectural dilemma when routing unstructured or noisy user requests:

```go
// ❌ RETRO BRANCHING: Brittle, explodes in complexity, collapses under real-world noise
if strings.Contains(input, "refund") || strings.Contains(input, "cancel") {
    // FAILS on: "sent the return box a week ago when do i get my money back"
    // FAILS on: "can u reverse the charge?" (typos, slang, synonyms)
    // MISROUTES on: "cancel shipment delay notifications" (word collision)
}

// ❌ CLOUD LLMs: Massive network latency, recurring per-token cost, third-party dependency
// Latency: 400ms – 2,500ms (Unusable in high-throughput microservices)
// Cost: $0.0015 – $0.03 per request (Bills explode under scale)
// Vulnerability: Outages, rate limits, JSON hallucination, network partitions

// ❌ LOCAL LLMs & SLMs (Ollama, llama.cpp, Mistral-7B, Phi-3): Severe host resource exhaustion
// Memory: Monopolizes 4.5 GB to 8.0 GB+ of RAM/VRAM just to pick a 4-byte enum
// CPU Starvation: Burns 100% CPU across multiple cores, starving companion microservices
// Deployment Complexity: Requires CGO, C++ shared libraries (libllama.so), or background daemons

// ✅ NEUROBRANCH: Self-Generated Micro-AI (In-Memory Go Engine)
// Memory Footprint: Under 180 KB (25,000x smaller than quantized 7B models)
// Latency: ~30 μs with 0 B/op (0 allocs) and deterministic 3-tier fallback
// Deployment: 100% Pure Go with CGO_ENABLED=0 single static binary
```

### Architectural Comparison Matrix

| Capability | Retro Branching (`if` / Regex) | Cloud LLMs (OpenAI / Claude) | Local LLMs (Ollama / llama.cpp) | **NeuroBranch v2.0 (Embedded Engine)** |
| :--- | :--- | :--- | :--- | :--- |
| **Inference Latency** | < 1 μs | 300 ms – 2,500 ms (Network bound) | 30 ms – 300 ms (Compute bound) | **~30 μs (In-Memory)** |
| **Throughput (per core)** | > 500,000 req/sec | ~50 req/sec (Rate limited) | ~20–50 req/sec (CPU saturated) | **> 33,000 req/sec (Zero Alloc)** |
| **Runtime Allocation** | 0 B/op | High (HTTP payload) | High (CGO buffers) | **0 B/op (0 allocs/op)** |
| **System Memory (RAM)** | Negligible | External service | **4.5 GB – 8.0 GB+ (VRAM / RAM)** | **< 180 KB (Format v3)** |
| **Token Order Awareness** | Rigid regex position | ✅ Transformer Attention | ✅ Transformer Attention | ✅ **Learned Positional Embeddings** |
| **Hardware Reqs** | Standard CPU | External service | High-end GPU or 8+ Core CPU | **Runs on a $5 VPS (16MB container)** |
| **Operational Cost** | $0.00 | $0.0015+ per call | High hardware/electricity cost | **$0.00 (Self-contained)** |
| **Hot Weight Reload** | Binary recompile | API model string switch | Multi-second model reload | **Lock-free Atomic Hot-Swap (`0 ns` stop)** |
| **Active Learning Loop** | N/A | Manual logging | N/A | **Built-in Ring Buffer Telemetry** |
| **Deployment Complexity** | Single binary | API client | CGO / C++ runtime / Ollama daemon | **Pure Go (`CGO_ENABLED=0`)** |

---

## Key Highlights

- **Zero Downloads & On-The-Fly AI Creation**: You never download gigabytes of pre-trained weights from HuggingFace or lease external APIs. NeuroBranch forges a domain neural AI model directly from your CSV in under 2 seconds.
- **Zero Allocations on Hot Path (`0 B/op`)**: `PredictSlots` executes inference without triggering GC pressure, returning zero-heap stack results.
- **Semantic XOR & Word Order Disambiguation**: Format v3 embeds 32 positional vectors coupled with non-linear $GELU(E_i + P_i)$ pooling, mathematically distinguishing permutations like `"delivery refund"` from `"refund delivery"`.
- **3-Tier Decision Pipeline**: Classifies predictions into **Definite** (High confidence), **Ambiguous** (Borderline/narrow margin), or **Fallback** (Out-of-Distribution / High Shannon Entropy).
- **100% Deterministic Reproducibility (`Seed`)**: Fixed pseudorandom seed configuration combined with map-iteration-invariant tie-breaking guarantees bit-identical weights, reproducible decision boundaries, and stable CI/CD regression testing.
- **Multi-Intent Pipeline Support**: Automatically executes composite pipelines when secondary intent confidence meets multi-intent thresholds.
- **Lock-Free Atomic Hot-Swap & Telemetry**: Replace model weights on live traffic without locks (`sync/atomic.Pointer`), and stream drift queries into a bounded ring buffer for active learning.

---

## Benchmarks

Benchmarked on an AMD Ryzen 5 5600H (12 threads) running pure Go standard runtime (`go test -bench="." -benchmem`):

| Benchmark Target | Ops / Sec | Latency | Memory / Op | Allocations |
| :--- | :--- | :--- | :--- | :--- |
| **`BenchmarkPredictSlots`** | **33,433 ops/sec** | **29.91 μs** | **0 B/op** | **0 allocs/op** |
| **`BenchmarkPredictTokens`** | **33,126 ops/sec** | **30.18 μs** | **0 B/op** | **0 allocs/op** |
| **`BenchmarkForward`** | **33,091 ops/sec** | **30.21 μs** | **24 B/op** | **1 allocs/op** |
| **`BenchmarkGELU`** | **494,071 ops/sec** | **2.02 μs** | **0 B/op** | **0 allocs/op** |

---

## Installation

```bash
go get github.com/gluedays-cyber/neurobranch
```

---

## 3-Step Lifecycle

### Step 1: AI Design — Prepare Your Domain Knowledge (`train.csv`)
Create a clean two-column CSV containing natural user queries and corresponding target labels:

```csv
text,label
I want to cancel my payment and request a refund,Refund
Where is my package and delivery tracking,Delivery
Forgot my account password please reset,Account
sent the return box a week ago when do i get my money back,Refund
yo i typed the wrong apt number please update address,Delivery
locked out of my account after 3 tries help pls,Account
```

### Step 2: Build Your Own AI — Programmatic In-Memory Compilation & Dynamic Updates
Train your domain vocabulary and neural weights directly from Go code without external CLI tools, either from files or entirely in-memory:

```go
package main

import (
	"log"

	"github.com/gluedays-cyber/neurobranch"
)

func main() {
	// Option A: 100% In-Memory One-Shot Training (Zero File I/O)
	ai, err := neurobranch.TrainAIFromMap(map[string][]string{
		"Refund":   {"cancel my payment and request refund", "want my money back"},
		"Delivery": {"where is my package tracking", "update shipping address"},
		"Account":  {"reset my account password", "locked out of my profile"},
	})
	if err != nil {
		log.Fatalf("In-memory training failed: %v", err)
	}

	// Dynamic Runtime Update: Append new intents without restarting service (Atomic Hot-Swap)
	_ = ai.AppendDataMap(map[string][]string{
		"Billing": {"change credit card on file", "download monthly tax invoice"},
	})

	log.Println("AI engine compiled and dynamically updated in RAM.")
}
```
*(Note: File-based persistence is equally supported via `neurobranch.SaveBinaryModel` and `neurobranch.OpenAI`)*.

### Step 3: Zero Learning Curve Branching — Native Go Control Flow (`ai.Select` & `ai.If`)
Initialize the AI engine using `TrainAIFromMap` or `OpenOrTrain`, and write intelligent branching using Go's **native `switch`, `if`, and `comma-ok` statements** without learning complex new DSLs:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/gluedays-cyber/neurobranch"
)

func main() {
	modelPath := "weights/model.bin"
	dataPath := "data/train.csv"

	// 1. Zero-boilerplate library API: Train or load compiled weights
	ai, err := neurobranch.OpenOrTrain(dataPath, modelPath, 0.60)
	if err != nil {
		log.Fatalf("NeuroBranch engine initialization failed: %v", err)
	}

	query := "sent the return box a week ago when do i get my money back"

	// 2. Go Native switch-case: OOD and low-confidence automatically fall through to default:
	switch ai.Select(query) {
	case "Refund":
		fmt.Println("[ACTION] Processing refund request")
	case "Delivery":
		fmt.Println("[ACTION] Querying shipment tracking")
	case "Account":
		fmt.Println("[ACTION] Initiating account password reset")
	default:
		// Safe isolation: noise, unlearned patterns, and out-of-domain queries
		fmt.Println("[FALLBACK] Transfer to human agent or clarify question")
	}

	// 3. Go Native if guard clauses: ai.If (Conditional) or ai.Is (Natural English)
	if ai.If(query, "Refund") {
		fmt.Println("[GUARD] Confirmed refund inquiry via ai.If")
	}
	if ai.Is(query, "Refund", 0.85) {
		fmt.Println("[GUARD] Confirmed refund inquiry with ≥ 85% confidence via ai.Is")
	}

	// 4. Go Native comma-ok idiom
	if intent, confident := ai.Match(query); confident {
		fmt.Printf("[COMMA-OK] Decisive intent identified: %s\n", intent)
	} else if intent != "" {
		fmt.Printf("[AMBIGUOUS] Borderline intent (%s): confirmation recommended\n", intent)
	}

	// 5. Declarative Fluent DSL with Interactive Confirmation (HITL)
	ctx := context.Background()
	_ = ai.Switch(query).
		Case("Refund").
			AtLeast(0.85).
			Auto(func(ctx context.Context) error {
				fmt.Println("[DSL] Auto-executing refund pipeline")
				return nil
			}).
			Confirm("Would you like to process a refund?", func(ctx context.Context, prompt string) error {
				fmt.Printf("[INTERRUPT] Prompting user: %s\n", prompt)
				return nil
			}).
		Default(func(ctx context.Context) error {
			fmt.Println("[DSL] Fallback triggered")
			return nil
		}).
		Evaluate(ctx)
}
```

### Advanced Production Pattern: Multi-Intent Pipeline, 3-Tier Policy & Hot-Reload
For mission-critical production services requiring 3-tier calibration, composite intent pipelines, and concurrent lock-free model hot-swapping:

```go
// 1. Configure 3-Tier Policy and Active Learning Telemetry Buffer
router.SetPolicy(neurobranch.DispatchPolicy{
	HighThreshold:     0.75,
	LowThreshold:      0.40,
	MarginCutoff:      0.15,
	MaxEntropy:        2.0,
	PipelineThreshold: 0.30,
}).EnableTelemetry(1024)

// 2. Bind composite multi-intent and ambiguous handlers
router.
	BindPipeline("Refund", "Delivery", func(ctx context.Context, p, s string, payload any) error {
		fmt.Printf("[PIPELINE: %s -> %s] Processing combined return & shipment: %v\n", p, s, payload)
		return nil
	}).
	Ambiguous(func(ctx context.Context, p, s string, payload any) error {
		fmt.Printf("[AMBIGUOUS: %s vs %s] Requesting user confirmation: %v\n", p, s, payload)
		return nil
	})

// 3. Execute multi-intent pipeline dispatch
_ = router.DispatchPipeline(ctx, "can u cancel order #49281 and update delivery?", "OrderPayload")

// 4. Lock-free atomic hot-reload on live traffic (0 ns stop-the-world)
_ = router.Reload("weights/intent.bin")

// 5. Drain active learning drift events
events := router.DrainTelemetry()
fmt.Printf("Harvested %d drift events for active learning retraining.\n", len(events))
```

---

## 2-Layer Fail-Safe Guardrails & Error-Returning Routing (`RouteQuery`)

In mission-critical backends, forced misclassification of unlearned words, repetitive flood attacks, or out-of-distribution (OOD) noise causes catastrophic routing accidents (e.g. triggering an unauthorized refund on garbage or repetitive input). NeuroBranch enforces a hardened multi-tier defense:

```
[ Incoming Query ]
        │
        ▼
[ Layer 1: Tokenizer OOV & Repetitive Flood Guard ] ──(SingleChar ≥ 0.85)──> ErrUnlearnedVocabulary (< 1 μs)
        │                                           ──(UniqueRatio < 0.25)──> ErrDegeneratedInput    (< 1 μs)
        │ (Passed)
        ▼
[ Layer 2 Guard: Strict Short-Circuit OOD Interception ]
  1. Geometric L2 Cosine Out-of-Domain Guard ───────(Cosine < MinCosine)─> ErrOutOfDomain / Fallback
  2. Neural Free Energy (LogSumExp) ────────────────(Energy < MinEnergy)─> ErrOutOfDomain / Fallback
  3. Shannon Distribution Entropy ──────────────────(Entropy > MaxEntropy)─> ErrHighEntropy / Fallback
  4. Excessive Unknown Tokens (≥ 0.50) ─────────────(UnkRatio ≥ 0.50)───> ErrOutOfDomain / Fallback
        │ (In-Distribution Confirmed)
        ▼
[ Layer 2 Ambiguity & Multi-Intent Decision ]
  - Top-1 Probability Margin (< MarginCutoff) ──────> ErrAmbiguousIntent / Ambiguous Handler
  - Raw Logit Margin (< RawLogitMargin) ────────────> ErrAmbiguousIntent / Ambiguous Handler
  - Multi-Intent Co-Activation (CoActive ≥ 2) ──────> 1.5x Margin Tightening or Multi-Intent Pipeline
  - Low Confidence (< LowThreshold) ────────────────> ErrLowConfidence / Fallback
        │ (Confident Single Intent)
        ▼
[ Confident Safe Execution ] ───────────────────────────────────────────────> RouteDecision (~29 μs, 0 B/op)
```

### Layer 1: Tokenizer-Level Unlearned Vocabulary & Flood Cutoff (< 1 μs)
Before invoking any neural forward arithmetic, Layer 1 executes two zero-allocation pre-checks:
1. **Unlearned Fragment Cutoff (`SingleCharRatio >= 0.85`)**: If unlearned slang or gibberish causes BPE to fragment 85%+ of tokens into single-byte glyphs, the engine halts immediately with `ErrUnlearnedVocabulary`.
2. **Repetitive Token Flood Guard (`UniqueTokenRatio < 0.25`)**: Repeatedly spamming a valid in-domain word (e.g. *"refund refund refund refund..."*) attempts to bypass OOD centroid guards. Layer 1 detects degenerated inputs with low unique token ratios and isolates them immediately with `ErrDegeneratedInput`.

### Layer 2: Strict Short-Circuit Evaluation Order (OOD Before Ambiguity) (~29 μs)
Standard Softmax enforces $\sum P_i = 1.0$, which causes neural networks to output artificially inflated confidence even on meaningless inputs. To eliminate edge cases where novel unlearned inputs are erroneously flagged as "Ambiguous" between two arbitrary classes, NeuroBranch enforces a **strict short-circuit evaluation order**:
1. **Geometric L2 Cosine OOD Guard**: If the input's normalized embedding vector falls outside the domain manifold radius ($\cos(\theta) < \text{MinCosine}$), the request is immediately rejected as OOD.
2. **Free Energy ($-\text{LogSumExp}$)**: Measures the absolute activation strength of unnormalized logits before Softmax. Unlearned inputs lack activation energy and are cleanly isolated before ambiguity checks.
3. **Shannon Entropy**: Measures probability distribution chaos. High entropy ($> 2.0$) triggers OOD isolation.
4. **Ambiguity & Logit Margin (Only for Confirmed In-Distribution Inputs)**: Only once in-domain membership is mathematically established, the engine evaluates probability margin ($< 0.15$), raw logit gap ($< \text{RawLogitMargin}$), and co-activation density (`CoActiveCount >= 2`).

### Format v3: Zero-Configuration Self-Calibrating Metadata
Under Format Version 3 (`0x0003`), `CalibrateDomainDistribution` automatically embeds calibrated domain boundaries (`CalibratedMinEnergy`, `CalibratedMargin`, `CalibratedMinCosine`) directly into the model binary header. Any microservice loading the `.bin` file instantly inherits production-calibrated thresholds with **zero manual configuration code**.

### Deserialization Guard: Finite Tensor Verification (`math.IsNaN`, `math.IsInf`)
Beyond SHA-256 integrity checksums, `DeserializeModel` actively validates all loaded tensor weights (`Embedding`, `Positional`, `W1`, `B1`, `W2`, `B2`). Any non-finite float value resulting from divergence during training is immediately rejected with `ErrCorruptedTensor`, preventing poisoned runtime states.

### Production Error-Returning Interface (`RouteQuery`)

While `.Dispatch()` routes to registered callback handlers, `.RouteQuery()` returns standard Go sentinel errors for idiomatic `errors.Is` pattern matching:

```go
package main

import (
	"context"
	"errors"
	"fmt"
	"log"

	"github.com/gluedays-cyber/neurobranch"
)

func main() {
	// 1. Initialize router and load lightweight domain weights
	router, err := neurobranch.Open("weights/model.bin", 0.75)
	if err != nil {
		log.Fatalf("Failed to initialize router: %v", err)
	}

	ctx := context.Background()
	userInput := "bruh can u refund order #49281"

	// 2. Evaluate query with synchronous 2-layer Fail-Safe guards (~29 μs, 0 B/op)
	decision, err := router.RouteQuery(ctx, userInput)
	if err != nil {
		switch {
		case errors.Is(err, neurobranch.ErrDegeneratedInput):
			// [Layer 1 Guard (< 1 μs)]: Repeated token flood or degenerated pattern
			fmt.Printf("[L1 FLOOD REJECT] Degenerated input pattern (Unique token ratio: %.1f%%).\n",
				decision.UniqueTokenRatio*100)
			return

		case errors.Is(err, neurobranch.ErrUnlearnedVocabulary):
			// [Layer 1 Guard (< 1 μs)]: Completely unlearned words or noisy characters
			// Rejects before neural forward pass, eliminating cloud LLM invocation cost
			fmt.Printf("[L1 REJECT] Unlearned words (single-char ratio: %.1f%%). Prompting user to rephrase.\n",
				decision.SingleCharRatio*100)
			return

		case errors.Is(err, neurobranch.ErrOutOfDomain), errors.Is(err, neurobranch.ErrLowConfidence):
			// [Layer 2 Guard (~29 μs)]: Low activation energy or high Shannon entropy
			// Safely shunts query to Cloud LLM (Gemini) or human support queue
			fmt.Printf("[L2 ESCALATE] Out-of-Domain detected (Energy: %.2f, Entropy: %.2f). Shunting to Cloud LLM.\n",
				decision.Energy, decision.Entropy)
			return

		case errors.Is(err, neurobranch.ErrAmbiguousIntent):
			// [Layer 2 Guard (~29 μs)]: Competitive margin between top-2 intent candidates
			// Prompts user to clarify between Intent and SecondaryIntent
			fmt.Printf("[L2 AMBIGUOUS] Ambiguous intent (%s vs %s, margin: %.2f). Requesting clarification.\n",
				decision.Intent, decision.SecondaryIntent, decision.Margin)
			return

		default:
			log.Fatalf("Internal routing failure: %v", err)
		}
	}

	// 3. Execute domain business logic safely with zero allocation
	fmt.Printf("[DISPATCH 30μs] Confident Route: %s (Confidence: %.2f%%)\n",
		decision.Intent, decision.Confidence*100)
}
```

---

## Observability & Whitebox Debugging

Need to understand why a query routed to a specific branch or why it fell back? Use `Inspect`:

```go
trace := router.Inspect("can u cancel order #49281? i bought it by mistake")
```

```json
{
  "input_text": "can u cancel order #49281? i bought it by mistake",
  "token_ids": [4, 5, 8, 12, 45, 98],
  "subwords": ["can", "u", "cancel", "order", "#", "mistake"],
  "single_char_ratio": 0.16,
  "unknown_token_ratio": 0.0,
  "class_probabilities": {
    "Account": 0.0012,
    "Delivery": 0.0035,
    "Refund": 0.9953
  },
  "predicted_label": "Refund",
  "secondary_label": "Delivery",
  "confidence": 0.9953,
  "margin": 0.9918,
  "entropy": 0.0351,
  "energy": 6.842,
  "threshold": 0.75,
  "is_ambiguous": false,
  "is_pipeline": false,
  "is_fallback": false,
  "latency_micros": 30
}
```

---

## NeuroBranch: 3-Head Geometric Intelligent Filtering Engine

`NeuroBranch` is an in-memory intelligent filtering gate that wraps a single shared neural backbone encoder with three orthogonal geometric and symbolic guard heads. It solves model overconfidence, Out-of-Domain (OOD) leakage, and slang ambiguity without spawning multiple fragmented networks or incurring heap allocations.

```
Incoming Request ("it is too dark in here please switch on lamps")
                               │
                               ▼
  ┌────────────────────────────────────────────────────────┐
  │ Shared Neural Backbone (In-Memory BPE + Positional MLP) │ ──> z ∈ ℝ⁶⁴ (Unit Norm)
  └────────────────────────────┬───────────────────────────┘
                               │
 ┌─────────────────────────────┴─────────────────────────────┐
 │ Stack-Allocated 3-Head Geometric Gate (~5 to ~28 μs)      │
 │                                                           │
 │  [Head 1]: L2 Cosine Out-of-Domain (OOD) Guard            │
 │            DotProduct(z, C_domain) < MinCosine ?          │
 │            --> Immediate Fallback Isolation if OOD        │
 │                                                           │
 │  [Head 2]: 1-Cycle Bitwise Symbolic Anchor Soft-Bias      │
 │            if (Bitmask & Anchor_i) != 0                   │
 │            --> Logit_i += Weight * PopCount(Mask)         │
 │                                                           │
 │  [Head 3]: Calibrated Top-2 Margin & Shannon Entropy      │
 │            Stack Softmax over Adjusted Logits             │
 │            --> Definite / Pipeline / Ambiguous / Fallback │
 └─────────────────────────────┬─────────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
 [Definite Action]     [Multi-Intent Pipeline] [Safe Fallback]
```

### Key Engineering Capabilities

1. **Strict Zero-Allocation Hot-Path (`0 B/op`, `0 allocs/op`)**:
   Internal inference operates on fixed stack buffers (`[16]float32` and `[64]float32`). Pre-tokenized inputs dispatched via `FilterTokens` execute in **~28 μs with strictly 0 B/op heap allocation**.
2. **Single Shared Backbone (No Error Cascading)**:
   Avoids training multiple fragmented networks. A single compact encoder extracts context, while downstream safety boundaries and semantic boosts are computed geometrically.
3. **Neuro-Symbolic Anchor Soft-Bias with Max Clamping (`SetMaxAnchorBoost`)**:
   Replaces fragile `strings.Contains` hardcoded branching with additive logit bonuses. Subword token IDs map to 64-bit masks (`uint64`), executing anchor boosts in a single CPU cycle (`&` and `popcount`). Cumulative boosts are capped per class (`DefaultMaxAnchorBoost = 3.0`), preventing multi-keyword payloads from overwhelming the neural boundary.
4. **Adaptive L2 Cosine OOD Boundary (`CalibrateDomainDistribution`)**:
   Beyond static radius thresholds, `CalibrateDomainDistribution` analyzes embedding manifold variance across training samples, computing an adaptive threshold $\text{MinCosine} = \mu - k\cdot\sigma$ to cleanly reject out-of-domain queries while maintaining 0 B/op runtime performance.
5. **Multi-Intent Co-Activation Density Guard & Dynamic Margin**:
   When inputs contain conflicting keywords from multiple classes (e.g., `"light cooling door soundbar"`), Free Energy and Softmax can suffer from distortion. NeuroBranch tracks pre-Softmax co-activation density (`CoActiveCount >= 2`) and automatically scales required logit margins by 1.5x. Furthermore, `CalibrateDomainDistribution` automatically measures inter-class centroid manifold distances, determining optimal `RawLogitMargin` without magic numbers.

### NeuroBranch Production Example

```go
package main

import (
	"context"
	"fmt"
	"log"

	"neurobranch/pkg/neurobranch"
)

func main() {
	// 1. Initialize NeuroBranch from binary model
	gate, err := neurobranch.NewNeuroBranch("weights/demo_iot.bin")
	if err != nil {
		log.Fatalf("NeuroBranch init failed: %v", err)
	}

	// 2. Calibrate adaptive domain boundary with 2-sigma variance threshold
	if samples, err := neurobranch.LoadCSVDataset("data/demo_iot.csv"); err == nil {
		gate.CalibrateDomainDistribution(samples, 2.0)
	}

	// 3. Configure max anchor boost cap (prevents logit explosion)
	gate.SetMaxAnchorBoost(3.0)

	// 4. Bind route handlers with symbolic anchor soft-biases
	gate.Bind("LightControl", func(ctx context.Context, payload any) error {
		fmt.Println(">>> [GPIO 18 HIGH] Toggle Living Room Chandelier")
		return nil
	}).WithAnchor(1.8, "dark", "light", "lamps", "lamp", "switch", "lights")

	gate.Bind("ClimateControl", func(ctx context.Context, payload any) error {
		fmt.Println(">>> [MODBUS UART] Set Daikin HVAC Inverter Temperature")
		return nil
	}).WithAnchor(1.8, "cooling", "heat", "fan", "temp", "ac", "air")

	gate.Fallback(func(ctx context.Context, payload any) error {
		fmt.Println(">>> [FALLBACK] Isolated Out-of-Domain or Ambiguous Request")
		return nil
	})

	// 4. Dispatch with microsecond latency and zero allocations
	ctx := context.Background()
	_ = gate.Filter(ctx, "it is too dark in here please switch on lamps", nil)
}
```

---

## The Evolution of Control Flow: Why Retro Branching Fails & How NeuroBranch Proves Its Architectural Superiority

Traditional programming languages force engineers into **discrete control flow** (`if`, `switch`, `hash map`, `regex`). These constructs were invented in the 1960s for deterministic, byte-exact hardware primitives. When applied to real-world strings, natural language, unstructured logs, or conversational commands, **they collapse entirely**.

NeuroBranch transforms control flow from brittle discrete matching into **continuous geometric vector-space routing ($text \to action$) in ~30 μs**.

The included multi-task demonstration driver (`cmd/ib-demo`) directly pits NeuroBranch against traditional programming primitives across 6 critical enterprise domains:

### 1. Structural Comparison: Retro Branching vs. NeuroBranch

| Control Flow Primitive | Why It Breaks Down on Real-World Input | How NeuroBranch Resolves It Permanently |
| :--- | :--- | :--- |
| **`switch` / `if (str == val)`** | **100% Failure on Variations**: A 1-character typo (`"refnd"`), colloquial phrasing (`"gimme my cash back"`), or extra whitespace causes silent fall-through. | **BPE Continuous Embedding**: Maps all semantic synonyms and misspelled subwords to contiguous vector coordinates in 64-D space. |
| **Hash Maps (`map[string]T`)** | **Exact-Key Blindness**: Cannot index semantic equivalence. Caching 10,000 phrasing variations requires 10,000 distinct hash keys, leading to memory bloat and constant cache misses. | **Semantic Coordinate Resolution**: Resolves infinite sentence variations into deterministic Go handlers in ~30 μs with zero external network overhead. |
| **Regular Expressions (`regex`)** | **Combinatorial Explosion & ReDoS**: Supporting synonyms requires nested lookaheads and permutations ($O(N!)$ rules), causing CPU exhaustion (ReDoS backtracking) and unmaintainable regex hell. | **Non-Linear GELU Tensor Layers**: Evaluates feature cross-products without regex backtracking, maintaining deterministic, capped compute times. |
| **String Search (`strings.Contains`)** | **Semantic XOR Failure**: Commutative addition collapses opposite meanings. Cannot distinguish `"refund my delivery"` from `"delivery instead of refund"`. Word collisions cause catastrophic misrouting. | **Learned Positional Embeddings**: Encodes token sequence coordinates ($P_{32 \times 64}$) into non-linear activations, mathematically distinguishing token order permutations. |
| **Binary Boolean Decisions** | **Forced Misclassification**: Discrete `if/else` forces ambiguous or out-of-distribution noise into whichever branch happens to have a loose wildcard match. | **3-Tier Calibrated Pipeline**: Quantifies Shannon Entropy to isolate OOD noise to Fallback, while detecting competitive top-2 margins to trigger Step-up 2FA/Ambiguous logic. |

---

### 2. 6 Enterprise Domains: Proven Architectural Superiority

The companion demonstration project ([NeuroBranch - demo](../NeuroBranch%20-%20demo)) proves these architectural advantages live across 6 isolated production models:

#### Domain 1: E-Commerce CS Gateway (Defeating the Semantic XOR Dilemma)
- **The Retro Collapse**: `if strings.Contains(msg, "refund") && strings.Contains(msg, "delivery")` collapses opposite business intents. Both `"refund delivery fee"` and `"delivery instead of refund"` trigger the same branch. Regex permutations explode exponentially.
- **The NeuroBranch Victory**: Learned positional vectors ($P_i$) coupled with non-linear $GELU(E_i + P_i)$ pooling mathematically separate token permutations. Furthermore, `DispatchPipeline` automatically executes composite operations (e.g. Return Approved $\to$ Reshipment Initiated) when both primary and secondary confidences qualify.

#### Domain 2: Semantic LLM Gateway (Defeating Hash Map Key Misses & API Waste)
- **The Retro Collapse**: Caching natural language with `map[string]Handler` achieves a near 0% hit rate because users never type the exact same string twice. Consequently, backends route 100% of routine traffic to OpenAI/Claude, burning $0.02–$0.05 and 1,500ms per request.
- **The NeuroBranch Victory**: Maps routine banking commands (`QueryBalance`, `TransferFunds`, `CardLock`) directly to in-memory Go handlers in **30 μs at $0.00 cost**. Out-of-Distribution (OOD) queries (e.g. `"explain quantum physics"`) are detected via high Shannon Entropy ($> 1.80$) and safely escalated to cloud LLMs.

#### Domain 3: High-Throughput SRE Log Triage (Defeating ReDoS & GC Pauses with 0 B/op)
- **The Retro Collapse**: Ingesting 100,000+ log lines/sec through complex regex engines burns 100% CPU due to catastrophic backtracking. String allocations trigger GC stop-the-world pauses, choking message brokers (Kafka, Vector).
- **The NeuroBranch Victory**: Executes stack-allocated zero-heap inference (`PredictSlots`) with **strictly 0 B/op and 0 allocs/op**. Instantly routes critical P0 panics (OOMKilled) to autoscalers while shunting low-priority health probes without heap garbage.

#### Domain 4: Offline Edge IoT Control (Defeating Brittle Keyword Matching in <180KB RAM)
- **The Retro Collapse**: Hard-coded `switch(cmd)` fails when users speak naturally: `"it's freezing in here"` fails to trigger `"turn on heater"`. Running local 7B models requires 4GB+ RAM, impossible on 64MB embedded Linux boards.
- **The NeuroBranch Victory**: Compiles into a single Little-Endian binary under 180 KB with zero external dependencies and zero CGO. Maps colloquial voice/text variants directly to hardware GPIO/UART actuators in single-digit microseconds.

#### Domain 5: Automated CI/CD Failure Triage (Defeating Fragile String Scrapers)
- **The Retro Collapse**: Compiler error messages change formatting across toolchains (Docker, Go, Gradle, Kubernetes). Hard-coded string pattern matching silently breaks, forcing DevOps engineers to manually triage build failures.
- **The NeuroBranch Victory**: Ingests unstructured build error tails and generalizes statistical subwords to trigger deterministic self-healing actions: `AutoRetry` (transient network 504), `ScaleUp` (OOM kill status 137), or `NotifyAuthor` (code syntax error).

#### Domain 6: FinTech Transaction Memo Audit (Defeating Naive Blacklists with 3-Tier Safety)
- **The Retro Collapse**: Keyword blacklists (`strings.Contains("scam")`) are trivially bypassed by fraudsters using typo obfuscation (`"p0lice f1ne"`). Rigid binary `if/else` either blocks legitimate transactions or lets fraud slip through.
- **The NeuroBranch Victory**: Evaluates semantic risk. When the margin between normal transfer and scam suspicion is borderline (`isAmbiguous`), it intercepts execution to trigger Step-Up 2FA (SMS OTP challenge), providing a dynamic middle-ground impossible in standard boolean control flow.

---

### 3. Companion Demonstration Project

To inspect and execute live demonstrations of the 6 enterprise domains, clone or navigate to the companion **[NeuroBranch - demo](../NeuroBranch%20-%20demo)** repository:

```bash
# Navigate to companion demonstration project
cd "../NeuroBranch - demo"

# Run all 6 domains (46 scenarios) in automated showcase mode
go run ./cmd/ib-demo -domain all

# Or inspect individual enterprise domains
go run ./cmd/ib-demo -domain cs       # 1. E-Commerce CS Gateway (XOR & Multi-Intent)
go run ./cmd/ib-demo -domain llm      # 2. Semantic LLM Gateway & Cloud Bypass ($0.00 vs $0.02)
go run ./cmd/ib-demo -domain sre      # 3. High-Throughput SRE Log Triage (0 B/op)
go run ./cmd/ib-demo -domain iot      # 4. Offline Edge IoT Command Dispatcher
go run ./cmd/ib-demo -domain cicd     # 5. Automated CI/CD Failure Triage & Self-Healing
go run ./cmd/ib-demo -domain fintech  # 6. FinTech Transaction Memo Audit & Fraud Prevention
```

---

## Core Routing API Reference

| Method / Struct | Signature | Operational Role |
| :--- | :--- | :--- |
| **`NewRouter`** | `NewRouter(path string, threshold float64) (*Router, error)` | Loads v3 binary weights, initializes atomic model pointer, and builds 3-tier router. |
| **`SetPolicy`** | `.SetPolicy(policy DispatchPolicy) *Router` | Configures high/low thresholds, margin cutoffs, raw logit margins, OOD entropy, and fail-safe rules. |
| **`SetTemperature`** | `.SetTemperature(t float32) *Router` / `*NeuroBranch` | Configures temperature scaling factor for Softmax probability smoothing ($T > 1.0$) or sharpening ($T < 1.0$). |
| **`Temperature`** | `.Temperature() float32` | Retrieves current temperature scaling factor (default: `1.0`). |
| **`SetSingleCharRatioCutoff`** | `.SetSingleCharRatioCutoff(cutoff float64) *Router` / `*NeuroBranch` | Configures Layer 1 threshold for unlearned single-character token ratio (default: `0.85`). |
| **`ScanUnlearnedPatterns`** | `ScanUnlearnedPatterns(text string) bool` | Pre-inference zero-alloc guard (< 1 μs) intercepting random hex/base64 strings. |
| **`Bind`** | `.Bind(label string, handler RouteAction) *Router` | Associates a trained class with a Go handler: `func(ctx context.Context, payload any) error`. |
| **`BindPipeline`** | `.BindPipeline(p, s string, handler PipelineAction) *Router` | Registers composite handler triggered when primary and secondary intents are both eligible. |
| **`Ambiguous`** | `.Ambiguous(handler AmbiguousAction) *Router` | Intercepts borderline confidence, probability margin, or raw logit margin anomalies. |
| **`Fallback`** | `.Fallback(handler RouteAction) *Router` | Designates safety handler for low confidence, high unknown token ratio, or OOD entropy. |
| **`RouteQuery`** | `.RouteQuery(ctx context.Context, text string) (RouteDecision, error)` | Evaluates 2-layer fail-safe guards and returns standard Go sentinel errors. |
| **`Dispatch`** | `.Dispatch(ctx context.Context, text string, payload any) error` | Evaluates 3-tier routing with 2-layer guards and executes bound branch in ~30 μs. |
| **`DispatchPipeline`**| `.DispatchPipeline(ctx context.Context, text string, payload any) error` | Executes multi-intent pipeline handlers if eligible, falling back to 3-tier routing. |
| **`Reload`** | `.Reload(path string) error` | Atomically swaps weights on live traffic without locks (`0 ns` stop-the-world). |
| **`EnableTelemetry`**| `.EnableTelemetry(capacity int) *Router` | Allocates thread-safe ring buffer capturing ambiguous, OOD, and pipeline requests. |
| **`DrainTelemetry`** | `.DrainTelemetry() []TelemetryEvent` | Extracts collected drift events in FIFO order for active learning retraining. |
| **`PredictSlots`** | `model.PredictSlots(tokens []uint32, temperature float32) (StaticInferenceResult, error)` | Stack-allocated inference primitive achieving strictly **`0 B/op, 0 allocs/op`**. |

### Temperature Scaling

NeuroBranch applies temperature scaling directly within its stack-allocated Softmax computation:

$$\sigma(z)_i = \frac{e^{z_i / T}}{\sum_{j=1}^K e^{z_j / T}}$$

- **Default ($T = 1.0$)**: Standard unscaled Softmax distribution.
- **Higher Temperature ($T > 1.0$, e.g. $1.5 \sim 2.0$)**: Softens logit disparities, reducing neural overconfidence and elevating entropy for borderline/unclear queries. Ideal when strict safety boundaries and ambiguity prompts are preferred.
- **Lower Temperature ($0 < T < 1.0$, e.g. $0.5 \sim 0.7$)**: Sharpens probability distribution towards the dominant class, reducing ambiguity on well-defined distinct intents.

```go
// Adjust temperature dynamically on live Router or NeuroBranch instances
router.SetTemperature(1.5)
fmt.Printf("Current inference temperature: %.2f\n", router.Temperature())
```

---

## Project Structure

```text
NeuroBranch - lib/
├── docs/
│   └── MANUAL.md          # Comprehensive manual, keyword guide & tutorial
├── pkg/
│   └── neurobranch/         # Pure-Go zero-dependency modular core engine
│       ├── anchor.go      # Symbolic anchors, asymmetric inhibition & fluent builder
│       ├── binary.go      # Format v1, v2 & v3 (Self-Calibrating) Little-Endian serializer
│       ├── calibration.go # Domain centroid, dispersion & adaptive margin calibration
│       ├── evaluator.go   # Zero-allocation 3-head gate evaluator & 3-tier filter execution
│       ├── facade.go      # High-level zero-boilerplate APIs (Open, OpenOrTrain, Train)
│       ├── neurobranch.go   # 3-Head geometric filter core, configuration & diagnostics
│       ├── ops.go         # SafeClamp, GELU, Softmax, MatMul, and Non-Linear Pooling
│       ├── policy.go      # 3-Tier routing policies, thresholds & handler type signatures
│       ├── router.go      # Router core lifecycle, atomic reload & handler registration
│       ├── router_dispatch.go # Fail-safe query routing & multi-intent pipeline dispatch
│       ├── runtime.go     # Zero-alloc PredictSlots, Shannon Entropy & in-memory model
│       ├── telemetry.go   # Thread-safe ring buffer for active learning feedback
│       ├── tokenizer.go   # Pure Go BPE subword tokenizer
│       ├── trace.go       # RouteDecision, RouteTrace & BranchTrace telemetry structures
│       └── trainer.go     # AdamW backprop trainer with positional embedding learning
├── neurobranch.go           # Root library export facade (import "neurobranch")
├── go.mod                 # Go module definition (pure library)
├── LICENSE                # MIT License
└── README.md              # Project documentation
```

*(Note: Executable binaries, demonstration CLI tools, and datasets are hosted in the companion repository [NeuroBranch - demo](../NeuroBranch%20-%20demo)).*

---

## Epilogue: An Architectural Manifesto on the Evolution of Control Flow

> *"In the beginning, there was `JMP`. Then came `if`. And for fifty years, computer science fell asleep."*  
> — Thoughts on Software Evolution from **gluedays@gmail.com**

In 1945, the Von Neumann architecture laid the physical foundation of modern computing with a crude primitive: the conditional jump (`JMP` / `goto`). The machine simply altered its instruction pointer based on zero-flags in silicon registers.

In 1968, Edsger W. Dijkstra published his seminal paper, *"Go To Statement Considered Harmful"*. That intellectual revolution forced programming languages to evolve: unruly jumps were disciplined into structured, deterministic control flow—giving birth to the ubiquitous `if`, `else`, and `switch`. For a world governed by punch cards, clean integers, and rigid ASCII strings, discrete equality matching was a masterpiece.

**However, that was half a century ago.**

Today, the digital landscape has undergone an irreversible phase transition. Humanity no longer feeds software with pristine 4-byte integers and sanitized alphanumeric enums. Modern systems are inundated with an ocean of **polymorphic, unstructured, noisy, colloquial, and contextual human reality**:
- Typo-ridden mobile messages, dialect slang, and conversational phrasing.
- Asynchronous high-throughput log streams with mutating compiler stack traces.
- Multi-dimensional contextual intents where word order inverts business logic (`"refund delivery"` vs `"delivery refund"`).

Yet, look at modern programming languages—whether Go, Rust, C++, Java, or Python. **Their fundamental control flow primitive has not evolved a single millimeter since the 1970s.**

Engineers are still desperately stringing together brittle `if` statements, bloating codebases with thousands of fragile regexes, and watching servers collapse under ReDoS backtracks and CPU saturation. When regex fails, the industry swings to the opposite extreme of absurdity: burning millions of dollars routing simple string branches to 400-billion-parameter cloud LLMs, waiting 2,000 milliseconds and paying $0.03 just to pick an execution branch.

**This is architectural stagnation. Retro conditional branching must evolve.**

Control flow must transcend discrete, byte-exact binary matching. It must evolve into **continuous geometric vector-space routing**:
1. Branching should not break because of a single misplaced character or slang synonym.
2. Control flow must natively understand semantic context, token permutation, and feature interactions in single-digit microseconds.
3. Decision boundaries must be probabilistic and multi-tiered—safely executing confident branches, gracefully prompting when ambiguous, and deterministically isolating out-of-distribution noise without panic.

**NeuroBranch is not just a tool; it is a working manifesto.** It proves that a self-contained, domain-trained neural routing engine running in pure Go can replace brittle retro branching at **~30 microseconds with strictly 0 B/op heap allocation**.

The future of programming languages lies in elevating the compiler and runtime to understand continuous semantic topology. The era of blind discrete branching is over.

---

## License

This project is licensed under the MIT License.

