package neurobranch

import (
	core "github.com/gluedays-cyber/neurobranch/pkg/neurobranch"
)

// Re-export core types for top-level root import
type (
	AI                  = core.AI
	Router              = core.Router
	DispatchPolicy      = core.DispatchPolicy
	RouteDecision       = core.RouteDecision
	RouteTrace          = core.RouteTrace
	BranchTrace         = core.BranchTrace
	GateTrace           = core.GateTrace
	BranchRouteBuilder  = core.BranchRouteBuilder
	GateRouteBuilder    = core.GateRouteBuilder
	TrainConfig         = core.TrainConfig
	DataSample          = core.DataSample
	InferenceModel      = core.InferenceModel
	NeuroBranch         = core.NeuroBranch
	TelemetryEvent      = core.TelemetryEvent
	TelemetryRingBuffer = core.TelemetryRingBuffer
	SwitchBuilder       = core.SwitchBuilder
	CaseClause          = core.CaseClause
	IntentHandler       = core.IntentHandler
	ConfirmationHandler = core.ConfirmationHandler
	Option              = core.Option
)

// Sentinel Errors
var (
	ErrModelNotInitialized  = core.ErrModelNotInitialized
	ErrClassIndexOutOfRange = core.ErrClassIndexOutOfRange
	ErrUnlearnedVocabulary  = core.ErrUnlearnedVocabulary
	ErrUnlearnedPattern     = core.ErrUnlearnedPattern
	ErrDegeneratedInput     = core.ErrDegeneratedInput
	ErrCorruptedTensor      = core.ErrCorruptedTensor
	ErrLowConfidence        = core.ErrLowConfidence
	ErrHighEntropy          = core.ErrHighEntropy
	ErrOutOfDomain          = core.ErrOutOfDomain
	ErrAmbiguousIntent      = core.ErrAmbiguousIntent
	ErrUnhandledIntent      = core.ErrUnhandledIntent
	ErrConfirmationReq      = core.ErrConfirmationReq
)

// High-level functions exported at the package root
var (
	// High-Level Facade APIs (Zero-boilerplate Train & Route)
	Train              = core.Train
	TrainAI            = core.TrainAI
	TrainAIFromMap     = core.TrainAIFromMap
	TrainInMemory      = core.TrainInMemory
	TrainFromMap       = core.TrainFromMap
	TrainWithOptions   = core.TrainWithOptions
	TrainAIWithOptions = core.TrainAIWithOptions
	EnsureModel        = core.EnsureModel
	Open               = core.Open
	OpenAI             = core.OpenAI
	OpenWithOptions     = core.OpenWithOptions
	OpenAIWithOptions   = core.OpenAIWithOptions
	OpenOrTrain        = core.OpenOrTrain

	// Functional Options for Configuration Overrides
	WithPolicy              = core.WithPolicy
	WithEnergyThreshold     = core.WithEnergyThreshold
	WithConfidenceThreshold = core.WithConfidenceThreshold
	WithMarginCutoff        = core.WithMarginCutoff
	WithMaxEntropy          = core.WithMaxEntropy
	WithPipelineThreshold   = core.WithPipelineThreshold
	WithPatternGuard        = core.WithPatternGuard

	// Core Training & Dataset Helpers
	DefaultTrainConfig    = core.DefaultTrainConfig
	LoadCSVDataset        = core.LoadCSVDataset
	TrainModel            = core.TrainModel
	DefaultBaseVocab      = core.DefaultBaseVocab
	TrainBPEWithBaseVocab = core.TrainBPEWithBaseVocab

	// Binary Serialization
	SaveBinaryModel = core.SaveBinaryModel
	LoadBinaryModel = core.LoadBinaryModel

	// In-Memory Routing Engines
	NewAI                 = core.NewAI
	NewAIFromModel        = core.NewAIFromModel
	NewRouter             = core.NewRouter
	NewRouterFromModel    = core.NewRouterFromModel
	NewNeuroBranch        = core.NewNeuroBranch
	DefaultDispatchPolicy = core.DefaultDispatchPolicy

	// Math & Ops
	LogSumExp                 = core.LogSumExp
	CosineSimilarity          = core.CosineSimilarity
	CalculateUniqueTokenRatio = core.CalculateUniqueTokenRatio
	ScanUnlearnedPatterns     = core.ScanUnlearnedPatterns
)
