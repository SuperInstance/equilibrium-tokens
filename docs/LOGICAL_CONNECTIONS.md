# Equilibrium Tokens: Logical Connections

## Overview

This document describes the logical connections between components in the Equilibrium Tokens architecture, showing how constraints compose to form the complete navigation system.

## Horizontal Composition (Constraint Conjunction)

Constraints compose **multiplicatively** because they are independent:

```
E = RateSpec ∧ ContextSpec ∧ InterruptionSpec ∧ SentimentSpec
  = Φ_rate × Φ_context × Φ_interrupt × Φ_sentiment
```

**Why multiplicative?** Because constraints are independent variables. A failure in rate doesn't invalidate context. The grammar reflects this through parallel composition.

### Mathematical Justification

Given independent random variables X₁, X₂, X₃, X₄ representing the satisfaction of each constraint:

```
P(all constraints satisfied) = P(X₁) × P(X₂) × P(X₃) × P(X₄)
```

This is fundamental probability theory.

## Vertical Composition (Constraint Dependency)

Some constraints **depend** on others. The grammar expresses this through sequential composition:

```
NavigationSpec = RateSpec → ContextSpec → SentimentSpec → InterruptionSpec
```

**Why sequential?** Because navigation requires temporal ordering: you must know the rate before you can navigate context.

### Dependency Graph

```
RateEquilibrium (independent)
    ↓
ContextEquilibrium (depends on Rate)
    ↓
SentimentEquilibrium (depends on Context)
    ↓
InterruptionEquilibrium (can interrupt at any point)
```

## Conditional Composition (Constraint Implication)

Interruptions **imply** reset:

```
InterruptionSpec(confidence > 0.7) → ResetSpec(attention_weight = 0.8)
```

**Why implication?** Because interruptions are **events** that trigger state changes in the navigation grammar.

### Implication Rules

| Condition | Action | New State |
|-----------|--------|-----------|
| `confidence > 0.7` | Reset attention | `attention_weight ← 0.8` |
| `confidence ≤ 0.7` | Log only | `attention_weight ← unchanged` |
| `sentiment_delta > 0.5` | Explore territory | `new_path ← random_walk()` |

## Component Interconnection Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   EQUILIBRIUM ORCHESTRATOR                   │
│                  (The Conductor / NavigationSpec)           │
└─────┬─────────────┬──────────────┬─────────────┬───────────┘
      │             │              │             │
      ▼             ▼              ▼             ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│   RATE   │  │  CONTEXT │  │SENTIMENT │  │INTERRUP- │
│  Spec    │  │   Spec   │  │   Spec   │  │  TION   │
│          │  │          │  │          │  │   Spec   │
│Temporal  │  │ Spatial  │  │ Affective│  │  Reset   │
│Adverbial │  │ Noun Phr │  │ Adjective│  │Conjunc-  │
│          │  │          │  │          │  │  tion    │
└─────┬────┘  └─────┬────┘  └─────┬────┘  └─────┬────┘
      │             │              │             │
      │             └──────────────┴─────────────┘
      │                      │
      ▼                      ▼
┌─────────────┐      ┌─────────────┐
│ TimerFd     │      │ RAG Index   │
│ (Hardware)  │      │ (Storage)   │
└─────────────┘      └─────────────┘
```

## Data Flow

```
1. Input Tokens
   │
   ├→ [RateSpec] ──→ Rate_Weight (0-1)
   │                 │
   │                 └→ Interval adjustment
   │
2. Context Embedding
   │
   ├→ [ContextSpec] ──→ Context_Weight (0-1)
   │                    │
   │                    ├→ Cosine similarity
   │                    └→ Basin interpolation
   │
3. Sentiment Scores (VAD)
   │
   ├→ [SentimentSpec] ──→ Sentiment_Weight (0-1)
   │                      │
   │                      ├→ Valence normalization
   │                      └→ Rolling average
   │
4. Interruption Event
   │
   ├→ [InterruptionSpec] ──→ Interruption_Weight (0-1)
   │                         │
   │                         ├→ Confidence threshold
   │                         └→ Attention reset
   │
5. [ORCHESTRATOR]
   │
   └→ Confidence = ∏ (all weights) ──→ Output Path
```

## Constraint Satisfaction Function

For each conversational turn, the orchestrator computes:

```
confidence_t = Φ_rate(t) × Φ_context(t) × Φ_sentiment(t) × Φ_interrupt(t)
```

Where:
- `Φ_rate(t) = 1 - |rate_in - rate_out| / max(rate_in, rate_out)`
- `Φ_context(t) = cosine_similarity(context_t, RAG_index)`
- `Φ_sentiment(t) = (valence_t + 1.0) / 2.0`
- `Φ_interrupt(t) = 0.8 if interrupted else 1.0`

## State Transitions

```
┌─────────┐     High Equilibrium     ┌──────────────┐
│  Start  │ ──────────────────────> │ Flow State   │
└─────────┘    (confidence > 0.7)   └──────────────┘
     │                                   │
     │ Interruption                     │ Rate drop
     ▼                                   ▼
┌─────────┐                        ┌──────────────┐
│  Reset  │ ─────────────────────> │Explore State │
└─────────┘    (confidence < 0.4)   └──────────────┘
     │                                   │
     │ Sentiment improve                 │ Context match
     ▼                                   ▼
     └──────────────> ┌──────────────┐
                      │ Flow State   │
                      └──────────────┘
```

## Interface Contracts

### RateEquilibrium → Orchestrator
```rust
pub fn on_rate_change(&self, new_rate: f64) -> Result<(), RateError>
pub fn calculate_rate_weight(&self, input_rate: f64) -> f64
```

**Contract**: Jitter < 2ms for all rate changes.

### ContextEquilibrium → Orchestrator
```rust
pub fn navigate(&mut self, context: Tensor) -> Result<NavigationResult, ContextError>
```

**Contract**: Cosine similarity > 0.7 for returned basins.

### InterruptionEquilibrium → Orchestrator
```rust
pub fn handle_interruption(&mut self, event: &InterruptionEvent) -> InterruptionResult
```

**Contract**: P(reset | confidence > 0.7) = 1.0

### SentimentEquilibrium → Orchestrator
```rust
pub fn update_sentiment(&mut self, vad: VADScores) -> Result<(), SentimentError>
pub fn conduct(&self, vad: &VADScores) -> ConductorResult
```

**Contract**: Sentiment weight ∈ [0, 1]

## Composition Examples

### Example 1: High Equilibrium (Fishing Boat)
```
Rate_Weight = 1.0      (perfect match: 2.0 Hz in, 2.0 Hz out)
Context_Weight = 0.9   (high similarity in calm basin)
Sentiment_Weight = 0.9 (positive valence: 0.8)
Interruption_Weight = 1.0 (no interruption)

Confidence = 1.0 × 0.9 × 0.9 × 1.0 = 0.81
```

### Example 2: Low Equilibrium (Jazz Band)
```
Rate_Weight = 0.25     (poor match: 0.5 Hz in, 2.0 Hz out)
Context_Weight = 0.7   (moderate similarity)
Sentiment_Weight = 0.65 (mixed valence: 0.3)
Interruption_Weight = 0.8 (interruption occurred)

Confidence = 0.25 × 0.7 × 0.65 × 0.8 = 0.091
```

### Example 3: Medium Equilibrium (Symphony Pops)
```
Rate_Weight = 0.75     (good match: 1.5 Hz in, 2.0 Hz out)
Context_Weight = 0.8   (good similarity)
Sentiment_Weight = 0.85 (positive valence: 0.7)
Interruption_Weight = 1.0 (no interruption)

Confidence = 0.75 × 0.8 × 0.85 × 1.0 = 0.51
```

## Error Propagation

```
┌─────────────┐
│ RateError   │ ──→ OrchestratorError::RateError
└─────────────┘

┌─────────────┐
│ ContextError│ ──→ OrchestratorError::ContextError
└─────────────┘

┌─────────────┐
│SentimentErr │ ──→ OrchestratorError::SentimentError
└─────────────┘

┌─────────────┐
│InterruptErr │ ──→ OrchestratorError::InterruptionError
└─────────────┘
```

All component errors are wrapped in `OrchestratorError` for unified error handling.

## Testing Strategy

### Unit Tests
- Each component tested independently
- All invariants verified
- Edge cases covered

### Integration Tests
- Component interactions tested
- Data flow validated
- Error propagation verified

### Property-Based Tests
- Invariants hold for all inputs
- Composition rules verified
- Mathematical properties proven

### Scenario Tests
- Fishing boat: High equilibrium
- Jazz band: Low equilibrium
- Symphony pops: Medium equilibrium
