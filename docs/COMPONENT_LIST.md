# Equilibrium Tokens: Component List

## Overview

This document provides detailed specifications for each component in the Equilibrium Tokens architecture. Components are organized by their role in the constraint grammar.

## Component Classes

### 1. RateSpec (The Temporal Adverbial)

**Type**: Rust struct `RateEquilibrium`

**Location**: `src/constraint_grammar/rate.rs`

**Purpose**: Modifies the temporal structure of the token stream

**Grammatical Role**: Must precede all other constraints (time is primary)

**Terminal Values**:
- `target_rate: f64` - Target tokens/second
- `jitter_threshold: f64` - Maximum acceptable jitter in milliseconds

**Production Rules**:
- `on_rate_change(new_rate) → adjust_interval()`
- `calculate_rate_weight(input_rate) → 1 - |rate_in - rate_out| / max(rate_in, rate_out)`

**Semantic Invariant**:
- `|actual_interval - target_interval| < 2ms`

**Timelessness**: This is physics—will never change regardless of model architecture

**Public API**:
```rust
pub struct RateEquilibrium { ... }
pub struct RateConfig { pub target_rate_hz: f64, pub jitter_threshold_ms: f64 }

impl RateEquilibrium {
    pub fn new(config: RateConfig) -> Result<Self, RateError>
    pub fn on_rate_change(&self, new_rate: f64) -> Result<(), RateError>
    pub fn calculate_interval_ns(&self, rate_hz: f64) -> u64
    pub fn get_target_rate(&self) -> f64
    pub fn calculate_rate_weight(&self, input_rate: f64) -> f64
}
```

---

### 2. ContextSpec (The Spatial Noun Phrase)

**Type**: Rust struct `ContextEquilibrium<VS: VectorStore>`

**Location**: `src/constraint_grammar/context.rs`

**Purpose**: Specifies the basin in frozen territory

**Grammatical Role**: Follows RateSpec, precedes SentimentSpec

**Terminal Values**:
- `context_vec: Tensor` - Current conversation embedding
- `rag_index: VS` - Local knowledge index
- `sentiment: VADScores` - Current sentiment for weighting

**Production Rules**:
- `navigate(context) → interpolate_weighted(basins, sentiment_weights)`
- `cosine_similarity(query, basin) → similarity_score`

**Semantic Invariant**:
- `cosine_similarity(query, selected_basin) > 0.7`

**Timelessness**: This is geometry—will never change regardless of embedding dimension

**Public API**:
```rust
pub struct ContextEquilibrium<VS: VectorStore> { ... }
pub struct VADScores { pub valence: f64, pub arousal: f64, pub dominance: f64 }
pub struct Tensor { data: Vec<f64>, shape: Vec<usize> }
pub struct NavigationResult { pub path: Tensor, pub confidence: f64, pub nearby_basins: Vec<ContextBasin> }

impl<VS: VectorStore> ContextEquilibrium<VS> {
    pub fn new(context_vec: Tensor, rag_index: VS) -> Self
    pub fn navigate(&mut self, context: Tensor) -> Result<NavigationResult, ContextError>
    pub fn update_sentiment(&mut self, sentiment: VADScores)
    pub fn equilibrium(&self) -> f64
}
```

---

### 3. InterruptionSpec (The Reset Conjunction)

**Type**: Rust struct `InterruptionEquilibrium`

**Location**: `src/constraint_grammar/interruption.rs`

**Purpose**: Resets the attention stack

**Grammatical Role**: Can appear anywhere, triggers recomputation

**Terminal Values**:
- `reset_threshold: f64` - Attention weight after reset (default 0.8)
- `confidence_threshold: f64` - Minimum confidence for reset (0.7)

**Production Rules**:
- `handle_interruption(event) → if confidence > 0.7 then reset_attention()`
- `is_high_confidence() → confidence > 0.7`

**Semantic Invariant**:
- `P(reset | confidence > 0.7) = 1.0`

**Timelessness**: This is logic—will never change regardless of interruption modality

**Public API**:
```rust
pub struct InterruptionEquilibrium { ... }
pub struct InterruptionEvent { pub timestamp: f64, pub source: String, pub confidence: f64, pub sentiment_delta: f64 }
pub struct InterruptionResult { pub attention_reset: bool, pub reset_weight: f64, pub new_path: Option<Vec<f64>>, pub sentiment_context: f64 }

impl InterruptionEquilibrium {
    pub fn new() -> Self
    pub fn with_threshold(reset_threshold: f64) -> Self
    pub fn handle_interruption(&mut self, event: &InterruptionEvent) -> InterruptionResult
    pub fn attention_weight(&self) -> f64
    pub fn reset_attention(&mut self)
    pub fn restore_attention(&mut self)
    pub fn equilibrium_weight(&self) -> f64
}
```

---

### 4. SentimentSpec (The Affective Adjective)

**Type**: Rust struct `SentimentEquilibrium`

**Location**: `src/constraint_grammar/sentiment.rs`

**Purpose**: Weights paths by emotional valence

**Grammatical Role**: Modifies ContextSpec and Navigation decisions

**Terminal Values**:
- `vad_history: Vec<VADScores>` - Rolling window of VAD scores
- `window_size: usize` - Size of rolling window (default 100)
- `equilibrium_weight: f64` - Current equilibrium weight

**Production Rules**:
- `update_sentiment(vad) → average_weighted(window, recency)`
- `conduct(vad) → path_weight = valence * arousal * dominance`

**Semantic Invariant**:
- `sentiment_weight ∈ [0,1]` where `0 = negative, 1 = positive`

**Timelessness**: This is affective science—VAD model is stable across cultures

**Public API**:
```rust
pub struct SentimentEquilibrium { ... }
pub struct VADScores { pub valence: f64, pub arousal: f64, pub dominance: f64 }
pub struct ConductorResult { pub path_weight: f64, pub suggested_path: Vec<f64>, pub emotional_context: VADScores }

impl SentimentEquilibrium {
    pub fn new() -> Self
    pub fn with_window_size(window_size: usize) -> Self
    pub fn update_sentiment(&mut self, vad: VADScores) -> Result<(), SentimentError>
    pub fn conduct(&self, vad: &VADScores) -> ConductorResult
    pub fn equilibrium_weight(&self) -> f64
    pub fn history_size(&self) -> usize
    pub fn average_sentiment(&self) -> Result<VADScores, SentimentError>
}
```

---

### 5. NavigationSpec (The Verbal Phrase)

**Type**: Rust struct `EquilibriumOrchestrator`

**Location**: `src/equilibrium_orchestrator/orchestrator.rs`

**Purpose**: The action of moving through frozen territory

**Grammatical Role**: Head of the constraint phrase, takes all other specs as arguments

**Terminal Values**:
- Composition of all other specs

**Production Rules**:
- `orchestrate(input, K) → OutputPath where ∏ constraints → max`
- `confidence = rate_weight * context_weight * interruption_weight * sentiment_weight`

**Semantic Invariant**:
- `confidence = ∏ Φᵢ(x, κᵢ)`

**Timelessness**: This is compositionality—will never change regardless of constraint count

**Public API**:
```rust
pub struct EquilibriumOrchestrator { ... }
pub struct EquilibriumResult { pub confidence: f64, pub suggested_path: Vec<f64>, pub rate_target: f64, pub attention_reset: bool, pub state: OrchestratorState }

impl EquilibriumOrchestrator {
    pub fn new(target_rate_hz: f64, rag_index: Option<RAGIndex>, sentiment_initial: VADScores) -> Result<Self, OrchestratorError>
    pub async fn orchestrate(&mut self, incoming_tokens: Vec<String>, rate: f64, context: Vec<f64>, interruption: bool, sentiment: VADScores) -> Result<EquilibriumResult, OrchestratorError>
    pub fn state(&self) -> &OrchestratorState
    pub fn reset(&mut self)
    pub fn rate_equilibrium(&self) -> &RateEquilibrium
    pub fn sentiment_equilibrium(&self) -> &SentimentEquilibrium
}
```

---

### 6. VectorStore (The Frozen Territory)

**Type**: Rust struct `VectorStoreImpl`

**Location**: `src/token_organization/vector_store.rs`

**Purpose**: Local semantic search without cloud dependencies

**Grammatical Role**: Provides the "real estate" for context navigation

**Terminal Values**:
- `vectors: HashMap<String, VectorEntry>` - Stored vectors
- `dimensions: usize` - Vector dimensionality
- `threshold: f64` - Similarity threshold for search (default 0.7)

**Production Rules**:
- `add_vector(id, vector, metadata) → store(vector)`
- `search(query, k) → top_k_by_cosine_similarity(query)`

**Semantic Invariant**:
- `cosine_similarity(result, query) > threshold`

**Timelessness**: This is data storage—vectors and similarity are universal concepts

**Public API**:
```rust
pub struct VectorStoreImpl { ... }
pub struct VectorEntry { pub id: String, pub vector: Vec<f64>, pub metadata: serde_json::Value, pub timestamp: u64 }
pub struct SearchResult { pub id: String, pub vector: Vec<f64>, pub similarity: f64, pub metadata: serde_json::Value }

impl VectorStoreImpl {
    pub fn new(dimensions: usize) -> Self
    pub fn with_threshold(dimensions: usize, threshold: f64) -> Self
    pub fn add_vector(&self, id: String, vector: Vec<f64>, metadata: serde_json::Value) -> Result<(), VectorStoreError>
    pub fn search(&self, query: &[f64], k: usize) -> Result<Vec<SearchResult>, VectorStoreError>
    pub fn get_current(&self) -> Result<Vec<f64>, VectorStoreError>
    pub fn get(&self, id: &str) -> Result<Vec<f64>, VectorStoreError>
    pub fn remove(&self, id: &str) -> Result<(), VectorStoreError>
    pub fn size(&self) -> usize
}
```

---

### 7. RAGIndex (The Context Map)

**Type**: Rust struct `RAGIndex`

**Location**: `src/token_organization/rag_index.rs`

**Purpose**: Retrieval-Augmented Generation indexing for conversation context

**Grammatical Role**: Maps conversation history to semantic space

**Terminal Values**:
- `store: VectorStoreImpl` - Underlying vector storage
- `dimensions: usize` - Embedding dimension

**Production Rules**:
- `add_document(id, embedding, metadata) → index(embedding)`
- `search(query_embedding, k) → retrieve_relevant(query, k)`

**Semantic Invariant**:
- `retrieved_docs = semantic_neighbors(query, k)`

**Timelessness**: This is information retrieval—semantic search is model-agnostic

**Public API**:
```rust
pub struct RAGIndex { ... }

impl RAGIndex {
    pub fn new(dimensions: usize) -> Self
    pub fn add_document(&self, id: String, embedding: Vec<f64>, metadata: serde_json::Value) -> Result<(), RAGIndexError>
    pub fn search(&self, query_embedding: &[f64], k: usize) -> Result<Vec<ContextBasin>, RAGIndexError>
    pub fn get_current(&self) -> Result<Tensor, RAGIndexError>
    pub fn size(&self) -> usize
}
```

---

### 8. TokenEmbedder (The Token Encoder)

**Type**: Rust struct `TokenEmbedder`

**Location**: `src/token_organization/embedding.rs`

**Purpose**: Converts tokens to vector embeddings

**Grammatical Role**: Converts discrete tokens to continuous territory

**Terminal Values**:
- `dimension: usize` - Embedding dimension

**Production Rules**:
- `embed_token(token) → vector`
- `embed_tokens(tokens) → average(vectors)`

**Semantic Invariant**:
- `similar_tokens → similar_embeddings`

**Timelessness**: This is representation learning—embeddings are universal

**Public API**:
```rust
pub struct TokenEmbedder { ... }

impl TokenEmbedder {
    pub fn new(dimension: usize) -> Self
    pub fn default() -> Self  // 384-dim
    pub fn embed_token(&self, token: &str) -> Result<Vec<f64>, EmbeddingError>
    pub fn embed_tokens(&self, tokens: &[String]) -> Result<Vec<f64>, EmbeddingError>
    pub fn dimension(&self) -> usize
}
```

---

## Component Dependencies

```
EquilibriumOrchestrator
├── RateEquilibrium (independent)
├── ContextEquilibrium
│   └── RAGIndex
│       └── VectorStoreImpl
├── InterruptionEquilibrium (independent)
└── SentimentEquilibrium (independent)
```

## Testing Requirements

Each component must have:
1. Unit tests for all public methods
2. Property-based tests for invariants
3. Integration tests with dependent components
4. Edge case coverage (empty inputs, extreme values)

## Documentation Requirements

Each component must have:
1. Rust doc comments on all public items
2. Usage examples in doc comments
3. Explanation of grammatical role
4. Statement of semantic invariants
5. Note on timelessness (why this outlasts specific implementations)
