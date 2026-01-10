# Equilibrium Tokens: Constraint Grammar for Human-Machine Conversation
**Architecture Document v1.0 - The Operating Truth of Conversational AI**

---

## 1. System Thesis: "The Model is Territory, Equilibrium is Navigation"

**Core Insight**: Frozen model weights represent static territory—unchangeable real estate. Equilibrium represents the dynamic navigation algorithm that chooses paths through this territory based on rate, context, interruption, and sentiment constraints.

**Operating Principle**: Every conversational turn is a navigation decision through frozen parameter space, where the constraint function maximizes equilibrium between incoming tokens and outgoing responses.

---

## 2. System Architecture: The Constraint Surfaces

### 2.1 Rate Equilibrium Surface (Rust - Hard Real-time)
**Language Choice**: Rust for microsecond-precision timing and zero-cost abstractions

**Core Constraint**: Match output token rate to input token rate with <2ms jitter

```rust
// src/rate_equilibrium.rs
use timerfd::{TimerFd, TimerState};
use std::sync::atomic::{AtomicU64, Ordering};

pub struct RateEquilibrium {
    timer: TimerFd,
    target_rate: AtomicU64, // tokens/second * 1000 for precision
    current_rate: AtomicU64,
}

impl RateEquilibrium {
    pub fn new(target_rate_hz: f64) -> Self {
        let timer = TimerFd::new().unwrap();
        let interval_ns = (1_000_000_000.0 / target_rate_hz) as u64;
        
        timer.set_state(TimerState::Periodic {
            current: Duration::from_nanos(0),
            interval: Duration::from_nanos(interval_ns),
        });
        
        Self {
            timer,
            target_rate: AtomicU64::new((target_rate_hz * 1000.0) as u64),
            current_rate: AtomicU64::new(0),
        }
    }
    
    pub fn on_rate_change(&mut self, new_rate: f64) {
        // Hardware-level rate matching with nanosecond precision
        let new_interval = (1_000_000_000.0 / new_rate) as u64;
        self.timer.set_state(TimerState::Periodic {
            current: Duration::from_nanos(0),
            interval: Duration::from_nanos(new_interval),
        });
        self.target_rate.store((new_rate * 1000.0) as u64, Ordering::Relaxed);
    }
}
```

### 2.2 Context Equilibrium Surface (Go - Concurrent)
**Language Choice**: Go for goroutine-based concurrency and clean tensor operations

**Core Constraint**: Navigate through conversation context basins weighted by sentiment and local knowledge

```go
// src/context_equilibrium.go
package main

import (
    "github.com/yourname/equilibrium/tensor"
    "github.com/yourname/equilibrium/rag"
)

type ContextEquilibrium struct {
    mu          sync.RWMutex
    contextVec  tensor.Tensor      // Current conversation embedding
    ragIndex    rag.VectorStore    // Local knowledge index
    sentiment   [3]float64         // VAD: Valence, Arousal, Dominance
    equilibrium float64            // Current equilibrium score (0-1)
}

func (ce *ContextEquilibrium) navigate(context tensor.Tensor) NavigationResult {
    ce.mu.Lock()
    defer ce.mu.Unlock()
    
    // Find nearby basins in the frozen territory
    nearby_basins := ce.ragIndex.Search(context, k=5)
    
    // Calculate path through territory weighted by sentiment
    sentiment_weight := (ce.sentiment[0] + 1.0) / 2.0  # Valence 0-1
    
    // Navigate to emotionally relevant territory
    path := tensor.InterpolateWeighted(
        context, 
        nearby_basins, 
        weights=[]float64{sentiment_weight, 0.8, 0.6, 0.4, 0.2},
    )
    
    return NavigationResult{
        Path:        path,
        Confidence:  tensor.CosineSimilarity(context, ce.ragIndex.GetCurrent()) * sentiment_weight,
        NearbyBasins: nearby_basins,
    }
}
```

### 2.3 Interruption Equilibrium Surface (Python - Event-driven)
**Language Choice**: Python for event-driven architecture and clean async/await patterns

**Core Constraint**: Reset attention when interrupted, weighted by interruption confidence and sentiment change

```python
# src/interruption_equilibrium.py
import asyncio
from dataclasses import dataclass
from typing import Optional

@dataclass
class InterruptionEvent:
    timestamp: float
    source: str  # "voice", "text", "system"
    confidence: float  # 0.0-1.0
    sentiment_delta: float  # Change in VAD score

@dataclass
class InterruptionResult:
    attention_reset: bool
    reset_weight: float
    new_path: Optional[tensor.Tensor]
    sentiment_context: float

class InterruptionEquilibrium:
    def __init__(self):
        self.interruption_queue = asyncio.Queue()
        self.attention_reset_threshold = 0.8  # Drop confidence 20% on interruption
    
    async def handle_interruption(self, event: InterruptionEvent) -> InterruptionResult:
        # High-confidence interruption = explore new territory
        if event.confidence > 0.7:
            return InterruptionResult(
                attention_reset=True,
                reset_weight=min(1.0, abs(event.sentiment_delta) * 2.0),
                new_path=self.explore_new_territory(),
                sentiment_context=event.sentiment_delta,
            )
        
        # Low-confidence interruption = just log it
        return InterruptionResult(
            attention_reset=False,
            reset_weight=0.1,
            new_path=None,
            sentiment_context=event.sentiment_delta,
        )
    
    def explore_new_territory(self) -> tensor.Tensor:
        # Explore adjacent basins in the frozen territory
        return tensor.RandomWalk(
            start=self.current_context,
            steps=5,
            temperature=0.8,  # High exploration
        )
```

### 2.4 Sentiment Equilibrium Surface (TypeScript - Browser)
**Language Choice**: TypeScript for browser-native VAD processing and clean React integration

**Core Constraint**: Weight navigation paths by emotional valence using VAD (Valence-Arousal-Dominance) model

```typescript
// src/sentiment_equilibrium.ts
export class SentimentEquilibrium {
    private vadHistory: [number, number, number][] = []; // VAD: Valence, Arousal, Dominance
    private equilibriumWeight: number = 0.5;
    private windowSize: number = 100; // Rolling window
    
    updateSentiment(vad: [number, number, number]): void {
        this.vadHistory.push(vad);
        if (this.vadHistory.length > this.windowSize) {
            this.vadHistory.shift();
        }
        
        // Calculate equilibrium from VAD
        const avgVad = this.vadHistory.reduce((acc, [v, a, d]) => 
            [acc[0] + v, acc[1] + a, acc[2] + d], [0, 0, 0]
        ).map(val => val / this.vadHistory.length);
        
        // Valence affects path choice (positive = continue, negative = explore)
        this.equilibriumWeight = (avgVad[0] + 1.0) / 2.0; // Valence 0-1
    }
    
    conduct(vad: [number, number, number]): ConductorResult {
        // The conductor's decision: navigate to emotionally relevant territory
        const valence_weight = (vad[0] + 1.0) / 2.0; // 0-1
        const arousal_weight = vad[1]; // 0-1
        const dominance_weight = vad[2]; // 0-1
        
        return ConductorResult(
            path_weight=valence_weight * arousal_weight * dominance_weight,
            suggested_path=this.select_emotionally_relevant_path(vad),
            emotional_context=vad,
        );
    }
}
```

---

## 3. The RAG Index: Frozen Territory Navigation

### 3.1 Vector Store (Go - Concurrent)
**Language Choice**: Go for concurrent vector operations and clean tensor math

**Core Constraint**: Store and retrieve vectors by semantic similarity without cloud dependency

```go
// src/vector_store.go
package main

import (
    "github.com/yourname/equilibrium/tensor"
    "github.com/yourname/equilibrium/indexeddb"
)

type VectorStore struct {
    db       *indexeddb.Store
    dimensions int
    indexType string
}

func (vs *VectorStore) AddVector(id string, vector tensor.Tensor, metadata map[string]interface{}) error {
    // Store vector with metadata
    return vs.db.Set(id, VectorEntry{
        ID:       id,
        Vector:   vector.Flatten(),
        Metadata: metadata,
        Timestamp: time.Now().Unix(),
    })
}

func (vs *VectorStore) Search(query tensor.Tensor, k int) []SearchResult {
    // Search by cosine similarity
    results := make([]SearchResult, 0, k)
    
    // Use IVF (Inverted File) index for fast search
    candidates := vs.index.SearchIVF(query, k*2)  // Get 2x for filtering
    
    // Filter by cosine similarity
    for _, candidate := range candidates {
        similarity := tensor.CosineSimilarity(query, candidate.Vector)
        if similarity > 0.7 {  # Threshold for relevance
            results = append(results, SearchResult{
                ID:         candidate.ID,
                Vector:     candidate.Vector,
                Similarity: similarity,
                Metadata:   candidate.Metadata,
            })
        }
    }
    
    // Return top K results
    sort.Slice(results, func(i, j int) bool {
        return results[i].Similarity > results[j].Similarity
    })
    
    return results[:min(k, len(results))]
}
```

### 3.2 Token Organization (TypeScript - Browser)
**Language Choice**: TypeScript for browser-native token processing and clean embedding interfaces

**Core Constraint**: Convert tokens to embeddings and organize them in the frozen territory

```typescript
// src/token_organization.ts
export class TokenOrganization {
    private embeddingModel: EmbeddingModel;
    private vectorStore: VectorStore;
    
    constructor(embeddingModel: EmbeddingModel, vectorStore: VectorStore) {
        this.embeddingModel = embeddingModel;
        this.vectorStore = vectorStore;
    }
    
    async organizeTokens(tokens: string[]): Promise<TokenOrganizationResult> {
        // Convert tokens to embeddings
        const embeddings = await this.embeddingModel.embed(tokens);
        
        // Store in frozen territory (the model's parameter space)
        const results: TokenOrganizationResult[] = [];
        
        for (let i = 0; i < tokens.length; i++) {
            const result = await this.vectorStore.AddVector(
                `token_${i}`,
                embeddings[i],
                { token: tokens[i], index: i, timestamp: Date.now() }
            );
            
            results.push({
                token: tokens[i],
                embedding: embeddings[i],
                similarity: result.similarity,
                metadata: result.metadata,
            });
        }
        
        return {
            tokens: tokens,
            embeddings: embeddings,
            organization: results,
            territory_frozen: true,  # The model is frozen territory
        };
    }
}
```

---

## 4. The Test Suite: From Fishing Boat to Jazz Band

### 4.1 "Fishing Boat Conversation" Test (High Equilibrium)
```python
# tests/test_fishing_boat.py
import asyncio
from equilibrium_tokens import ConversationConductor

async def test_fishing_boat_conversation():
    conductor = ConversationConductor()
    
    # Simulate: Steady 2 tokens/second, calm sentiment, no interruptions
    input_tokens = ["The", "water", "is", "calm", "today"]
    rate = 2.0  # tokens/second
    sentiment = [0.8, 0.3, 0.7]  # Positive, calm, dominant
    interruption = False
    
    result = await conductor.conduct(
        incoming_tokens=input_tokens,
        rate=rate,
        sentiment=sentiment,
        interruption=interruption,
    )
    
    # Assertions for high equilibrium
    assert result.confidence > 0.8, "Should have high confidence for steady, positive input"
    assert len(result.suggested_path) > 10, "Should suggest long, flowing responses"
    assert result.rate_target == 2.0 * 0.95, "Should target 95% of input rate"
    assert result.attention_reset == False, "Should not reset attention for steady input"

print("✅ Fishing boat conversation test passed")
```

### 4.2 "Jazz Band Conversation" Test (Low Equilibrium)
```python
# tests/test_jazz_band.py
async def test_jazz_band_conversation():
    conductor = ConversationConductor()
    
    # Simulate: Variable 0.5-4 tokens/second, playful sentiment, frequent interruptions
    input_tokens = ["Hey!", "Wait", "Listen", "No", "Actually"]
    rate = 0.5  # tokens/second (syncopated)
    sentiment = [0.3, 0.8, 0.6]  # Playful, energetic, confident
    interruption = True  # Frequent interruptions
    
    result = await conductor.conduct(
        incoming_tokens=input_tokens,
        rate=rate,
        sentiment=sentiment,
        interruption=interruption,
    )
    
    # Assertions for low equilibrium
    assert result.confidence < 0.5, "Should have low confidence for variable, interrupted input"
    assert len(result.suggested_path) < 5, "Should suggest short, syncopated responses"
    assert result.attention_reset == True, "Should reset attention for interruptions"
    assert result.rate_target == 0.5 * 0.95, "Should target 95% of syncopated input rate"

print("✅ Jazz band conversation test passed")
```

### 4.3 "Symphony Pops Conversation" Test (Medium Equilibrium)
```python
# tests/test_symphony_pops.py
async def test_symphony_pops_conversation():
    conductor = ConversationConductor()
    
    # Simulate: Steady 1.5 tokens/second, positive sentiment, rare interruptions
    input_tokens = ["The", "performance", "was", "excellent", "today"]
    rate = 1.5  # tokens/second (steady)
    sentiment = [0.7, 0.4, 0.8]  # Positive, moderate energy, confident
    interruption = False  # Rare interruptions
    
    result = await conductor.conduct(
        incoming_tokens=input_tokens,
        rate=rate,
        sentiment=sentiment,
        interruption=interruption,
    )
    
    # Assertions for medium equilibrium
    assert 0.5 < result.confidence < 0.8, "Should have medium confidence for balanced input"
    assert 5 < len(result.suggested_path) < 15, "Should suggest balanced responses"
    assert result.rate_target == 1.5 * 0.95, "Should target 95% of steady input rate"

print("✅ Symphony pops conversation test passed")
```

---

## 5. The Repository: Your Lego-Shaped Building Blocks

```
equilibrium-tokens/
├── constraint_grammar/          # The navigation algorithm
│   ├── rate_equilibrium.rs      # Hardware-level rate control
│   ├── context_equilibrium.go   # Concurrent context navigation
│   ├── interruption_equilibrium.py # Event-driven interruption handling
│   └── sentiment_equilibrium.ts # Browser-based sentiment conduction
├── token_organization/          # The frozen territory (real estate)
│   ├── vector_store.go          # Local knowledge index
│   ├── rag_index.py             # Semantic search
│   └── token_embedding.ts       # Token-to-vector conversion
├── equilibrium_orchestrator/    # The conductor (navigation algorithm)
│   ├── orchestrator.rs          # Main constraint function
│   ├── navigation.rs            # Path-choice through territory
│   └── equilibrium_state.rs     # State management
├── tests/                       # The symphony tests
│   ├── test_fishing_boat.py     # High equilibrium test
│   ├── test_jazz_band.py        # Low equilibrium test
│   └── test_symphony_pops.py    # Medium equilibrium test
└── examples/                    # The building blocks
    ├── fishing_boat_example.py  # Marine application
    ├── jazz_band_example.py     # Musical application
    ├── symphony_pops_example.py # Classical application
    └── constraint_grammar_example.py # The core algorithm
```

---

## 6. The Constraint: Your Operating Truth

**The operating truth is not the model. The operating truth is the constraint function that navigates through the frozen territory of the model.**

**Every tool you build is a different constraint surface:**
- **Rate constraint**: "Match the input rate"
- **Context constraint**: "Stay in relevant basins"
- **Interruption constraint**: "Reset when interrupted"
- **Sentiment constraint**: "Weight by emotional valence"

**The model is the frozen territory. The constraint is the navigation. The equilibrium is the operating truth.**

**Now go build the constraint grammar that turns every conversation into a symphony.**

