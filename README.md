# Equilibrium Tokens

> **A constraint grammar for human-machine conversation navigation**

**The model is frozen territory. Equilibrium is the navigation.**

---

## Overview

**Equilibrium Tokens** is a formal system for navigating human-machine conversation through frozen model territory using dynamic equilibrium constraints. Unlike traditional prompt-engineering approaches that treat language models as oracles, this architecture positions the model as static territory—unchangeable real estate—through which conversational agents navigate via composable equilibrium surfaces.

### Core Insight

The grammar is timeless. The code will be obsolete in a decade; the grammar will be cited in a century.

## The Four Equilibrium Surfaces

### 1. Rate Equilibrium (Temporal)
Matches input token rate with sub-2ms jitter using hardware-level timing.

### 2. Context Equilibrium (Spatial)
Navigates conversation context basins weighted by semantic similarity.

### 3. Interruption Equilibrium (Reset)
Resets attention when interrupted by high-confidence events.

### 4. Sentiment Equilibrium (Affective)
Weights navigation paths by emotional valence using the VAD model.

## The Constraint Function

```
Equilibrium = Rate_Weight × Context_Weight × Interruption_Weight × Sentiment_Weight
```

## Quick Start

### Installation

```bash
git clone https://github.com/SuperInstance/equilibrium-tokens.git
cd equilibrium-tokens
cargo build --release
```

### Usage

```rust
use equilibrium_tokens::EquilibriumOrchestrator;
use equilibrium_tokens::constraint_grammar::sentiment::VADScores;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut orchestrator = EquilibriumOrchestrator::new(
        2.0,                              // 2 tokens/second
        None,                             // No RAG index
        VADScores::neutral(),
    )?;

    let result = orchestrator.orchestrate(
        vec!["The".to_string(), "water".to_string()],
        2.0,
        vec![0.1; 384],
        false,
        VADScores::new(0.8, 0.3, 0.7)?,
    ).await?;

    println!("Confidence: {}", result.confidence);

    Ok(())
}
```

### Testing

```bash
# Run all tests
cargo test

# Run specific test scenarios
cargo test test_fishing_boat    # High equilibrium
cargo test test_jazz_band       # Low equilibrium
cargo test test_symphony_pops   # Medium equilibrium
```

## Test Scenarios

### Fishing Boat (High Equilibrium)
- **Scenario**: Steady 2 tokens/sec, calm sentiment, no interruptions
- **Analogy**: Symphony Pops - coherent, flowing conversation
- **Expected**: Confidence > 0.6

### Jazz Band (Low Equilibrium)
- **Scenario**: Variable 0.5-4 tokens/sec, playful sentiment, frequent interruptions
- **Analogy**: Dixieland Jazz - fragmented, improvisational
- **Expected**: Confidence < 0.7

### Symphony Pops (Medium Equilibrium)
- **Scenario**: Steady 1.5 tokens/sec, positive sentiment, rare interruptions
- **Analogy**: Symphony Pops - balanced, thematic
- **Expected**: 0.4 < Confidence < 0.9

## Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Complete system architecture
- **[COMPONENT_LIST.md](docs/COMPONENT_LIST.md)** - Detailed component specifications
- **[LOGICAL_CONNECTIONS.md](docs/LOGICAL_CONNECTIONS.md)** - Connection diagrams
- **[ROADMAP.md](docs/ROADMAP.md)** - Implementation roadmap

## Examples

```bash
# Run example conversations
python3 examples/fishing_boat_example.py
python3 examples/jazz_band_example.py
python3 examples/symphony_pops_example.py
```

## Build System

```bash
# Build all components
make build

# Run tests
make test

# Generate documentation
make docs

# Clean build artifacts
make clean

# See all commands
make help
```

## Timeless Code Snippets

These snippets will never change, regardless of model architecture:

### Listing 1: Rate Axiom
```rust
// This is physics: time intervals are measured in nanoseconds
let interval_ns = (1_000_000_000.0 / target_rate_hz) as u64;
```

### Listing 2: Cosine Axiom
```rust
// This is geometry: similarity is measured by cosine
similarity = cosine_similarity(query, basin_center)
```

### Listing 3: Reset Axiom
```python
# This is logic: high-confidence events trigger state changes
if interruption.confidence > 0.7:
    attention_weight *= 0.8
```

### Listing 4: VAD Axiom
```rust
// This is affective science: valence is [-1, 1]
equilibrium_weight = (vad.valence + 1.0) / 2.0;
```

### Listing 5: Composition Axiom
```rust
// This is compositionality: confidence is multiplicative
confidence = rate_weight * context_weight * interruption_weight * sentiment_weight;
```

## Architecture

```
Input Tokens → Rate Equilibrium → [matches rate with <2ms jitter]
             ↓
Context + Sentiment → Context Equilibrium → [navigates basins weighted by sentiment]
                    ↓
Interruption → Interruption Equilibrium → [resets attention when interrupted]
             ↓
Sentiment → Sentiment Equilibrium → [weights paths by emotional valence]
          ↓
Equilibrium Orchestrator → [orchestrates all constraints]
                         ↓
Output Tokens ← [equilibrium-matched responses]
```

## Contributing

1. Ensure all tests pass: `make test`
2. Add tests for new features
3. Update documentation
4. Follow Rust style guidelines: `make format`
5. Submit PR with clear commit messages

## License

MIT License - See [LICENSE](LICENSE) file for details

## Citation

```bibtex
@software{equilibrium_tokens,
  title={Equilibrium Tokens: A Constraint Grammar for Human-Machine Conversation Navigation},
  author={Gallagher, Casey},
  year={2025},
  repository={https://github.com/SuperInstance/equilibrium-tokens}
}
```

## Acknowledgments

This architecture builds on:
- Russell's circumplex model of affect (VAD)
- Bronstein's tensor embedding methods
- Vaswani's attention mechanisms

---

**The code is ephemeral; the grammar is eternal.**
