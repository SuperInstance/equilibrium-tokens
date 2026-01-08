//! Equilibrium Orchestrator
//!
//! Main constraint function. Orchestrates all equilibrium surfaces
//! to navigate conversation through frozen territory.
//!
//! # Timeless Code Principles
//!
//! This module implements the **main constraint function** that composes all
//! equilibrium surfaces multiplicatively to produce a single confidence score.
//!
//! ## Timeless Code (Listing 5): The Composition Axiom
//!
//! ```rust
//! // This is compositionality: confidence is multiplicative
//! confidence = rate_weight * context_weight * interruption_weight * sentiment_weight;
//! ```
//!
//! This is timeless because it's how independent probabilities combine.
//!
//! ### Why Multiplicative Composition Is Timeless
//!
//! 1. **Probability theory**: For independent events A, B, C, D:
//!    ```text
//!    P(A ∧ B ∧ C ∧ D) = P(A) × P(B) × P(C) × P(D)
//!    ```
//!
//! 2. **Information theory**: Independent constraints multiply entropy:
//!    ```text
//!    H(total) = H(rate) + H(context) + H(interruption) + H(sentiment)
//!    ```
//!
//! 3. **Bayesian inference**: Posterior ∝ Prior × Likelihood × Evidence
//!
//! 4. **Fuzzy logic**: AND operation = min(x, y) ≈ x × y for small values
//!
//! This composition rule will be valid as long as:
//! - We model uncertainty with probabilities [0, 1]
//! - We treat constraints as independent
//! - We use Bayesian reasoning
//!
//! ### What Makes This Architecture Timeless
//!
//! The **Equilibrium Tokens** architecture separates concerns into independent
//! dimensions (rate, context, interruption, sentiment), then composes them
//! multiplicatively. This is timeless because:
//!
//! 1. **Modular**: Each constraint can be updated independently
//! 2. **Composable**: New constraints multiply into existing ones
//! 3. **Bounded**: Result always in [0, 1] if inputs are in [0, 1]
//! 4. **Interpretable**: Low confidence means "something violated"

use crate::constraint_grammar::{
    rate::{RateEquilibrium, RateConfig},
    context::{ContextEquilibrium, Tensor, VectorStore as ContextVectorStore},
    interruption::{InterruptionEquilibrium, InterruptionEvent},
    sentiment::{SentimentEquilibrium, VADScores},
};
use crate::token_organization::rag_index::RAGIndex;
use crate::equilibrium_orchestrator::state::OrchestratorState;
use thiserror::Error;

/// Errors that can occur during orchestration
#[derive(Debug, Error)]
pub enum OrchestratorError {
    #[error("Rate error: {0}")]
    RateError(#[from] crate::constraint_grammar::rate::RateError),

    #[error("Context error: {0}")]
    ContextError(#[from] crate::constraint_grammar::context::ContextError),

    #[error("Interruption error: {0}")]
    InterruptionError(#[from] crate::constraint_grammar::interruption::InterruptionError),

    #[error("Sentiment error: {0}")]
    SentimentError(#[from] crate::constraint_grammar::sentiment::SentimentError),

    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

/// Result of equilibrium orchestration
#[derive(Debug, Clone)]
pub struct EquilibriumResult {
    /// Overall confidence score (0-1)
    pub confidence: f64,
    /// Suggested path through territory
    pub suggested_path: Vec<f64>,
    /// Target rate in Hz
    pub rate_target: f64,
    /// Whether attention should be reset
    pub attention_reset: bool,
    /// Orchestrator state
    pub state: OrchestratorState,
}

/// Adapter to make RAGIndex work with ContextEquilibrium
struct RAGAdapter {
    rag_index: RAGIndex,
}

impl ContextVectorStore for RAGAdapter {
    fn search(&self, query: &Tensor, k: usize) -> Result<Vec<crate::constraint_grammar::context::ContextBasin>, crate::constraint_grammar::context::ContextError> {
        self.rag_index.search(query.data(), k)
            .map_err(|e| crate::constraint_grammar::context::ContextError::VectorStore(
                format!("RAG search error: {}", e)
            ))
    }

    fn get_current(&self) -> Result<Tensor, crate::constraint_grammar::context::ContextError> {
        self.rag_index.get_current()
            .map_err(|e| crate::constraint_grammar::context::ContextError::VectorStore(
                format!("RAG get_current error: {}", e)
            ))
    }
}

/// Main equilibrium orchestrator
///
/// Composes all equilibrium surfaces to implement the constraint function:
///
/// # Timeless Code (Listing 5)
///
/// ```rust
/// // This is compositionality: confidence is multiplicative
/// confidence = rate_weight * context_weight * interruption_weight * sentiment_weight;
/// ```
///
/// This is timeless because it's how independent probabilities combine.
pub struct EquilibriumOrchestrator {
    /// Rate equilibrium controller
    rate_eq: RateEquilibrium,
    /// Context equilibrium controller
    context_eq: Option<ContextEquilibrium<RAGAdapter>>,
    /// Interruption equilibrium controller
    interruption_eq: InterruptionEquilibrium,
    /// Sentiment equilibrium controller
    sentiment_eq: SentimentEquilibrium,
    /// Current state
    state: OrchestratorState,
}

impl EquilibriumOrchestrator {
    /// Create a new equilibrium orchestrator
    ///
    /// # Arguments
    ///
    /// * `target_rate_hz` - Initial target rate in tokens/second
    /// * `rag_index` - Optional RAG index for context navigation
    /// * `sentiment_initial` - Initial VAD scores
    pub fn new(
        target_rate_hz: f64,
        rag_index: Option<RAGIndex>,
        _sentiment_initial: VADScores,
    ) -> Result<Self, OrchestratorError> {
        let rate_config = RateConfig::new(target_rate_hz);
        let rate_eq = RateEquilibrium::new(rate_config)?;

        let context_eq = rag_index.map(|rag| {
            ContextEquilibrium::new(
                Tensor::new(vec![0.0; 384]),
                RAGAdapter { rag_index: rag },
            )
        });

        Ok(Self {
            rate_eq,
            context_eq,
            interruption_eq: InterruptionEquilibrium::new(),
            sentiment_eq: SentimentEquilibrium::new(),
            state: OrchestratorState::new(),
        })
    }

    /// Orchestrate a conversation turn
    ///
    /// This is the **main constraint function** that composes all equilibrium
    /// surfaces to navigate through frozen territory.
    ///
    /// # Timeless Code (Listing 5): Composition Axiom
    ///
    /// ```rust
    /// // This is compositionality: confidence is multiplicative
    /// confidence = rate_weight * context_weight * interruption_weight * sentiment_weight;
    /// ```
    ///
    /// # Why This Composition Is Timeless
    ///
    /// 1. **Probability theory**: P(all constraints satisfied) = Π P(constraintᵢ)
    /// 2. **Bounded**: If all weights ∈ [0,1], then confidence ∈ [0,1]
    /// 3. **Interpretable**: Low confidence = "at least one constraint violated"
    /// 4. **Composable**: Adding new constraints = multiply by new weight
    ///
    /// # Algorithm Flow
    ///
    /// 1. **Rate Equilibrium**: Match input rate, calculate `rate_weight` ∈ [0, 1]
    /// 2. **Sentiment Equilibrium**: Update VAD, calculate `sentiment_weight` ∈ [0, 1]
    /// 3. **Context Equilibrium**: Navigate basins, calculate `context_weight` ∈ [0, 1]
    /// 4. **Interruption Equilibrium**: Check for reset, calculate `interruption_weight` ∈ [0, 1]
    /// 5. **Compose**: `confidence = rate_weight × context_weight × interruption_weight × sentiment_weight`
    /// 6. **Select Path**: Choose navigation path through frozen territory
    /// 7. **Update State**: Record equilibrium, rate, sentiment, context
    ///
    /// # Arguments
    ///
    /// * `incoming_tokens` - Input tokens (may be empty if context provided)
    /// * `rate` - Measured input rate in Hz
    /// * `context` - Context embedding vector (may be empty if tokens provided)
    /// * `interruption` - Whether high-confidence interruption occurred
    /// * `sentiment` - Current VAD scores (valence, arousal, dominance)
    ///
    /// # Returns
    ///
    /// An `EquilibriumResult` containing:
    /// - `confidence`: Overall equilibrium score [0, 1] (Timeless Code Listing 5)
    /// - `suggested_path`: Navigation path through frozen territory
    /// - `rate_target`: Target token emission rate (Hz)
    /// - `attention_reset`: Whether attention was reset by interruption
    /// - `state`: Updated orchestrator state
    ///
    /// # Errors
    ///
    /// Returns `OrchestratorError::InvalidInput` if both tokens and context are empty.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use equilibrium_tokens::EquilibriumOrchestrator;
    /// # use equilibrium_tokens::constraint_grammar::sentiment::VADScores;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut orchestrator = EquilibriumOrchestrator::new(
    ///     2.0,  // 2 Hz target rate
    ///     None,  // No RAG index
    ///     VADScores::neutral(),
    /// )?;
    ///
    /// // Steady conversation: high confidence
    /// let result = orchestrator.orchestrate(
    ///     vec!["The".to_string(), "water".to_string()],
    ///     2.0,
    ///     vec![0.1; 384],
    ///     false,
    ///     VADScores::new(0.8, 0.3, 0.7)?,
    /// ).await?;
    ///
    /// assert!(result.confidence > 0.3);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn orchestrate(
        &mut self,
        incoming_tokens: Vec<String>,
        rate: f64,
        context: Vec<f64>,
        interruption: bool,
        sentiment: VADScores,
    ) -> Result<EquilibriumResult, OrchestratorError> {
        // Validate input: need at least tokens or context
        if incoming_tokens.is_empty() && context.is_empty() {
            return Err(OrchestratorError::InvalidInput(
                "Either tokens or context must be provided".to_string(),
            ));
        }

        // Step 1: Rate equilibrium - match input rate
        // Calculates how well output rate matches input rate [0, 1]
        self.rate_eq.on_rate_change(rate)?;
        let rate_weight = self.rate_eq.calculate_rate_weight(rate);

        // Step 2: Sentiment equilibrium - update and weight by valence
        // Maps valence [-1, 1] to weight [0, 1]
        self.sentiment_eq.update_sentiment(sentiment.clone())?;
        let sentiment_weight = sentiment.equilibrium_weight();

        // Step 3: Context equilibrium - navigate context basins
        // Returns confidence in context navigation [0, 1]
        let context_weight = if let Some(ref mut context_eq) = self.context_eq {
            let context_tensor = Tensor::new(context.clone());
            match context_eq.navigate(context_tensor) {
                Ok(result) => result.confidence,
                Err(_) => 0.5, // Default if navigation fails
            }
        } else {
            0.5 // Default if no RAG index (neutral stance)
        };

        // Step 4: Interruption equilibrium - reset when interrupted
        // Returns attention weight after interruption [0, 1]
        let interruption_weight = if interruption {
            let event = InterruptionEvent::new("system".to_string(), 0.9, 0.0)?;
            let result = self.interruption_eq.handle_interruption(&event);
            result.reset_weight
        } else {
            1.0 // No interruption = full attention
        };

        // Step 5: Calculate equilibrium (Timeless Code Listing 5)
        // Confidence = product of all constraint weights
        // This is compositionality: independent probabilities multiply
        let confidence = rate_weight * context_weight * interruption_weight * sentiment_weight;

        // Step 6: Select path through frozen territory
        let suggested_path = if let Some(ref mut context_eq) = self.context_eq {
            let context_tensor = Tensor::new(context);
            context_eq.navigate(context_tensor)
                .map(|r| r.path.flatten())
                .unwrap_or_else(|_| vec
![0.0f64; 384])
        } else {
            vec
![0.0f64; 384]
        };

        // Step 7: Calculate rate target (slightly lag input for stability)
        // 95% lag prevents oscillation in rate control
        let rate_target = rate * 0.95;

        // Step 8: Update orchestrator state
        self.state.update_equilibrium(confidence);
        self.state.update_rate_target(rate_target);
        self.state.set_attention_reset(interruption);
        self.state.update_sentiment(sentiment);
        self.state.update_context(suggested_path.clone());

        Ok(EquilibriumResult {
            confidence,
            suggested_path,
            rate_target,
            attention_reset: interruption,
            state: self.state.clone(),
        })
    }

    /// Get current state
    pub fn state(&self) -> &OrchestratorState {
        &self.state
    }

    /// Reset orchestrator state
    pub fn reset(&mut self) {
        self.state.reset();
        self.interruption_eq.restore_attention();
    }

    /// Get rate equilibrium controller
    pub fn rate_equilibrium(&self) -> &RateEquilibrium {
        &self.rate_eq
    }

    /// Get sentiment equilibrium controller
    pub fn sentiment_equilibrium(&self) -> &SentimentEquilibrium {
        &self.sentiment_eq
    }
}

impl Default for EquilibriumOrchestrator {
    fn default() -> Self {
        Self::new(
            2.0,
            None,
            VADScores::neutral(),
        ).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_orchestrator_creation() {
        let orchestrator = EquilibriumOrchestrator::new(
            2.0,
            None,
            VADScores::neutral(),
        ).unwrap();

        assert_eq!(orchestrator.state().rate_target, 2.0);
    }

    #[tokio::test]
    async fn test_orchestrate() {
        let mut orchestrator = EquilibriumOrchestrator::new(
            2.0,
            None,
            VADScores::neutral(),
        ).unwrap();

        let result = orchestrator.orchestrate(
            vec
!["hello".to_string()],
            2.0,
            vec
![0.1; 384],
            false,
            VADScores::new(0.5, 0.7, 0.8).unwrap(),
        ).await.unwrap();

        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert_eq!(result.rate_target, 1.9); // 2.0 * 0.95
        assert!(!result.attention_reset);
    }

    #[tokio::test]
    async fn test_orchestrate_with_interruption() {
        let mut orchestrator = EquilibriumOrchestrator::new(
            2.0,
            None,
            VADScores::neutral(),
        ).unwrap();

        let result = orchestrator.orchestrate(
            vec
!["hello".to_string()],
            2.0,
            vec
![0.1; 384],
            true,
            VADScores::new(0.5, 0.7, 0.8).unwrap(),
        ).await.unwrap();

        assert!(result.attention_reset);
        assert!(result.confidence < 1.0); // Should be reduced
    }

    #[tokio::test]
    async fn test_invalid_input() {
        let mut orchestrator = EquilibriumOrchestrator::new(
            2.0,
            None,
            VADScores::neutral(),
        ).unwrap();

        let result = orchestrator.orchestrate(
            vec![], // Empty tokens
            2.0,
            vec![], // Empty context
            false,
            VADScores::neutral(),
        ).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_reset() {
        let mut orchestrator = EquilibriumOrchestrator::new(
            2.0,
            None,
            VADScores::new(0.9, 0.9, 0.9).unwrap(),
        ).unwrap();

        orchestrator.reset();

        assert_eq!(orchestrator.state().equilibrium, 0.5);
        assert_eq!(orchestrator.state().sentiment_context, VADScores::neutral());
    }
}
