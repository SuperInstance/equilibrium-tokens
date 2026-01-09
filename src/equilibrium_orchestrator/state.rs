//! Orchestrator State Management
//!
//! Maintains the state of the equilibrium orchestration.

use serde::{Deserialize, Serialize};

/// State of the equilibrium orchestrator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorState {
    /// Current equilibrium score (0-1)
    pub equilibrium: f64,
    /// Current rate target in Hz
    pub rate_target: f64,
    /// Current attention state
    pub attention_reset: bool,
    /// Current sentiment context
    pub sentiment_context: crate::constraint_grammar::sentiment::VADScores,
    /// Current context vector
    pub context_vector: Option<Vec<f64>>,
}

impl Default for OrchestratorState {
    fn default() -> Self {
        Self {
            equilibrium: 0.5,
            rate_target: 2.0,
            attention_reset: false,
            sentiment_context: crate::constraint_grammar::sentiment::VADScores::neutral(),
            context_vector: None,
        }
    }
}

impl OrchestratorState {
    /// Create a new orchestrator state
    pub fn new() -> Self {
        Self::default()
    }

    /// Update equilibrium score
    pub fn update_equilibrium(&mut self, equilibrium: f64) {
        self.equilibrium = equilibrium.clamp(0.0, 1.0);
    }

    /// Update rate target
    pub fn update_rate_target(&mut self, rate: f64) {
        self.rate_target = rate.max(0.1);
    }

    /// Set attention reset flag
    pub fn set_attention_reset(&mut self, reset: bool) {
        self.attention_reset = reset;
    }

    /// Update sentiment context
    pub fn update_sentiment(&mut self, sentiment: crate::constraint_grammar::sentiment::VADScores) {
        self.sentiment_context = sentiment;
    }

    /// Update context vector
    pub fn update_context(&mut self, context: Vec<f64>) {
        self.context_vector = Some(context);
    }

    /// Reset to default state
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}
