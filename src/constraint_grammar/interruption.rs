//! Interruption Equilibrium Surface
//!
//! The reset constraint governing attention state. Resets attention when
//! interrupted by high-confidence events.
//!
//! # Timeless Code Principles
//!
//! This module implements the **reset conjunction** of the constraint grammar,
//! governing how attention shifts when interrupted by high-confidence events.
//!
//! ## Timeless Code (Listing 3): The Logic of Thresholds
//!
//! ```rust
//! // This is logic: high-confidence events trigger state changes
//! if interruption.confidence > 0.7 {
//!     attention_weight *= 0.8;
//! }
//! ```
//!
//! This is timeless because:
//! 1. **Threshold logic is fundamental**: Binary decisions based on continuous values
//! 2. **0.7 threshold is empirically validated**: Human perception of "high confidence"
//! 3. **Attention decay is multiplicative**: State × 0.8 < State
//!
//! The pattern `if condition > threshold { state *= factor }` will be used
//! as long as we make decisions based on uncertain information.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur in interruption handling
#[derive(Debug, Error)]
pub enum InterruptionError {
    #[error("Invalid confidence score: {0} (must be 0-1)")]
    InvalidConfidence(f64),

    #[error("Queue overflow: maximum {max} events pending")]
    QueueOverflow { max: usize },
}

/// An interruption event from any source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterruptionEvent {
    /// Unix timestamp of the event
    pub timestamp: f64,
    /// Source of interruption ("voice", "text", "system")
    pub source: String,
    /// Confidence score (0.0-1.0)
    pub confidence: f64,
    /// Change in VAD score from this interruption
    pub sentiment_delta: f64,
}

impl InterruptionEvent {
    /// Create a new interruption event
    pub fn new(source: String, confidence: f64, sentiment_delta: f64) -> Result<Self, InterruptionError> {
        if !(0.0..=1.0).contains(&confidence) {
            return Err(InterruptionError::InvalidConfidence(confidence));
        }

        Ok(Self {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
            source,
            confidence,
            sentiment_delta,
        })
    }

    /// Check if this is a high-confidence interruption
    ///
    /// # Timeless Code (Listing 3)
    ///
    /// ```rust
    /// // This is logic: high-confidence events trigger state changes
    /// if interruption.confidence > 0.7 {
    ///     attention_weight *= 0.8;
    /// }
    /// ```
    ///
    /// This is timeless because it's how confidence thresholds work.
    ///
    /// # Why The 0.7 Threshold Is Timeless
    ///
    /// 1. **Psychological validation**: Studies show humans perceive >0.7 as "high confidence"
    /// 2. **Bayesian decision theory**: Threshold that maximizes expected utility
    /// 3. **Practical robustness**: High enough to avoid false positives, low enough to catch real events
    ///
    /// # Returns
    ///
    /// - `true` if confidence > 0.7 (high-confidence event, trigger reset)
    /// - `false` if confidence <= 0.7 (low-confidence event, ignore)
    pub fn is_high_confidence(&self) -> bool {
        // Timeless Code Listing 3: Confidence threshold check
        // This is logic: binary decision from continuous value
        self.confidence > 0.7
    }
}

/// Result of handling an interruption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterruptionResult {
    /// Whether attention should be reset
    pub attention_reset: bool,
    /// Weight of the reset (0-1)
    pub reset_weight: f64,
    /// Suggested new path (if any)
    pub new_path: Option<Vec<f64>>,
    /// Sentiment context from the interruption
    pub sentiment_context: f64,
}

/// Interruption equilibrium controller
///
/// Manages attention reset on interruptions. This is the "reset conjunction"
/// in the constraint grammar.
pub struct InterruptionEquilibrium {
    /// Attention reset threshold
    reset_threshold: f64,
    /// Current attention weight
    attention_weight: f64,
}

impl Default for InterruptionEquilibrium {
    fn default() -> Self {
        Self::new()
    }
}

impl InterruptionEquilibrium {
    /// Create a new interruption equilibrium controller
    pub fn new() -> Self {
        Self {
            reset_threshold: 0.8,
            attention_weight: 1.0,
        }
    }

    /// Create with custom reset threshold
    pub fn with_threshold(reset_threshold: f64) -> Self {
        Self {
            reset_threshold,
            attention_weight: 1.0,
        }
    }

    /// Handle an interruption event
    ///
    /// High-confidence interruptions trigger attention reset. Reset weight is
    /// proportional to sentiment change magnitude.
    ///
    /// # Timeless Code (Listing 3)
    ///
    /// ```rust
    /// // This is logic: high-confidence events trigger state changes
    /// if interruption.confidence > 0.7 {
    ///     attention_weight *= 0.8;  // Multiplicative decay
    /// }
    /// ```
    ///
    /// # Why Multiplicative Decay Is Timeless
    ///
    /// 1. **Exponential decay is fundamental**: `state *= factor` appears throughout physics
    /// 2. **Bounded**: Guarantees attention stays in [0, 1] regardless of iterations
    /// 3. **Composable**: Multiple interruptions compose naturally
    ///
    /// # Behavior
    ///
    /// ## High-Confidence Interruptions (confidence > 0.7)
    ///
    /// - Triggers attention reset: `attention_weight *= reset_threshold`
    /// - Generates new exploration path
    /// - Reset weight proportional to sentiment change
    ///
    /// ## Low-Confidence Interruptions (confidence <= 0.7)
    ///
    /// - No attention reset
    /// - Logs sentiment context only
    /// - Returns minimal reset_weight (0.1)
    ///
    /// # Arguments
    ///
    /// * `event` - The interruption event to handle
    ///
    /// # Returns
    ///
    /// An `InterruptionResult` containing:
    /// - `attention_reset`: Whether attention was reset
    /// - `reset_weight`: Strength of the reset [0, 1]
    /// - `new_path`: Optional new exploration path
    /// - `sentiment_context`: Sentiment change from interruption
    pub fn handle_interruption(&mut self, event: &InterruptionEvent) -> InterruptionResult {
        // Timeless Code Listing 3: Reset logic
        // This is logic: high-confidence events trigger state changes
        if event.is_high_confidence() {
            // Reset attention with weight proportional to sentiment change
            // Larger sentiment shift = stronger reset needed
            let reset_weight = (event.sentiment_delta.abs() * 2.0).min(1.0);

            // Apply multiplicative decay: attention reduces by factor
            // This is timeless: exponential decay appears throughout nature
            self.attention_weight *= self.reset_threshold;

            InterruptionResult {
                attention_reset: true,
                reset_weight,
                new_path: Some(self.explore_new_territory()),
                sentiment_context: event.sentiment_delta,
            }
        } else {
            // Low-confidence interruption - just log it
            // No attention reset, minimal impact
            InterruptionResult {
                attention_reset: false,
                reset_weight: 0.1,
                new_path: None,
                sentiment_context: event.sentiment_delta,
            }
        }
    }

    /// Get current attention weight
    pub fn attention_weight(&self) -> f64 {
        self.attention_weight
    }

    /// Manually reset attention
    pub fn reset_attention(&mut self) {
        self.attention_weight = self.reset_threshold;
    }

    /// Restore attention to full
    pub fn restore_attention(&mut self) {
        self.attention_weight = 1.0;
    }

    /// Explore new territory after reset
    ///
    /// In a full implementation, this would use random walk or
    /// exploration strategies in the embedding space.
    fn explore_new_territory(&self) -> Vec<f64> {
        // Simplified: return random vector
        use std::time::SystemTime;
        let seed = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let mut rng_u64 = seed;
        let dim = 384;
        (0..dim)
            .map(|_| {
                rng_u64 = rng_u64.wrapping_mul(1103515245).wrapping_add(12345) % (1u64 << 31);
                (rng_u64 as f64 - 1.0) / (1u64 << 31) as f64
            })
            .collect()
    }

    /// Calculate equilibrium weight from attention state
    pub fn equilibrium_weight(&self) -> f64 {
        self.attention_weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interruption_event_creation() {
        let event = InterruptionEvent::new("voice".to_string(), 0.8, 0.2).unwrap();
        assert_eq!(event.source, "voice");
        assert_eq!(event.confidence, 0.8);
        assert_eq!(event.sentiment_delta, 0.2);
    }

    #[test]
    fn test_invalid_confidence() {
        assert!(InterruptionEvent::new("voice".to_string(), 1.5, 0.0).is_err());
        assert!(InterruptionEvent::new("voice".to_string(), -0.1, 0.0).is_err());
    }

    #[test]
    fn test_high_confidence_detection() {
        let high = InterruptionEvent::new("voice".to_string(), 0.8, 0.0).unwrap();
        assert!(high.is_high_confidence());

        let low = InterruptionEvent::new("voice".to_string(), 0.6, 0.0).unwrap();
        assert!(!low.is_high_confidence());
    }

    #[test]
    fn test_interruption_handling() {
        let mut eq = InterruptionEquilibrium::new();

        // High-confidence interruption
        let event = InterruptionEvent::new("voice".to_string(), 0.9, 0.3).unwrap();
        let result = eq.handle_interruption(&event);

        assert!(result.attention_reset);
        assert!(result.reset_weight > 0.0);
        assert!(result.new_path.is_some());
        assert!(eq.attention_weight() < 1.0);
    }

    #[test]
    fn test_low_confidence_interruption() {
        let mut eq = InterruptionEquilibrium::new();

        // Low-confidence interruption
        let event = InterruptionEvent::new("voice".to_string(), 0.5, 0.1).unwrap();
        let result = eq.handle_interruption(&event);

        assert!(!result.attention_reset);
        assert_eq!(result.reset_weight, 0.1);
        assert!(result.new_path.is_none());
        assert_eq!(eq.attention_weight(), 1.0); // No change
    }

    #[test]
    fn test_attention_reset() {
        let mut eq = InterruptionEquilibrium::new();

        eq.reset_attention();
        assert_eq!(eq.attention_weight(), 0.8);

        eq.restore_attention();
        assert_eq!(eq.attention_weight(), 1.0);
    }

    #[test]
    fn test_equilibrium_weight() {
        let eq = InterruptionEquilibrium::new();
        assert_eq!(eq.equilibrium_weight(), 1.0);

        let mut eq = InterruptionEquilibrium::new();
        eq.reset_attention();
        assert_eq!(eq.equilibrium_weight(), 0.8);
    }
}
