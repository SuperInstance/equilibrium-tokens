//! Sentiment Equilibrium Surface
//!
//! The affective constraint governing path weighting. Weights navigation
//! paths by emotional valence using the VAD model.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur in sentiment processing
#[derive(Debug, Error)]
pub enum SentimentError {
    #[error("Invalid VAD score: {score} for dimension '{dim}' (must be in valid range)")]
    InvalidScore { score: f64, dim: String },

    #[error("Insufficient history: need at least {min} samples, got {actual}")]
    InsufficientHistory { min: usize, actual: usize },
}

/// VAD (Valence, Arousal, Dominance) sentiment scores
///
/// # Timeless Code (Listing 4)
///
/// ```rust
/// // This is affective science: valence is [-1, 1]
/// equilibrium_weight = (vad.valence + 1.0) / 2.0;
/// ```
///
/// This is timeless because Russell's circumplex model of affect
/// is empirically validated across cultures.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VADScores {
    /// Valence: -1.0 (negative) to 1.0 (positive)
    pub valence: f64,
    /// Arousal: 0.0 (calm) to 1.0 (energetic)
    pub arousal: f64,
    /// Dominance: 0.0 (submissive) to 1.0 (dominant)
    pub dominance: f64,
}

impl VADScores {
    /// Create new VAD scores with validation
    pub fn new(valence: f64, arousal: f64, dominance: f64) -> Result<Self, SentimentError> {
        if !(-1.0..=1.0).contains(&valence) {
            return Err(SentimentError::InvalidScore {
                score: valence,
                dim: "valence".to_string(),
            });
        }

        if !(0.0..=1.0).contains(&arousal) {
            return Err(SentimentError::InvalidScore {
                score: arousal,
                dim: "arousal".to_string(),
            });
        }

        if !(0.0..=1.0).contains(&dominance) {
            return Err(SentimentError::InvalidScore {
                score: dominance,
                dim: "dominance".to_string(),
            });
        }

        Ok(Self {
            valence,
            arousal,
            dominance,
        })
    }

    /// Default neutral sentiment
    pub fn neutral() -> Self {
        Self {
            valence: 0.0,
            arousal: 0.5,
            dominance: 0.7,
        }
    }

    /// Calculate equilibrium weight from VAD scores
    ///
    /// Maps valence from [-1, 1] to [0, 1] for equilibrium calculation.
    pub fn equilibrium_weight(&self) -> f64 {
        // Timeless Code Listing 4: VAD conversion
        (self.valence + 1.0) / 2.0
    }

    /// Calculate combined path weight from all VAD dimensions
    pub fn path_weight(&self) -> f64 {
        let valence_weight = self.equilibrium_weight();
        valence_weight * self.arousal * self.dominance
    }
}

impl Default for VADScores {
    fn default() -> Self {
        Self::neutral()
    }
}

/// Result of sentiment conduction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConductorResult {
    /// Weight for path selection (0-1)
    pub path_weight: f64,
    /// Suggested path based on emotional context
    pub suggested_path: Vec<f64>,
    /// Emotional context used for decision
    pub emotional_context: VADScores,
}

/// Sentiment equilibrium controller
///
/// Manages path weighting by emotional valence using VAD model.
/// This is the "affective adjective" in the constraint grammar.
pub struct SentimentEquilibrium {
    /// History of VAD scores
    vad_history: Vec<VADScores>,
    /// Rolling window size
    window_size: usize,
    /// Current equilibrium weight
    equilibrium_weight: f64,
}

impl Default for SentimentEquilibrium {
    fn default() -> Self {
        Self::new()
    }
}

impl SentimentEquilibrium {
    /// Create a new sentiment equilibrium controller
    pub fn new() -> Self {
        Self {
            vad_history: Vec::new(),
            window_size: 100,
            equilibrium_weight: 0.5,
        }
    }

    /// Create with custom window size
    pub fn with_window_size(window_size: usize) -> Self {
        Self {
            vad_history: Vec::new(),
            window_size,
            equilibrium_weight: 0.5,
        }
    }

    /// Update sentiment with new VAD scores
    ///
    /// Maintains a rolling window of sentiment history and calculates
    /// the average equilibrium weight.
    pub fn update_sentiment(&mut self, vad: VADScores) -> Result<(), SentimentError> {
        // Add to history
        self.vad_history.push(vad.clone());

        // Maintain window size
        if self.vad_history.len() > self.window_size {
            self.vad_history.remove(0);
        }

        // Calculate average VAD scores
        let count = self.vad_history.len();
        let avg_valence: f64 = self.vad_history.iter().map(|v| v.valence).sum::<f64>() / count as f64;
        let avg_arousal: f64 = self.vad_history.iter().map(|v| v.arousal).sum::<f64>() / count as f64;
        let avg_dominance: f64 = self.vad_history.iter().map(|v| v.dominance).sum::<f64>() / count as f64;

        // Update equilibrium weight
        let avg_vad = VADScores::new(avg_valence, avg_arousal, avg_dominance)?;
        self.equilibrium_weight = avg_vad.equilibrium_weight();

        Ok(())
    }

    /// Conduct sentiment analysis and generate path suggestion
    ///
    /// This is the "conductor's decision" - navigate to emotionally
    /// relevant territory based on current sentiment.
    pub fn conduct(&self, vad: &VADScores) -> ConductorResult {
        // Timeless Code: VAD weight calculation
        let valence_weight = vad.equilibrium_weight();
        let arousal_weight = vad.arousal;
        let dominance_weight = vad.dominance;

        // Combined weight
        let path_weight = valence_weight * arousal_weight * dominance_weight;

        // Select path based on emotional valence
        let suggested_path = self.select_path(vad);

        ConductorResult {
            path_weight,
            suggested_path,
            emotional_context: vad.clone(),
        }
    }

    /// Get current equilibrium weight
    pub fn equilibrium_weight(&self) -> f64 {
        self.equilibrium_weight
    }

    /// Get sentiment history size
    pub fn history_size(&self) -> usize {
        self.vad_history.len()
    }

    /// Select a path based on emotional valence
    ///
    /// In a full implementation, this would navigate to emotionally
    /// relevant territory in the embedding space.
    fn select_path(&self, vad: &VADScores) -> Vec<f64> {
        // Simplified: create path weighted by valence
        let dim = 384;
        let valence = vad.equilibrium_weight();

        (0..dim)
            .map(|i| {
                // Create pattern based on valence
                let phase = (i as f64 / dim as f64) * std::f64::consts::PI * 2.0;
                valence * phase.sin()
            })
            .collect()
    }

    /// Get average sentiment over history
    pub fn average_sentiment(&self) -> Result<VADScores, SentimentError> {
        if self.vad_history.is_empty() {
            return Err(SentimentError::InsufficientHistory {
                min: 1,
                actual: 0,
            });
        }

        let count = self.vad_history.len();
        let avg_valence: f64 = self.vad_history.iter().map(|v| v.valence).sum::<f64>() / count as f64;
        let avg_arousal: f64 = self.vad_history.iter().map(|v| v.arousal).sum::<f64>() / count as f64;
        let avg_dominance: f64 = self.vad_history.iter().map(|v| v.dominance).sum::<f64>() / count as f64;

        VADScores::new(avg_valence, avg_arousal, avg_dominance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vad_creation() {
        let vad = VADScores::new(0.5, 0.7, 0.8).unwrap();
        assert_eq!(vad.valence, 0.5);
        assert_eq!(vad.arousal, 0.7);
        assert_eq!(vad.dominance, 0.8);
    }

    #[test]
    fn test_invalid_vad() {
        // Invalid valence (must be -1 to 1)
        assert!(VADScores::new(1.5, 0.5, 0.5).is_err());

        // Invalid arousal (must be 0 to 1)
        assert!(VADScores::new(0.5, 1.5, 0.5).is_err());

        // Invalid dominance (must be 0 to 1)
        assert!(VADScores::new(0.5, 0.5, -0.1).is_err());
    }

    #[test]
    fn test_equilibrium_weight() {
        let vad = VADScores::new(0.0, 0.5, 0.5).unwrap();
        // Valence 0 should map to weight 0.5
        assert!((vad.equilibrium_weight() - 0.5).abs() < 0.01);

        let vad = VADScores::new(1.0, 0.5, 0.5).unwrap();
        // Valence 1 should map to weight 1.0
        assert!((vad.equilibrium_weight() - 1.0).abs() < 0.01);

        let vad = VADScores::new(-1.0, 0.5, 0.5).unwrap();
        // Valence -1 should map to weight 0.0
        assert!((vad.equilibrium_weight() - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_path_weight() {
        let vad = VADScores::new(0.5, 0.8, 0.9).unwrap();
        let weight = vad.path_weight();
        // equilibrium_weight() = (0.5 + 1.0) / 2.0 = 0.75
        // path_weight() = 0.75 * 0.8 * 0.9 = 0.54
        assert!((weight - 0.54).abs() < 0.01);
    }

    #[test]
    fn test_sentiment_update() {
        let mut eq = SentimentEquilibrium::new();
        let vad = VADScores::new(0.5, 0.7, 0.8).unwrap();

        eq.update_sentiment(vad).unwrap();
        assert_eq!(eq.history_size(), 1);
    }

    #[test]
    fn test_window_size() {
        let mut eq = SentimentEquilibrium::with_window_size(3);

        for _ in 0..5 {
            let vad = VADScores::new(0.5, 0.7, 0.8).unwrap();
            eq.update_sentiment(vad).unwrap();
        }

        // Should only keep last 3
        assert_eq!(eq.history_size(), 3);
    }

    #[test]
    fn test_conduct() {
        let eq = SentimentEquilibrium::new();
        let vad = VADScores::new(0.5, 0.7, 0.8).unwrap();

        let result = eq.conduct(&vad);
        assert!(result.path_weight > 0.0);
        assert!(!result.suggested_path.is_empty());
        assert_eq!(result.emotional_context, vad);
    }

    #[test]
    fn test_average_sentiment() {
        let mut eq = SentimentEquilibrium::new();

        let vad1 = VADScores::new(0.0, 0.5, 0.5).unwrap();
        let vad2 = VADScores::new(1.0, 0.5, 0.5).unwrap();

        eq.update_sentiment(vad1).unwrap();
        eq.update_sentiment(vad2).unwrap();

        let avg = eq.average_sentiment().unwrap();
        assert!((avg.valence - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_empty_history() {
        let eq = SentimentEquilibrium::new();
        assert!(eq.average_sentiment().is_err());
    }
}
