//! Navigation Through Frozen Territory
//!
//! Implements path selection through the model's parameter space.

use crate::constraint_grammar::context::Tensor;
use thiserror::Error;

/// Errors that can occur during navigation
#[derive(Debug, Error)]
pub enum NavigationError {
    #[error("Invalid path dimensions: expected {expected}, got {actual}")]
    InvalidDimensions { expected: usize, actual: usize },

    #[error("No valid path found")]
    NoPathFound,
}

/// Navigation path through frozen territory
#[derive(Debug, Clone)]
pub struct NavigationPath {
    /// Path vectors
    pub vectors: Vec<Tensor>,
    /// Confidence score for this path
    pub confidence: f64,
    /// Path metadata
    pub metadata: serde_json::Value,
}

impl NavigationPath {
    /// Create a new navigation path
    pub fn new(vectors: Vec<Tensor>, confidence: f64) -> Self {
        Self {
            vectors,
            confidence,
            metadata: serde_json::json!({}),
        }
    }

    /// Get average vector along path
    pub fn average_vector(&self) -> Option<Vec<f64>> {
        if self.vectors.is_empty() {
            return None;
        }

        let dim = self.vectors[0].data().len();
        let mut sum = vec![0.0; dim];

        for vec in &self.vectors {
            for (i, &val) in vec.data().iter().enumerate() {
                sum[i] += val;
            }
        }

        let count = self.vectors.len() as f64;
        for val in sum.iter_mut() {
            *val /= count;
        }

        Some(sum)
    }

    /// Get path length
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if path is empty
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}

/// Navigator for path selection
pub struct Navigator {
    /// Exploration factor (0 = greedy, 1 = random)
    exploration_factor: f64,
}

impl Navigator {
    /// Create a new navigator
    pub fn new() -> Self {
        Self {
            exploration_factor: 0.1,
        }
    }

    /// Create with custom exploration factor
    pub fn with_exploration(exploration_factor: f64) -> Self {
        Self {
            exploration_factor: exploration_factor.max(0.0).min(1.0),
        }
    }

    /// Select path from candidates
    ///
    /// Uses epsilon-greedy selection: explore with probability
    /// equal to exploration_factor, otherwise select best path.
    pub fn select_path(&self, candidates: Vec<NavigationPath>) -> Result<NavigationPath, NavigationError> {
        if candidates.is_empty() {
            return Err(NavigationError::NoPathFound);
        }

        // Explore or exploit
        if self.should_explore() && candidates.len() > 1 {
            // Random selection (explore)
            let idx = (self.random() * candidates.len() as f64) as usize;
            return Ok(candidates[idx.min(candidates.len() - 1)].clone());
        }

        // Select best path (exploit)
        let best = candidates
            .into_iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
            .ok_or(NavigationError::NoPathFound)?;

        Ok(best)
    }

    /// Should explore (vs. exploit)?
    fn should_explore(&self) -> bool {
        self.random() < self.exploration_factor
    }

    /// Generate random number [0, 1]
    fn random(&self) -> f64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // Simple linear congruential generator
        let a = 1103515245u64;
        let c = 12345u64;
        let m = 1u64 << 31;
        ((a.wrapping_mul(seed).wrapping_add(c)) % m) as f64 / m as f64
    }
}

impl Default for Navigator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_navigation_path_creation() {
        let path = NavigationPath::new(vec![], 0.8);
        assert_eq!(path.confidence, 0.8);
        assert!(path.is_empty());
    }

    #[test]
    fn test_average_vector() {
        let tensors = vec![
            Tensor::new(vec![1.0, 2.0]),
            Tensor::new(vec![3.0, 4.0]),
        ];

        let path = NavigationPath::new(tensors, 0.8);
        let avg = path.average_vector().unwrap();

        assert_eq!(avg, vec![2.0, 3.0]);
    }

    #[test]
    fn test_navigator_creation() {
        let nav = Navigator::new();
        assert_eq!(nav.exploration_factor, 0.1);
    }

    #[test]
    fn test_select_path() {
        let nav = Navigator::with_exploration(0.0); // Pure exploitation

        let paths = vec![
            NavigationPath::new(vec![], 0.5),
            NavigationPath::new(vec![], 0.8),
            NavigationPath::new(vec![], 0.3),
        ];

        let selected = nav.select_path(paths).unwrap();
        assert_eq!(selected.confidence, 0.8); // Should select highest
    }

    #[test]
    fn test_empty_candidates() {
        let nav = Navigator::new();
        let result = nav.select_path(vec![]);
        assert!(result.is_err());
    }
}
