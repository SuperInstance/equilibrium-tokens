//! Context Equilibrium Surface
//!
//! The spatial constraint governing basin navigation in the frozen territory.
//! Navigates conversation context using semantic similarity and sentiment weighting.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur in context equilibrium
#[derive(Debug, Error)]
pub enum ContextError {
    #[error("Invalid tensor dimensions: expected {expected}, got {actual}")]
    InvalidDimensions { expected: usize, actual: usize },

    #[error("Vector store not initialized")]
    VectorStoreNotInitialized,

    #[error("No similar contexts found")]
    NoSimilarContexts,

    #[error("Vector store error: {0}")]
    VectorStore(String),
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

impl Default for VADScores {
    fn default() -> Self {
        Self {
            valence: 0.0,
            arousal: 0.5,
            dominance: 0.7,
        }
    }
}

impl VADScores {
    /// Calculate equilibrium weight from VAD scores
    ///
    /// Maps valence from [-1, 1] to [0, 1] for equilibrium calculation.
    pub fn equilibrium_weight(&self) -> f64 {
        // Timeless Code Listing 4: VAD conversion
        (self.valence + 1.0) / 2.0
    }
}

/// A tensor/vector in the embedding space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    data: Vec<f64>,
    shape: Vec<usize>,
}

impl Tensor {
    /// Create a new tensor from a vector
    pub fn new(data: Vec<f64>) -> Self {
        let shape = vec![data.len()];
        Self { data, shape }
    }

    /// Get the tensor data
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Get tensor shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Calculate cosine similarity with another tensor
    ///
    /// # Timeless Code (Listing 2)
    ///
    /// ```rust
    /// // This is geometry: similarity is measured by cosine
    /// similarity = cosine_similarity(query, basin_center)
    /// ```
    ///
    /// This is timeless because it's how vector similarity works
    /// regardless of embedding dimension.
    pub fn cosine_similarity(&self, other: &Tensor) -> f64 {
        if self.data.len() != other.data.len() {
            return 0.0;
        }

        let dot_product: f64 = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum();

        let magnitude_a: f64 = self.data.iter().map(|x| x * x).sum::<f64>().sqrt();
        let magnitude_b: f64 = other.data.iter().map(|x| x * x).sum::<f64>().sqrt();

        if magnitude_a == 0.0 || magnitude_b == 0.0 {
            return 0.0;
        }

        dot_product / (magnitude_a * magnitude_b)
    }

    /// Interpolate with other tensors using weighted average
    pub fn interpolate_weighted(&self, others: &[Tensor], weights: &[f64]) -> Tensor {
        let dim = self.data.len();
        let mut result = vec![0.0; dim];

        // Add self with weight 1.0
        for i in 0..dim {
            result[i] = self.data[i];
        }

        // Add weighted contributions from others
        for (other, &weight) in others.iter().zip(weights.iter()) {
            for i in 0..dim.min(other.data.len()) {
                result[i] += other.data[i] * weight;
            }
        }

        Tensor::new(result)
    }

    /// Flatten tensor to 1D vector
    pub fn flatten(&self) -> Vec<f64> {
        self.data.clone()
    }
}

/// Result of context navigation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavigationResult {
    /// Suggested path through the territory
    pub path: Tensor,
    /// Confidence score (0-1)
    pub confidence: f64,
    /// Nearby context basins
    pub nearby_basins: Vec<ContextBasin>,
}

/// A context basin in the frozen territory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextBasin {
    /// Basin identifier
    pub id: String,
    /// Basin center vector
    pub vector: Tensor,
    /// Similarity score
    pub similarity: f64,
    /// Additional metadata
    pub metadata: serde_json::Value,
}

/// Vector store interface for RAG index
pub trait VectorStore: Send + Sync {
    /// Search for similar vectors
    fn search(&self, query: &Tensor, k: usize) -> Result<Vec<ContextBasin>, ContextError>;

    /// Get current context vector
    fn get_current(&self) -> Result<Tensor, ContextError>;
}

/// Error type for vector store issues
#[derive(Debug, Clone)]
pub struct VectorStoreError(pub String);

/// Context equilibrium controller
///
/// Manages navigation through conversation context basins weighted by sentiment.
/// This is the "spatial noun phrase" in the constraint grammar.
pub struct ContextEquilibrium<VS: VectorStore> {
    /// Current conversation context vector
    context_vec: Tensor,
    /// RAG index for semantic search
    rag_index: VS,
    /// Current sentiment scores
    sentiment: VADScores,
    /// Current equilibrium score
    equilibrium: f64,
}

impl<VS: VectorStore> ContextEquilibrium<VS> {
    /// Create a new context equilibrium controller
    pub fn new(context_vec: Tensor, rag_index: VS) -> Self {
        Self {
            context_vec,
            rag_index,
            sentiment: VADScores::default(),
            equilibrium: 0.5,
        }
    }

    /// Navigate through context basins
    ///
    /// Finds nearby basins in the frozen territory and navigates to
    /// emotionally relevant territory based on sentiment weights.
    pub fn navigate(&mut self, context: Tensor) -> Result<NavigationResult, ContextError> {
        // Find nearby basins in the frozen territory
        let nearby_basins = self.rag_index.search(&context, 5)?;

        if nearby_basins.is_empty() {
            return Err(ContextError::NoSimilarContexts);
        }

        // Calculate path through territory weighted by sentiment
        let sentiment_weight = self.sentiment.equilibrium_weight();

        // Navigate to emotionally relevant territory
        let weights: Vec<f64> = vec![sentiment_weight, 0.8, 0.6, 0.4, 0.2];
        let basin_tensors: Vec<Tensor> = nearby_basins.iter()
            .map(|b| b.vector.clone())
            .collect();

        let path = context.interpolate_weighted(&basin_tensors, &weights);

        // Calculate confidence
        let current = self.rag_index.get_current()?;
        let similarity = context.cosine_similarity(&current);
        let confidence = similarity * sentiment_weight;

        // Update state
        self.context_vec = context;
        self.equilibrium = confidence;

        Ok(NavigationResult {
            path,
            confidence,
            nearby_basins,
        })
    }

    /// Update sentiment scores
    pub fn update_sentiment(&mut self, sentiment: VADScores) {
        self.sentiment = sentiment;
    }

    /// Get current equilibrium score
    pub fn equilibrium(&self) -> f64 {
        self.equilibrium
    }

    /// Get current sentiment
    pub fn sentiment(&self) -> &VADScores {
        &self.sentiment
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockVectorStore {
        vectors: Vec<Tensor>,
    }

    impl VectorStore for MockVectorStore {
        fn search(&self, query: &Tensor, k: usize) -> Result<Vec<ContextBasin>, ContextError> {
            let mut basins = Vec::new();

            for (i, vec) in self.vectors.iter().enumerate() {
                let similarity = query.cosine_similarity(vec);
                if similarity > 0.5 {
                    basins.push(ContextBasin {
                        id: format!("basin_{}", i),
                        vector: vec.clone(),
                        similarity,
                        metadata: serde_json::json!({}),
                    });
                }
            }

            basins.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
            basins.truncate(k);
            Ok(basins)
        }

        fn get_current(&self) -> Result<Tensor, ContextError> {
            if self.vectors.is_empty() {
                Err(ContextError::VectorStoreNotInitialized)
            } else {
                Ok(self.vectors[0].clone())
            }
        }
    }

    #[test]
    fn test_tensor_creation() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(tensor.data(), &[1.0, 2.0, 3.0]);
        assert_eq!(tensor.shape(), &[3]);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0]);
        let c = Tensor::new(vec![-1.0, -2.0, -3.0]);

        // Identical vectors
        assert!((a.cosine_similarity(&b) - 1.0).abs() < 0.001);

        // Opposite vectors
        assert!((a.cosine_similarity(&c) + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_vad_scores() {
        let vad = VADScores {
            valence: 0.5,
            arousal: 0.7,
            dominance: 0.8,
        };

        // Valence 0.5 should map to equilibrium weight 0.75
        assert!((vad.equilibrium_weight() - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_context_navigate() {
        let context = Tensor::new(vec![1.0, 0.0, 0.0]);
        let store = MockVectorStore {
            vectors: vec![
                Tensor::new(vec![1.0, 0.1, 0.0]),
                Tensor::new(vec![0.9, 0.0, 0.1]),
            ],
        };

        let mut eq = ContextEquilibrium::new(context.clone(), store);
        let result = eq.navigate(context);

        assert!(result.is_ok());
        let nav = result.unwrap();
        assert!(nav.confidence > 0.0);
        assert!(!nav.nearby_basins.is_empty());
    }

    #[test]
    fn test_tensor_interpolation() {
        let a = Tensor::new(vec![1.0, 0.0]);
        let b = Tensor::new(vec![0.0, 1.0]);
        let c = Tensor::new(vec![0.5, 0.5]);

        let result = a.interpolate_weighted(&[b], &[0.5]);
        // Result should be closer to [1.0, 0.5] (a + 0.5*b)
        assert_eq!(result.data()[0], 1.0);
        assert_eq!(result.data()[1], 0.5);
    }
}
