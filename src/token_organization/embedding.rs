//! Token Embedding Generation
//!
//! Converts tokens to vector embeddings for semantic processing.

use thiserror::Error;

/// Errors that can occur in embedding generation
#[derive(Debug, Error)]
pub enum EmbeddingError {
    #[error("Empty input")]
    EmptyInput,

    #[error("Token too long: {len} chars (max {max})")]
    TokenTooLong { len: usize, max: usize },

    #[error("Embedding model not loaded")]
    ModelNotLoaded,
}

/// Token embedder
///
/// Converts tokens/text to vector embeddings. In a full implementation,
/// this would use a proper embedding model like sentence-transformers.
pub struct TokenEmbedder {
    /// Embedding dimension
    dimension: usize,
}

impl TokenEmbedder {
    /// Create a new token embedder
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Create default embedder (384-dim, sentence-transformers style)
    pub fn with_default_dimension() -> Self {
        Self::new(384)
    }

    /// Embed a single token
    ///
    /// In a full implementation, this would use an actual embedding model.
    /// For now, we use a simplified hash-based approach.
    pub fn embed_token(&self, token: &str) -> Result<Vec<f64>, EmbeddingError> {
        if token.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        // Simplified: use hash to generate deterministic embeddings
        let mut embedding = Vec::with_capacity(self.dimension);
        let hash = Self::hash_string(token);

        for i in 0..self.dimension {
            // Generate deterministic values from hash
            let value = ((hash >> (i % 64)) & 0xFF) as f64 / 255.0;
            embedding.push(value);
        }

        Ok(embedding)
    }

    /// Embed multiple tokens and average
    pub fn embed_tokens(&self, tokens: &[String]) -> Result<Vec<f64>, EmbeddingError> {
        if tokens.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        let mut sum = vec![0.0; self.dimension];

        for token in tokens {
            let emb = self.embed_token(token)?;
            for (i, &val) in emb.iter().enumerate() {
                sum[i] += val;
            }
        }

        // Average
        let count = tokens.len() as f64;
        for val in sum.iter_mut() {
            *val /= count;
        }

        Ok(sum)
    }

    /// Hash string to u64
    fn hash_string(s: &str) -> u64 {
        let mut hash: u64 = 5381;
        for byte in s.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
        }
        hash
    }

    /// Get embedding dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

impl Default for TokenEmbedder {
    fn default() -> Self {
        Self::with_default_dimension()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedder_creation() {
        let embedder = TokenEmbedder::new(384);
        assert_eq!(embedder.dimension(), 384);
    }

    #[test]
    fn test_embed_token() {
        let embedder = TokenEmbedder::default();
        let emb = embedder.embed_token("hello").unwrap();

        assert_eq!(emb.len(), 384);
    }

    #[test]
    fn test_empty_token() {
        let embedder = TokenEmbedder::default();
        let result = embedder.embed_token("");
        assert!(result.is_err());
    }

    #[test]
    fn test_embed_tokens() {
        let embedder = TokenEmbedder::default();
        let tokens = vec
!["hello".to_string(), "world".to_string()];

        let emb = embedder.embed_tokens(&tokens).unwrap();
        assert_eq!(emb.len(), 384);
    }

    #[test]
    fn test_deterministic_embedding() {
        let embedder = TokenEmbedder::default();

        let emb1 = embedder.embed_token("hello").unwrap();
        let emb2 = embedder.embed_token("hello").unwrap();

        assert_eq!(emb1, emb2);
    }

    #[test]
    fn test_different_embeddings() {
        let embedder = TokenEmbedder::default();

        let emb1 = embedder.embed_token("hello").unwrap();
        let emb2 = embedder.embed_token("world").unwrap();

        // Should be different (with very high probability)
        assert_ne!(emb1, emb2);
    }
}
