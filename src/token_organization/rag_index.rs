//! RAG Index for Semantic Search
//!
//! Provides Retrieval-Augmented Generation indexing for conversation context.

use crate::constraint_grammar::context::{Tensor, ContextBasin};
use crate::token_organization::vector_store::{VectorStoreImpl, VectorStoreError};
use thiserror::Error;

/// Errors specific to RAG index
#[derive(Debug, Error)]
pub enum RAGIndexError {
    #[error("Vector store error: {0}")]
    VectorStoreError(#[from] VectorStoreError),

    #[error("Index not initialized")]
    NotInitialized,
}

/// RAG index for conversation context
///
/// Manages semantic indexing of conversation history for context retrieval.
pub struct RAGIndex {
    /// Underlying vector store
    store: VectorStoreImpl,
}

impl RAGIndex {
    /// Create a new RAG index
    ///
    /// # Arguments
    ///
    /// * `dimensions` - Embedding dimension for vectors in the index
    pub fn new(dimensions: usize) -> Self {
        Self {
            store: VectorStoreImpl::new(dimensions),
        }
    }

    /// Add a document to the index
    pub fn add_document(
        &self,
        id: String,
        embedding: Vec<f64>,
        metadata: serde_json::Value,
    ) -> Result<(), RAGIndexError> {
        self.store
            .add_vector(id, embedding, metadata)?;
        Ok(())
    }

    /// Search for relevant context
    ///
    /// Returns top-k relevant documents by semantic similarity.
    pub fn search(&self, query_embedding: &[f64], k: usize) -> Result<Vec<ContextBasin>, RAGIndexError> {
        let results = self.store.search(query_embedding, k)?;

        Ok(results
            .into_iter()
            .map(|r| ContextBasin {
                id: r.id,
                vector: Tensor::new(r.vector),
                similarity: r.similarity,
                metadata: r.metadata,
            })
            .collect())
    }

    /// Get current context vector
    pub fn get_current(&self) -> Result<Tensor, RAGIndexError> {
        let vec = self.store.get_current()?;
        Ok(Tensor::new(vec))
    }

    /// Get index size
    pub fn size(&self) -> usize {
        self.store.size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rag_index_creation() {
        let index = RAGIndex::new(384);
        assert_eq!(index.size(), 0);
    }

    #[test]
    fn test_add_document() {
        let index = RAGIndex::new(3);

        index
            .add_document(
                "doc1".to_string(),
                vec![1.0, 0.0, 0.0],
                serde_json::json!({"text": "hello"}),
            )
            .unwrap();

        assert_eq!(index.size(), 1);
    }

    #[test]
    fn test_search() {
        let index = RAGIndex::new(3);

        index
            .add_document(
                "doc1".to_string(),
                vec![1.0, 0.0, 0.0],
                serde_json::json!({}),
            )
            .unwrap();

        index
            .add_document(
                "doc2".to_string(),
                vec![0.0, 1.0, 0.0],
                serde_json::json!({}),
            )
            .unwrap();

        let results = index.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "doc1");
    }
}
