//! Common test utilities and fixtures for secret-manager tests
//!
//! This module provides reusable test infrastructure including:
//! - Test setup/teardown
//! - Mock objects
//! - Test fixtures
//! - Custom assertions
//! - Test helpers

pub mod fixtures;
pub mod helpers;
pub mod mock;

use std::sync::{Arc, RwLock};
use tempfile::TempDir;

/// Test manager with automatic cleanup
pub struct TestManager {
    pub manager: Arc<crate::SecretManager>,
    pub temp_dir: Option<TempDir>,
}

impl TestManager {
    /// Create a new test manager with in-memory storage
    pub fn new() -> Self {
        let manager = crate::SecretManager::new_in_memory();
        Self {
            manager: Arc::new(manager),
            temp_dir: None,
        }
    }

    /// Create a new test manager with temporary file storage
    pub fn with_file_storage() -> Self {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let storage_path = temp_dir.path().join("secrets.db");
        let manager = crate::SecretManager::new_with_file_storage(&storage_path)
            .expect("Failed to create file storage");

        Self {
            manager: Arc::new(manager),
            temp_dir: Some(temp_dir),
        }
    }

    /// Get the underlying manager
    pub fn manager(&self) -> &Arc<crate::SecretManager> {
        &self.manager
    }
}

impl Drop for TestManager {
    fn drop(&mut self) {
        // TempDir will be automatically cleaned up when dropped
    }
}

/// Setup function for tests
pub fn setup() -> TestManager {
    TestManager::new()
}

/// Setup function for file-based storage tests
pub fn setup_with_storage() -> TestManager {
    TestManager::with_file_storage()
}

/// Cleanup function (automatically called by TestManager drop)
pub fn teardown(_manager: TestManager) {
    // Automatic cleanup handled by Drop implementation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_test_manager_creation() {
        let manager = setup();
        assert!(Arc::strong_count(&manager.manager) >= 1);
    }

    #[test]
    fn test_test_manager_with_storage() {
        let manager = setup_with_storage();
        assert!(manager.temp_dir.is_some());
    }
}
