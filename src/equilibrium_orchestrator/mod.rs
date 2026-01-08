//! Equilibrium Orchestrator
//!
//! Main constraint function orchestration. Composes all equilibrium
//! surfaces to navigate conversation through frozen territory.

pub mod orchestrator;
pub mod navigation;
pub mod state;

pub use orchestrator::{EquilibriumOrchestrator, EquilibriumResult};
pub use state::OrchestratorState;
