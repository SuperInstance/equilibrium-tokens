//! # Equilibrium Tokens
//!
//! A constraint grammar for human-machine conversation navigation.
//!
//! This library implements a formal system of composable equilibrium surfaces
//! that govern rate, context, interruption, and sentiment in conversations.
//!
//! ## Architecture
//!
//! The model is frozen territory—unchangeable real estate. Equilibrium is the
//! navigation algorithm that chooses paths through this territory based on:
//! - Rate constraints (temporal)
//! - Context constraints (spatial)
//! - Interruption constraints (reset)
//! - Sentiment constraints (affective)
//!
//! ## Components
//!
//! - [`rate_equilibrium`]: Hardware-level token rate control
//! - [`context_equilibrium`]: Navigate conversation context basins
//! - [`interruption_equilibrium`]: Reset attention on interruptions
//! - [`sentiment_equilibrium`]: Weight paths by emotional valence
//! - [`orchestrator`]: Main constraint function orchestration

pub mod constraint_grammar;
pub mod token_organization;
pub mod equilibrium_orchestrator;

// Re-export key types for convenience
pub use equilibrium_orchestrator::EquilibriumOrchestrator;
pub use constraint_grammar::rate::RateEquilibrium;
pub use constraint_grammar::rate::RateConfig;
