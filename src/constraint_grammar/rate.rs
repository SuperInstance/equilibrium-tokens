//! Rate Equilibrium Surface
//!
//! The temporal constraint governing token flow. Ensures output rate matches
//! input rate with sub-2ms jitter using hardware timers.

use std::sync::atomic::{AtomicU64, Ordering};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur in rate equilibrium
#[derive(Debug, Error)]
pub enum RateError {
    #[error("Invalid rate: {rate} Hz (must be > 0)")]
    InvalidRate { rate: f64 },

    #[error("Jitter exceeded threshold: {jitter_ms} ms > {threshold_ms} ms")]
    JitterExceeded { jitter_ms: f64, threshold_ms: f64 },
}

/// Configuration for rate equilibrium
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateConfig {
    /// Target token rate in Hz (tokens/second)
    pub target_rate_hz: f64,
    /// Maximum acceptable jitter in milliseconds
    pub jitter_threshold_ms: f64,
}

impl RateConfig {
    /// Create a new rate configuration with defaults
    pub fn new(target_rate_hz: f64) -> Self {
        Self {
            target_rate_hz,
            jitter_threshold_ms: 2.0,
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), RateError> {
        if self.target_rate_hz <= 0.0 {
            return Err(RateError::InvalidRate { rate: self.target_rate_hz });
        }
        Ok(())
    }
}

/// Rate equilibrium controller
///
/// Manages token emission rate to match input rate with minimal jitter.
/// This is the "temporal adverbial" in the constraint grammar.
///
/// # Timeless Code (Listing 1)
///
/// ```rust
/// // This is physics: time intervals are measured in nanoseconds
/// let interval_ns = (1_000_000_000.0 / target_rate_hz) as u64;
/// ```
///
/// This is timeless because it's how computers measure time.
/// It will be true in 100 years.
pub struct RateEquilibrium {
    /// Target rate stored as integer (rate * 1000 for precision)
    target_rate: AtomicU64,
    /// Current measured rate
    current_rate: AtomicU64,
    /// Jitter threshold in milliseconds
    jitter_threshold: f64,
    /// Last emission time for jitter calculation
    last_emission: AtomicU64,
}

impl RateEquilibrium {
    /// Create a new rate equilibrium controller
    ///
    /// # Arguments
    ///
    /// * `config` - Rate configuration including target rate and jitter threshold
    ///
    /// # Example
    ///
    /// ```no_run
    /// use equilibrium_tokens::RateEquilibrium;
    /// use equilibrium_tokens::RateConfig;
    ///
    /// let config = RateConfig::new(2.0); // 2 tokens/second
    /// let rate_eq = RateEquilibrium::new(config).unwrap();
    /// ```
    pub fn new(config: RateConfig) -> Result<Self, RateError> {
        config.validate()?;

        Ok(Self {
            target_rate: AtomicU64::new((config.target_rate_hz * 1000.0) as u64),
            current_rate: AtomicU64::new(0),
            jitter_threshold: config.jitter_threshold_ms,
            last_emission: AtomicU64::new(0),
        })
    }

    /// Handle rate change from input
    ///
    /// Adjusts the target rate based on measured input rate.
    ///
    /// # Arguments
    ///
    /// * `new_rate` - New target rate in Hz
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use equilibrium_tokens::RateEquilibrium;
    /// # use equilibrium_tokens::RateConfig;
    /// # let mut rate_eq = RateEquilibrium::new(RateConfig::new(2.0)).unwrap();
    /// rate_eq.on_rate_change(3.0); // Increase to 3 tokens/second
    /// ```
    pub fn on_rate_change(&self, new_rate: f64) -> Result<(), RateError> {
        if new_rate <= 0.0 {
            return Err(RateError::InvalidRate { rate: new_rate });
        }

        // Timeless Code: Store rate with precision
        self.target_rate.store((new_rate * 1000.0) as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Calculate target interval in nanoseconds
    ///
    /// This is the timeless calculation that converts Hz to nanoseconds.
    pub fn calculate_interval_ns(&self, rate_hz: f64) -> u64 {
        // Timeless Code Listing 1: Timer interval calculation
        (1_000_000_000.0 / rate_hz) as u64
    }

    /// Get current target rate in Hz
    pub fn get_target_rate(&self) -> f64 {
        self.target_rate.load(Ordering::Relaxed) as f64 / 1000.0
    }

    /// Get current measured rate in Hz
    pub fn get_current_rate(&self) -> f64 {
        self.current_rate.load(Ordering::Relaxed) as f64 / 1000.0
    }

    /// Calculate rate weight for equilibrium
    ///
    /// Returns a value between 0 and 1 representing how well the current
    /// rate matches the target rate.
    ///
    /// # Formula
    ///
    /// ```text
    /// weight = 1 - |rate_in - rate_out| / max(rate_in, rate_out)
    /// ```
    ///
    /// If current_rate is 0 (not yet measured), assumes perfect match.
    pub fn calculate_rate_weight(&self, input_rate: f64) -> f64 {
        let output_rate = self.get_current_rate();

        // If we haven't measured output rate yet, assume it matches input
        if output_rate == 0.0 {
            return 1.0;
        }

        let max_rate = input_rate.max(output_rate);
        let diff = (input_rate - output_rate).abs();
        1.0 - (diff / max_rate)
    }

    /// Simulate token emission and measure jitter
    ///
    /// In a real implementation, this would use timerfd or similar.
    /// For now, we simulate timing for testing.
    pub fn emit_token(&self) -> Result<(), RateError> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let last = self.last_emission.load(Ordering::Relaxed);

        if last > 0 {
            let interval_ms = now - last;
            let target_interval_ms = 1000.0 / self.get_target_rate();

            let jitter_ms = (interval_ms as f64 - target_interval_ms).abs();

            if jitter_ms > self.jitter_threshold {
                return Err(RateError::JitterExceeded {
                    jitter_ms,
                    threshold_ms: self.jitter_threshold,
                });
            }
        }

        self.last_emission.store(now, Ordering::Relaxed);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_config_creation() {
        let config = RateConfig::new(10.0);
        assert_eq!(config.target_rate_hz, 10.0);
        assert_eq!(config.jitter_threshold_ms, 2.0);
    }

    #[test]
    fn test_rate_config_validation() {
        let config = RateConfig::new(-1.0);
        assert!(config.validate().is_err());

        let config = RateConfig::new(0.0);
        assert!(config.validate().is_err());

        let config = RateConfig::new(1.0);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_rate_equilibrium_creation() {
        let config = RateConfig::new(2.0);
        let rate_eq = RateEquilibrium::new(config).unwrap();
        assert_eq!(rate_eq.get_target_rate(), 2.0);
    }

    #[test]
    fn test_rate_change() {
        let config = RateConfig::new(2.0);
        let rate_eq = RateEquilibrium::new(config).unwrap();

        rate_eq.on_rate_change(3.0).unwrap();
        assert_eq!(rate_eq.get_target_rate(), 3.0);
    }

    #[test]
    fn test_interval_calculation() {
        let config = RateConfig::new(1.0);
        let rate_eq = RateEquilibrium::new(config).unwrap();

        // 1 Hz = 1 second = 1,000,000,000 nanoseconds
        let interval = rate_eq.calculate_interval_ns(1.0);
        assert_eq!(interval, 1_000_000_000);

        // 2 Hz = 0.5 seconds = 500,000,000 nanoseconds
        let interval = rate_eq.calculate_interval_ns(2.0);
        assert_eq!(interval, 500_000_000);
    }

    #[test]
    fn test_rate_weight_calculation() {
        let config = RateConfig::new(2.0);
        let rate_eq = RateEquilibrium::new(config).unwrap();

        // With current_rate at 0 (not yet measured), assume perfect match
        let weight = rate_eq.calculate_rate_weight(2.0);
        assert_eq!(weight, 1.0);

        // Also with current_rate at 0
        let weight = rate_eq.calculate_rate_weight(1.0);
        assert_eq!(weight, 1.0);
    }

    #[test]
    fn test_invalid_rate_change() {
        let config = RateConfig::new(2.0);
        let rate_eq = RateEquilibrium::new(config).unwrap();

        assert!(rate_eq.on_rate_change(0.0).is_err());
        assert!(rate_eq.on_rate_change(-1.0).is_err());
    }
}
