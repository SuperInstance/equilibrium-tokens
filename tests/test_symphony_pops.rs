//! Test: Symphony Pops Conversation (Medium Equilibrium)
//!
//! Tests medium equilibrium scenario: steady rate, positive sentiment, rare interruptions

use equilibrium_tokens::EquilibriumOrchestrator;
use equilibrium_tokens::constraint_grammar::sentiment::VADScores;

#[tokio::test]
async fn test_symphony_pops_conversation() {
    let mut orchestrator = EquilibriumOrchestrator::new(
        1.5,
        None,
        VADScores::neutral(),
    ).unwrap();

    // Simulate: Steady 1.5 tokens/second, positive sentiment, rare interruptions
    let input_tokens = vec![
        "The".to_string(),
        "performance".to_string(),
        "was".to_string(),
        "excellent".to_string(),
        "today".to_string(),
    ];
    let rate = 1.5;
    let sentiment = VADScores::new(0.7, 0.4, 0.8).unwrap(); // Positive, moderate, confident
    let interruption = false;

    let result = orchestrator
        .orchestrate(
            input_tokens,
            rate,
            vec
![0.15; 384],
            interruption,
            sentiment,
        )
        .await
        .unwrap();

    // Assertions for medium equilibrium
    assert!(
        result.confidence > 0.4 && result.confidence < 0.9,
        "Expected medium confidence 0.4-0.9, got {}",
        result.confidence
    );
    assert!(
        (result.rate_target - 1.425).abs() < 0.001,
        "Expected rate target ~1.425 (1.5 * 0.95), got {}",
        result.rate_target
    );
    assert_eq!(
        result.attention_reset, false,
        "Should not reset attention without interruption"
    );

    println!("✅ Symphony pops test passed: confidence={:.3}", result.confidence);
}

#[tokio::test]
async fn test_symphony_to_jazz_transition() {
    let mut orchestrator = EquilibriumOrchestrator::new(
        1.5,
        None,
        VADScores::neutral(),
    ).unwrap();

    // Start with symphony pops (medium equilibrium)
    let result1 = orchestrator
        .orchestrate(
            vec
!["The".to_string(), "performance".to_string()],
            1.5,
            vec
![0.15; 384],
            false,
            VADScores::new(0.7, 0.4, 0.8).unwrap(),
        )
        .await
        .unwrap();

    // Transition to jazz band (low equilibrium)
    let result2 = orchestrator
        .orchestrate(
            vec
!["Hey".to_string(), "wait".to_string()],
            0.5,
            vec
![0.2; 384],
            true,
            VADScores::new(0.3, 0.8, 0.6).unwrap(),
        )
        .await
        .unwrap();

    // Confidence should drop significantly
    assert!(
        result1.confidence > result2.confidence,
        "Confidence should drop when transitioning from symphony to jazz"
    );

    println!("✅ Symphony to jazz transition test passed");
    println!("   Symphony confidence: {:.3}", result1.confidence);
    println!("   Jazz confidence: {:.3}", result2.confidence);
}
