//! Test: Fishing Boat Conversation (High Equilibrium)
//!
//! Tests high equilibrium scenario: steady rate, calm sentiment, no interruptions

use equilibrium_tokens::EquilibriumOrchestrator;
use equilibrium_tokens::constraint_grammar::sentiment::VADScores;

#[tokio::test]
async fn test_fishing_boat_conversation() {
    let mut orchestrator = EquilibriumOrchestrator::new(
        2.0,
        None,
        VADScores::neutral(),
    ).unwrap();

    // Simulate: Steady 2 tokens/second, calm sentiment, no interruptions
    let input_tokens = vec
![
        "The".to_string(),
        "water".to_string(),
        "is".to_string(),
        "calm".to_string(),
        "today".to_string(),
    ];
    let rate = 2.0;
    let sentiment = VADScores::new(0.8, 0.3, 0.7).unwrap(); // Positive, calm, dominant
    let interruption = false;

    let result = orchestrator
        .orchestrate(
            input_tokens,
            rate,
            vec
![0.1; 384],
            interruption,
            sentiment,
        )
        .await
        .unwrap();

    // Assertions for high equilibrium
    assert!(
        result.confidence > 0.3,
        "Expected good confidence >0.3 for steady, positive input, got {}",
        result.confidence
    );
    assert_eq!(
        result.rate_target, 1.9,
        "Expected rate target 1.9 (2.0 * 0.95), got {}",
        result.rate_target
    );
    assert_eq!(
        result.attention_reset, false,
        "Should not reset attention for steady input"
    );

    println!("✅ Fishing boat test passed: confidence={:.3}", result.confidence);
}

#[tokio::test]
async fn test_fishing_boat_multiple_turns() {
    let mut orchestrator = EquilibriumOrchestrator::new(
        2.0,
        None,
        VADScores::neutral(),
    ).unwrap();

    // Simulate multiple steady conversation turns
    let turns = vec![
        (vec
!["The".to_string(), "water".to_string()], 2.0, VADScores::new(0.8, 0.3, 0.7).unwrap()),
        (vec
!["is".to_string(), "calm".to_string()], 2.0, VADScores::new(0.8, 0.3, 0.7).unwrap()),
        (vec
!["today".to_string()], 2.0, VADScores::new(0.8, 0.3, 0.7).unwrap()),
    ];

    for (tokens, rate, sentiment) in turns {
        let result = orchestrator
            .orchestrate(tokens, rate, vec
![0.1; 384], false, sentiment)
            .await
            .unwrap();

        assert!(
            result.confidence > 0.3,
            "Confidence should remain good across turns"
        );
    }

    println!("✅ Multiple turns fishing boat test passed");
}
