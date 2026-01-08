//! Test: Jazz Band Conversation (Low Equilibrium)
//!
//! Tests low equilibrium scenario: variable rate, playful sentiment, frequent interruptions

use equilibrium_tokens::EquilibriumOrchestrator;
use equilibrium_tokens::constraint_grammar::sentiment::VADScores;

#[tokio::test]
async fn test_jazz_band_conversation() {
    let mut orchestrator = EquilibriumOrchestrator::new(
        2.0,
        None,
        VADScores::neutral(),
    ).unwrap();

    // Simulate: Variable 0.5 tokens/second, playful sentiment, frequent interruptions
    let input_tokens = vec
![
        "Hey".to_string(),
        "wait".to_string(),
        "listen".to_string(),
        "no".to_string(),
        "actually".to_string(),
    ];
    let rate = 0.5;
    let sentiment = VADScores::new(0.3, 0.8, 0.6).unwrap(); // Playful, energetic, confident
    let interruption = true;

    let result = orchestrator
        .orchestrate(
            input_tokens,
            rate,
            vec
![0.2; 384],
            interruption,
            sentiment,
        )
        .await
        .unwrap();

    // Assertions for low equilibrium
    assert!(
        result.confidence < 0.7,
        "Expected reduced confidence <0.7 for interrupted, variable input, got {}",
        result.confidence
    );
    assert_eq!(
        result.rate_target, 0.475,
        "Expected rate target 0.475 (0.5 * 0.95), got {}",
        result.rate_target
    );
    assert_eq!(
        result.attention_reset, true,
        "Should reset attention for interruptions"
    );

    println!("✅ Jazz band test passed: confidence={:.3}", result.confidence);
}

#[tokio::test]
async fn test_jazz_band_syncopated_rhythm() {
    let mut orchestrator = EquilibriumOrchestrator::new(
        2.0,
        None,
        VADScores::neutral(),
    ).unwrap();

    // Simulate syncopated rhythm (variable rates)
    let turns = vec![
        (vec
!["Hey".to_string()], 4.0, true),
        (vec
!["wait".to_string()], 0.5, true),
        (vec
!["listen".to_string()], 3.0, true),
        (vec
!["no".to_string()], 0.5, false),
        (vec
!["actually".to_string()], 2.0, true),
    ];

    let mut total_confidence = 0.0;

    for (tokens, rate, interruption) in &turns {
        let sentiment = VADScores::new(0.3, 0.8, 0.6).unwrap();
        let result = orchestrator
            .orchestrate(tokens.clone(), *rate, vec
![0.2; 384], *interruption, sentiment)
            .await
            .unwrap();

        total_confidence += result.confidence;
    }

    let avg_confidence = total_confidence / turns.len() as f64;

    // Average confidence should be lower due to syncopation
    assert!(
        avg_confidence < 0.7,
        "Average confidence should be lower for syncopated rhythm, got {}",
        avg_confidence
    );

    println!("✅ Syncopated rhythm test passed: avg confidence={:.3}", avg_confidence);
}
