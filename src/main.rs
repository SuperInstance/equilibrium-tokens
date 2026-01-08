use equilibrium_tokens::EquilibriumOrchestrator;
use equilibrium_tokens::constraint_grammar::sentiment::VADScores;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Equilibrium Tokens Daemon v0.1.0");
    println!("Constraint Grammar for Human-Machine Conversation Navigation");
    println!();

    // Create orchestrator
    let mut orchestrator = EquilibriumOrchestrator::new(
        2.0, // 2 tokens/second
        None, // No RAG index for this demo
        VADScores::neutral(),
    )?;

    println!("✅ Orchestrator initialized successfully");
    println!("📊 Rate: 2.0 Hz");
    println!("📊 Sentiment: VAD=[0.0, 0.5, 0.7] (neutral)");
    println!();
    println!("Ready for conversation navigation...");
    println!();

    // Run test conversation
    println!("--- Test Conversation 1: High Equilibrium ---");
    let result = orchestrator.orchestrate(
        vec
!["The".to_string(), "water".to_string(), "is".to_string(), "calm".to_string(), "today".to_string()],
        2.0,
        vec
![0.1; 384],
        false,
        VADScores::new(0.8, 0.3, 0.7)?, // Positive, calm, dominant
    ).await?;

    println!("✅ Test completed");
    println!("📊 Confidence: {:.3}", result.confidence);
    println!("📊 Rate Target: {:.2} Hz", result.rate_target);
    println!("📊 Attention Reset: {}", result.attention_reset);
    println!();

    // Run test with interruption
    println!("--- Test Conversation 2: Low Equilibrium (Interruption) ---");
    let result = orchestrator.orchestrate(
        vec
!["Hey".to_string(), "wait".to_string()],
        0.5,
        vec
![0.2; 384],
        true,
        VADScores::new(0.3, 0.8, 0.6)?, // Playful, energetic
    ).await?;

    println!("✅ Test completed");
    println!("📊 Confidence: {:.3}", result.confidence);
    println!("📊 Rate Target: {:.2} Hz", result.rate_target);
    println!("📊 Attention Reset: {}", result.attention_reset);
    println!();

    println!("🎉 All tests passed!");
    println!();
    println!("The Equilibrium Tokens system is ready for integration.");
    println!("See docs/ for more information.");
    println!();

    Ok(())
}
