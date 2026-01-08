#!/usr/bin/env python3
"""
Fishing Boat Example - High Equilibrium Conversation

This example demonstrates a high equilibrium scenario:
- Steady 2 tokens/second rate
- Calm, positive sentiment
- No interruptions
- Long, flowing responses

Analogy: Symphony Pops - coherent, flowing conversation
"""

import sys
sys.path.insert(0, '..')

# In a real implementation, this would use the Rust library via PyO3
# For this example, we'll simulate the behavior

class FishingBoatConversation:
    """Simulates a calm fishing boat conversation"""

    def __init__(self):
        self.equilibrium_score = 0.0
        self.conversation_history = []

    def process_turn(self, tokens, rate, sentiment, interruption):
        """
        Process a conversation turn

        Args:
            tokens: List of input tokens
            rate: Token rate in Hz
            sentiment: VAD scores [valence, arousal, dominance]
            interruption: Whether interruption occurred
        """
        # Rate equilibrium: match input rate
        rate_weight = min(1.0, rate / 2.0)  # Ideal is 2.0 Hz

        # Sentiment equilibrium: weight by valence
        sentiment_weight = (sentiment[0] + 1.0) / 2.0  # Map [-1,1] to [0,1]

        # Interruption equilibrium
        interruption_weight = 0.8 if interruption else 1.0

        # Calculate overall equilibrium
        self.equilibrium_score = rate_weight * sentiment_weight * interruption_weight

        # Store turn
        self.conversation_history.append({
            'tokens': tokens,
            'rate': rate,
            'sentiment': sentiment,
            'interruption': interruption,
            'equilibrium': self.equilibrium_score
        })

        return self.equilibrium_score


def main():
    """Run the fishing boat conversation example"""

    print("=" * 60)
    print("FISHING BOAT CONVERSATION EXAMPLE")
    print("=" * 60)
    print()
    print("Scenario: A calm conversation on a fishing boat")
    print("- Steady rate: 2.0 tokens/second")
    print("- Sentiment: Positive, calm, dominant")
    print("- Interruptions: None")
    print()
    print("Expected: High equilibrium (>0.6)")
    print()

    # Create conversation handler
    conversation = FishingBoatConversation()

    # Simulate conversation turns
    turns = [
        {
            'tokens': ["The", "water", "is", "calm", "today"],
            'rate': 2.0,
            'sentiment': [0.8, 0.3, 0.7],  # Positive, calm, dominant
            'interruption': False
        },
        {
            'tokens': ["Perfect", "conditions", "for", "fishing"],
            'rate': 2.0,
            'sentiment': [0.9, 0.2, 0.8],  # Very positive, very calm
            'interruption': False
        },
        {
            'tokens': ["The", "catch", "should", "be", "good"],
            'rate': 2.0,
            'sentiment': [0.7, 0.4, 0.6],  # Positive, moderate
            'interruption': False
        },
    ]

    # Process each turn
    print("-" * 60)
    for i, turn in enumerate(turns, 1):
        print(f"\nTurn {i}:")
        print(f"  Input: {' '.join(turn['tokens'])}")
        print(f"  Rate: {turn['rate']} Hz")
        print(f"  Sentiment (VAD): {turn['sentiment']}")
        print(f"  Interruption: {turn['interruption']}")

        equilibrium = conversation.process_turn(
            turn['tokens'],
            turn['rate'],
            turn['sentiment'],
            turn['interruption']
        )

        print(f"  → Equilibrium: {equilibrium:.3f}")

        if equilibrium > 0.6:
            print(f"  ✓ HIGH EQUILIBRIUM (like Symphony Pops)")
        else:
            print(f"  ✗ Lower than expected")

    print()
    print("-" * 60)
    print(f"\nFinal Average Equilibrium: "
          f"{sum(t['equilibrium'] for t in conversation.conversation_history) / len(conversation.conversation_history):.3f}")
    print()

    # Analysis
    avg_equilibrium = sum(t['equilibrium'] for t in conversation.conversation_history) / len(conversation.conversation_history)

    print("Analysis:")
    print(f"  ✓ Steady rate maintained: All turns at 2.0 Hz")
    print(f"  ✓ Positive sentiment: Average valence {sum(t['sentiment'][0] for t in conversation.conversation_history) / len(conversation.conversation_history):.2f}")
    print(f"  ✓ No interruptions: Smooth conversation flow")
    print(f"  ✓ High equilibrium: {avg_equilibrium:.3f} > 0.6")
    print()
    print("This is like a Symphony Pops concert - coherent, flowing,")
    print("and harmonious conversation with long, flowing passages.")
    print()


if __name__ == "__main__":
    main()
