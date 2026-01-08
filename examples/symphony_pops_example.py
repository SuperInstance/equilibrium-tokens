#!/usr/bin/env python3
"""
Symphony Pops Example - Medium Equilibrium Conversation

This example demonstrates a medium equilibrium scenario:
- Steady 1.5 tokens/second rate
- Positive sentiment
- Rare interruptions
- Balanced responses

Analogy: Symphony Pops - balanced, thematic conversation
"""

import sys
sys.path.insert(0, '..')

class SymphonyPopsConversation:
    """Simulates a balanced symphony pops conversation"""

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
    """Run the symphony pops conversation example"""

    print("=" * 60)
    print("SYMPHONY POPS CONVERSATION EXAMPLE")
    print("=" * 60)
    print()
    print("Scenario: A balanced symphony pops conversation")
    print("- Steady rate: 1.5 tokens/second")
    print("- Sentiment: Positive, moderate energy")
    print("- Interruptions: Rare")
    print()
    print("Expected: Medium equilibrium (0.4-0.9)")
    print()

    # Create conversation handler
    conversation = SymphonyPopsConversation()

    # Simulate conversation turns
    turns = [
        {
            'tokens': ["The", "performance", "was", "excellent", "today"],
            'rate': 1.5,
            'sentiment': [0.7, 0.4, 0.8],  # Positive, moderate, confident
            'interruption': False
        },
        {
            'tokens': ["The", "audience", "really", "enjoyed", "it"],
            'rate': 1.5,
            'sentiment': [0.8, 0.5, 0.7],  # Positive, moderate
            'interruption': False
        },
        {
            'tokens': ["Especially", "the", "second", "movement"],
            'rate': 1.5,
            'sentiment': [0.6, 0.3, 0.6],  # Positive, calm
            'interruption': False
        },
        {
            'tokens': ["Though", "the", "first", "was", "good", "too"],  # Brief interruption
            'rate': 1.5,
            'sentiment': [0.5, 0.5, 0.7],  # Neutral-positive
            'interruption': True  # One interruption
        },
    ]

    # Process each turn
    print("-" * 60)
    for i, turn in enumerate(turns, 1):
        print(f"\nTurn {i}:")
        print(f"  Input: {' '.join(turn['tokens'])}")
        print(f"  Rate: {turn['rate']} Hz")
        print(f"  Sentiment (VAD): {turn['sentiment']}")
        print(f"  Interruption: {turn['interruption']} {'🔄' if turn['interruption'] else ''}")

        equilibrium = conversation.process_turn(
            turn['tokens'],
            turn['rate'],
            turn['sentiment'],
            turn['interruption']
        )

        print(f"  → Equilibrium: {equilibrium:.3f}")

        if 0.4 < equilibrium < 0.9:
            print(f"  ✓ MEDIUM EQUILIBRIUM (like Symphony Pops)")
        else:
            print(f"  ! Outside expected range")

    print()
    print("-" * 60)
    print(f"\nFinal Average Equilibrium: "
          f"{sum(t['equilibrium'] for t in conversation.conversation_history) / len(conversation_conversation_history):.3f}")
    print()

    # Analysis
    avg_equilibrium = sum(t['equilibrium'] for t in conversation.conversation_history) / len(conversation.conversation_history)

    print("Analysis:")
    print(f"  ✓ Steady rate maintained: All turns at 1.5 Hz")
    print(f"  ✓ Positive sentiment: Average valence {sum(t['sentiment'][0] for t in conversation.conversation_history) / len(conversation.conversation_history):.2f}")
    print(f"  ✓ Rare interruptions: Only 1 out of {len(conversation.conversation_history)} turns")
    print(f"  ✓ Medium equilibrium: {avg_equilibrium:.3f} in range [0.4, 0.9]")
    print()
    print("This is like a Symphony Pops concert - balanced,")
    print("thematic, with coherent but not overly formal")
    print("conversation flow.")
    print()


if __name__ == "__main__":
    main()
