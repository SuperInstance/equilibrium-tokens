#!/usr/bin/env python3
"""
Jazz Band Example - Low Equilibrium Conversation

This example demonstrates a low equilibrium scenario:
- Variable 0.5-4 tokens/second rate (syncopated)
- Playful, energetic sentiment
- Frequent interruptions
- Short, staccato responses

Analogy: Dixieland Jazz - fragmented, improvisational conversation
"""

import sys
sys.path.insert(0, '..')

class JazzBandConversation:
    """Simulates a playful jazz band conversation"""

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
        # Rate equilibrium: penalize variable rates
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
    """Run the jazz band conversation example"""

    print("=" * 60)
    print("JAZZ BAND CONVERSATION EXAMPLE")
    print("=" * 60)
    print()
    print("Scenario: A playful jazz band conversation")
    print("- Variable rate: 0.5-4.0 tokens/second (syncopated)")
    print("- Sentiment: Playful, energetic")
    print("- Interruptions: Frequent")
    print()
    print("Expected: Low equilibrium (<0.7)")
    print()

    # Create conversation handler
    conversation = JazzBandConversation()

    # Simulate conversation turns (syncopated rhythm)
    turns = [
        {
            'tokens': ["Hey"],
            'rate': 4.0,  # Fast!
            'sentiment': [0.5, 0.9, 0.7],  # Playful, very energetic
            'interruption': True
        },
        {
            'tokens': ["wait"],
            'rate': 0.5,  # Slow!
            'sentiment': [0.3, 0.8, 0.6],  # Playful
            'interruption': True
        },
        {
            'tokens': ["listen"],
            'rate': 3.0,  # Fast again!
            'sentiment': [0.4, 0.9, 0.8],  # Very energetic
            'interruption': True
        },
        {
            'tokens': ["no"],
            'rate': 0.5,  # Slow again!
            'sentiment': [0.2, 0.7, 0.5],  # Less playful
            'interruption': False  # Brief pause
        },
        {
            'tokens': ["actually"],
            'rate': 2.0,  # Moderate
            'sentiment': [0.6, 0.8, 0.7],  # Back to playful
            'interruption': True
        },
    ]

    # Process each turn
    print("-" * 60)
    for i, turn in enumerate(turns, 1):
        print(f"\nTurn {i}:")
        print(f"  Input: {' '.join(turn['tokens'])}")
        print(f"  Rate: {turn['rate']} Hz {'⚡' if turn['rate'] > 2.0 else '🐢' if turn['rate'] < 1.0 else '🚶'}")
        print(f"  Sentiment (VAD): {turn['sentiment']}")
        print(f"  Interruption: {turn['interruption']} {'🔄' if turn['interruption'] else ''}")

        equilibrium = conversation.process_turn(
            turn['tokens'],
            turn['rate'],
            turn['sentiment'],
            turn['interruption']
        )

        print(f"  → Equilibrium: {equilibrium:.3f}")

        if equilibrium < 0.7:
            print(f"  ✓ LOW EQUILIBRIUM (like Dixieland Jazz)")
        else:
            print(f"  ✗ Higher than expected")

    print()
    print("-" * 60)
    print(f"\nFinal Average Equilibrium: "
          f"{sum(t['equilibrium'] for t in conversation.conversation_history) / len(conversation.conversation_history):.3f}")
    print()

    # Analysis
    avg_equilibrium = sum(t['equilibrium'] for t in conversation.conversation_history) / len(conversation.conversation_history)

    print("Analysis:")
    print(f"  ✓ Variable rate: Range {min(t['rate'] for t in conversation.conversation_history):.1f}-{max(t['rate'] for t in conversation.conversation_history):.1f} Hz")
    print(f"  ✓ High arousal: Average {sum(t['sentiment'][1] for t in conversation.conversation_history) / len(conversation.conversation_history):.2f}")
    print(f"  ✓ Frequent interruptions: {sum(1 for t in conversation.conversation_history if t['interruption'])} out of {len(conversation.conversation_history)} turns")
    print(f"  ✓ Low equilibrium: {avg_equilibrium:.3f} < 0.7")
    print()
    print("This is like a Dixieland Jazz band - fragmented,")
    print("improvisational, with short, staccato bursts of")
    print("conversation and constant interruptions.")
    print()


if __name__ == "__main__":
    main()
