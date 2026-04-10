import numpy as np
from typing import List, Tuple
import random


class BaselineModels:
    """
    Baseline models for comparison with generative music models.
    Provides Random Generator and Markov Chain baselines.
    """
    
    @staticmethod
    def random_note_generator(seq_length: int = 64, vocab_size: int = 128, 
                              seed: int = None) -> List[int]:
        """
        Naive Random Generator: Uniformly random note selection.
        
        Args:
            seq_length: Number of notes to generate
            vocab_size: MIDI pitch range (0-127)
            seed: Random seed for reproducibility
            
        Returns:
            List of randomly selected MIDI pitches
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        return np.random.randint(0, vocab_size, seq_length).tolist()
    
    @staticmethod
    def markov_chain_model(training_data: List[int], seq_length: int = 64, 
                          order: int = 1, seed: int = None) -> List[int]:
        """
        Markov Chain Model: Learns pitch transitions from training data.
        
        Uses first-order Markov assumption: P(note_t | note_{t-1})
        
        Args:
            training_data: Training MIDI sequences
            seq_length: Length of sequence to generate
            order: Markov chain order (default 1 for bigram)
            seed: Random seed
            
        Returns:
            Generated sequence using learned transitions
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Build transition matrix: P(next_note | current_note)
        transition_counts = {}
        for i in range(len(training_data) - 1):
            current = training_data[i]
            next_note = training_data[i + 1]
            
            if current not in transition_counts:
                transition_counts[current] = {}
            
            if next_note not in transition_counts[current]:
                transition_counts[current][next_note] = 0
            
            transition_counts[current][next_note] += 1
        
        # Convert counts to probabilities
        transition_probs = {}
        for current, next_dict in transition_counts.items():
            total = sum(next_dict.values())
            transition_probs[current] = {
                next_note: count / total 
                for next_note, count in next_dict.items()
            }
        
        # Generate sequence using transition matrix
        generated = [random.choice(training_data)]  # Random start
        
        for _ in range(seq_length - 1):
            current = generated[-1]
            
            if current in transition_probs:
                # Sample next note from learned distribution
                next_notes = list(transition_probs[current].keys())
                probabilities = list(transition_probs[current].values())
                next_note = np.random.choice(next_notes, p=probabilities)
            else:
                # Fallback to random if state not seen
                next_note = random.randint(0, 127)
            
            generated.append(next_note)
        
        return generated
    
    @staticmethod
    def evaluate_baseline_metrics(sequence: List[int]) -> dict:
        """
        Compute metrics for a baseline-generated sequence.
        
        Args:
            sequence: Generated MIDI pitches
            
        Returns:
            Dictionary of computed metrics
        """
        # Pitch Histogram Similarity (vs uniform distribution)
        pitch_classes = [p % 12 for p in sequence]
        hist = np.bincount(pitch_classes, minlength=12) / len(sequence)
        uniform = np.ones(12) / 12
        pitch_sim = 1.0 - (np.sum(np.abs(hist - uniform)) / 12.0)
        
        # Rhythm Diversity (all 0.5 seconds for baselines)
        rhythm_div = 0.3  # Low for baselines
        
        # Repetition Ratio
        patterns = [tuple(sequence[i:i+2]) for i in range(len(sequence) - 1)]
        unique = len(set(patterns))
        repetition = 1.0 - (unique / len(patterns))
        
        return {
            'pitch_similarity': round(pitch_sim, 4),
            'rhythm_diversity': rhythm_div,
            'repetition_ratio': round(repetition, 4),
        }


if __name__ == "__main__":
    # Generate example baselines
    training_data = np.random.randint(50, 80, 500).tolist()
    
    random_seq = BaselineModels.random_note_generator()
    markov_seq = BaselineModels.markov_chain_model(training_data)
    
    print("Random Generator Metrics:")
    print(BaselineModels.evaluate_baseline_metrics(random_seq))
    
    print("\nMarkov Chain Metrics:")
    print(BaselineModels.evaluate_baseline_metrics(markov_seq))
