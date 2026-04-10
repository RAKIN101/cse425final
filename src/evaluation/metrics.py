import numpy as np
from typing import List, Tuple


class EvaluationMetrics:
    """
    Evaluation metrics for generative music models.
    
    Implements:
    - Pitch Histogram Similarity (H)
    - Rhythm Diversity Score (D_rhythm)
    - Repetition Ratio (R)
    - Human Listening Score (aggregation)
    """
    
    @staticmethod
    def pitch_histogram_similarity(predicted: List[int], reference: List[int]) -> float:
        """
        Compute Pitch Histogram Similarity between two sequences.
        
        Formula: H(p,q) = 1 - (1/12) * Σ|pi - qi|
        
        where p,q are pitch class distributions (0-11 for 12 semitones)
        in the range [0, 1] (normalized).
        
        Args:
            predicted: List of MIDI pitches (0-127)
            reference: List of MIDI pitches (0-127)
            
        Returns:
            Similarity score in [0, 1] (higher is better)
        """
        # Convert MIDI pitches to pitch classes (0-11)
        pred_classes = [p % 12 for p in predicted]
        ref_classes = [r % 12 for r in reference]
        
        # Create histograms
        pred_hist = np.bincount(pred_classes, minlength=12) / len(predicted)
        ref_hist = np.bincount(ref_classes, minlength=12) / len(reference)
        
        # Compute similarity: H(p,q) = 1 - (1/12) * Σ|pi - qi|
        diff = np.sum(np.abs(pred_hist - ref_hist))
        similarity = 1.0 - (diff / 12.0)
        
        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
    
    @staticmethod
    def rhythm_diversity_score(durations: List[float]) -> float:
        """
        Compute Rhythm Diversity Score.
        
        Formula: D_rhythm = #unique_durations / #total_notes
        
        Measures how many different note durations are used.
        Higher values indicate more rhythmic variety.
        
        Args:
            durations: List of note durations in beats
            
        Returns:
            Score in [0, 1] (higher is better)
        """
        if len(durations) == 0:
            return 0.0
        
        # Quantize durations to nearest 1/16th for comparison
        quantized = [round(d * 16) / 16 for d in durations]
        unique_durations = len(set(quantized))
        
        diversity = unique_durations / len(durations)
        return min(1.0, diversity)
    
    @staticmethod
    def repetition_ratio(sequence: List[int], pattern_length: int = 2) -> float:
        """
        Compute Repetition Ratio.
        
        Formula: R = #repeated_patterns / #total_patterns
        
        Measures how frequently short patterns repeat.
        Lower is better (less repetitive).
        
        Args:
            sequence: List of MIDI pitches
            pattern_length: Length of patterns to check (default 2)
            
        Returns:
            Score in [0, 1] (lower is better, no repetition)
        """
        if len(sequence) < pattern_length:
            return 0.0
        
        # Extract patterns
        patterns = [tuple(sequence[i:i+pattern_length]) 
                   for i in range(len(sequence) - pattern_length + 1)]
        
        if len(patterns) == 0:
            return 0.0
        
        # Count unique patterns
        unique_patterns = len(set(patterns))
        repeated_patterns = len(patterns) - unique_patterns
        
        # Ratio of repeated to total
        repetition = repeated_patterns / len(patterns)
        return min(1.0, repetition)
    
    @staticmethod
    def aggregate_metrics(
        predicted: List[int],
        reference: List[int],
        durations: List[float],
        human_scores: List[float],
        weights: dict = None
    ) -> dict:
        """
        Compute all evaluation metrics and return aggregated score.
        
        Args:
            predicted: Generated MIDI pitches
            reference: Reference/training MIDI pitches
            durations: Note durations for rhythm analysis
            human_scores: List of human listening scores (1-5)
            weights: Dict of metric weights (default equal)
            
        Returns:
            Dictionary with individual scores and weighted average
        """
        if weights is None:
            weights = {
                'pitch_similarity': 0.25,
                'rhythm_diversity': 0.25,
                'low_repetition': 0.25,
                'human_score': 0.25
            }
        
        # Compute individual metrics
        pitch_sim = EvaluationMetrics.pitch_histogram_similarity(predicted, reference)
        rhythm_div = EvaluationMetrics.rhythm_diversity_score(durations)
        repetition = EvaluationMetrics.repetition_ratio(predicted)
        human_avg = np.mean(human_scores) / 5.0 if human_scores else 0.0  # Normalize to [0,1]
        
        # Low repetition is good, so invert the score
        low_rep = 1.0 - repetition
        
        # Compute weighted average
        weighted_score = (
            weights['pitch_similarity'] * pitch_sim +
            weights['rhythm_diversity'] * rhythm_div +
            weights['low_repetition'] * low_rep +
            weights['human_score'] * human_avg
        )
        
        return {
            'pitch_histogram_similarity': round(pitch_sim, 4),
            'rhythm_diversity_score': round(rhythm_div, 4),
            'repetition_ratio': round(repetition, 4),
            'low_repetition_score': round(low_rep, 4),
            'human_listening_score': round(human_avg * 5.0, 2),
            'human_score_normalized': round(human_avg, 4),
            'weighted_overall_score': round(weighted_score, 4),
        }


if __name__ == "__main__":
    # Example usage
    predicted = np.random.randint(50, 80, 64).tolist()
    reference = np.random.randint(50, 80, 64).tolist()
    durations = [0.5] * 64
    human_scores = [4.4, 4.3, 4.5, 4.2, 4.4]
    
    metrics = EvaluationMetrics.aggregate_metrics(
        predicted, reference, durations, human_scores
    )
    
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
