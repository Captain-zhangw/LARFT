"""
[LARFT] Length-oriented reward function for reinforcement learning.

Implements the verified length reward R_len (Eq. 3 in paper):
    R_len(tau, c) = max(0, 1 - |L(tau) - c| / c)

where L(tau) is the word count of the generated response and c is the target length.
"""

import re
from collections import Counter


def extract_solution(solution_str: str) -> str:
    """Extract and clean the final generated content from model's raw output."""
    separators = ["<|im_start|>assistant", "<｜Assistant｜>", "Assistant:"]
    processed_str = solution_str

    for sep in separators:
        if sep in solution_str:
            processed_str = solution_str.split(sep)[-1]
            break

    processed_str = re.sub(r"<think>.*?</think>", "", processed_str, flags=re.DOTALL)
    processed_str = processed_str.replace("<|im_end|>", "")
    return processed_str.strip()


def count_words(text: str) -> int:
    """Count words in text (supports both Chinese characters and English words).

    Implements COUNTWORDS (Eq. 1, Eq. 9 in paper):
        COUNTWORDS(s) = |MATCH(s, P)|
    where P = [\\u4e00-\\u9fff]|[a-zA-Z0-9\\'-]+ matches Chinese characters and English words.
    """
    patt = re.compile(r"[一-鿿]|[a-zA-Z0-9\'-]+")
    return len(patt.findall(text))


def is_degenerate_sequence(text: str, n: int = 3, threshold: float = 0.5) -> bool:
    """Detect degenerate sequences characterized by excessive n-gram repetition.

    As described in the paper: "we explicitly discard degenerate sequences characterized
    by excessive repetition (e.g., n-gram loops)".

    This function checks whether the ratio of repeated n-grams exceeds a threshold,
    indicating a degenerate (looping) generation.

    Args:
        text: The text to check for repetition.
        n: The n-gram size to check (default: 3, i.e., trigrams).
        threshold: The maximum allowed ratio of repeated n-grams. If the fraction
            of n-grams that appear more than once exceeds this threshold, the
            sequence is considered degenerate. Default: 0.5.

    Returns:
        True if the sequence is degenerate (excessive repetition), False otherwise.
    """
    words = text.split()
    if len(words) < n + 1:
        return False

    # Build n-gram list
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    total = len(ngrams)
    if total == 0:
        return False

    # Count n-gram frequencies
    counts = Counter(ngrams)
    # Number of n-gram occurrences that are duplicates (total - unique)
    repeated = total - len(counts)
    repetition_ratio = repeated / total

    return repetition_ratio > threshold


def compute_length_reward(actual_length: int, target_length: int) -> float:
    """Compute the piecewise linear length reward (Eq. 2-3 in paper).

    R_len(tau, c) = max(0, 1 - delta(tau, c))
    where delta(tau, c) = |L(tau) - c| / c

    Args:
        actual_length: The word count of the generated response L(tau).
        target_length: The target length constraint c.

    Returns:
        Reward value in [0, 1].
    """
    if target_length == 0:
        return 1.0 if actual_length == 0 else 0.0
    delta = abs(actual_length - target_length) / target_length
    return max(0.0, 1.0 - delta)


def compute_score(
    solution_str: str,
    ground_truth: int,
    split: str,
    current_step: int = -1,
    total_steps: int = -1,
    linear_truth: bool = True,
    filter_repetition: bool = False,
    repetition_n: int = 3,
    repetition_threshold: float = 0.5,
) -> dict:
    """Main reward function entry point, called by the verl framework.

    Args:
        solution_str: Raw model output string.
        ground_truth: Target word count constraint c.
        split: 'train' or 'test'.
        current_step: Current training step (unused, kept for interface compatibility).
        total_steps: Total training steps (unused, kept for interface compatibility).
        linear_truth: Kept for interface compatibility.
        filter_repetition: Whether to detect and penalize degenerate (repetitive)
            sequences. When enabled, sequences with excessive n-gram repetition
            receive a reward of 0. Default: False (disabled).
        repetition_n: N-gram size for repetition detection. Default: 3.
        repetition_threshold: Maximum allowed repetition ratio. Default: 0.5.

    Returns:
        Dictionary with 'score', 'quality', and 'length' keys.
    """
    clean_solution = extract_solution(solution_str)
    actual_count = count_words(clean_solution)
    length_score = compute_length_reward(actual_count, ground_truth)

    # [LARFT] Optional: discard degenerate sequences with excessive repetition
    if filter_repetition and is_degenerate_sequence(
        clean_solution, n=repetition_n, threshold=repetition_threshold
    ):
        length_score = 0.0

    return {
        "score": length_score,
        "quality": length_score,
        "length": length_score,
    }


if __name__ == "__main__":
    # Quick sanity check
    test_str = "<think>The user is testing.</think>This is the final answer with ten words here now."
    extracted = extract_solution(test_str)
    print(f"Extracted: '{extracted}'")
    print(f"Word count: {count_words(extracted)}")

    # Test reward computation
    print(f"\nReward (target=100, actual=90): {compute_length_reward(90, 100):.4f}")  # 0.9
    print(f"Reward (target=100, actual=50): {compute_length_reward(50, 100):.4f}")  # 0.5
    print(f"Reward (target=100, actual=100): {compute_length_reward(100, 100):.4f}")  # 1.0
    print(f"Reward (target=100, actual=200): {compute_length_reward(200, 100):.4f}")  # 0.0

    # Test repetition detection
    print("\n--- Repetition Detection ---")
    normal_text = "The quick brown fox jumps over the lazy dog and then runs away."
    print(f"Normal text degenerate: {is_degenerate_sequence(normal_text)}")  # False

    repetitive_text = "hello world hello world hello world hello world hello world " * 10
    print(f"Repetitive text degenerate: {is_degenerate_sequence(repetitive_text)}")  # True

    # Test compute_score with filter_repetition enabled
    print(f"\nScore (normal, filter=True): {compute_score(normal_text, 12, 'train', filter_repetition=True)['score']:.4f}")
    print(f"Score (repetitive, filter=True): {compute_score(repetitive_text, 50, 'train', filter_repetition=True)['score']:.4f}")
    print(f"Score (repetitive, filter=False): {compute_score(repetitive_text, 50, 'train', filter_repetition=False)['score']:.4f}")
