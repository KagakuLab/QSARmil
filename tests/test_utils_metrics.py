import pytest

from qsarmil.utils.metrics import kid_accuracy


def test_kid_accuracy_perfect_predictions():
    true_key_inst = [[0, 1, 0], [1, 0, 0]]
    predicted_weights = [[0.1, 0.9, 0.2], [0.8, 0.1, 0.1]]
    acc, exp = kid_accuracy(true_key_inst, predicted_weights, top_n=1)
    assert acc == 1.0
    assert 0 <= exp <= 1


def test_kid_accuracy_miss():
    true_key_inst = [[0, 1, 0]]
    predicted_weights = [[0.9, 0.1, 0.2]]
    acc, _ = kid_accuracy(true_key_inst, predicted_weights, top_n=1)
    assert acc == 0.0


def test_kid_accuracy_hit_probability_guard_branch():
    """bag_size - num_key_instances < num_pred_instances -> hit_probability forced to 1.0."""
    true_key_inst = [[1, 1, 1]]  # every instance is a key instance
    predicted_weights = [[0.1, 0.2, 0.3]]
    _, exp = kid_accuracy(true_key_inst, predicted_weights, top_n=2)
    assert exp == 1.0


def test_kid_accuracy_skips_empty_or_keyless_bags():
    """bag_size == 0 or num_key_instances == 0 -> contributes 0 to expected_hits, no crash."""
    true_key_inst = [[0, 0, 0], [1, 0, 0]]
    predicted_weights = [[0.1, 0.2, 0.3], [0.5, 0.4, 0.3]]
    acc, exp = kid_accuracy(true_key_inst, predicted_weights, top_n=1)
    assert 0 <= acc <= 1
    assert 0 <= exp <= 1


def test_kid_accuracy_mismatched_lengths_raises():
    with pytest.raises(AssertionError):
        kid_accuracy([[1, 0]], [[0.1, 0.2], [0.3, 0.4]])
