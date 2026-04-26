"""
Property-based tests for differential privacy noise statistical properties.

**Validates: Requirements 15.10**

This module tests that the Laplace noise mechanism used for differential
privacy satisfies the correct statistical properties:
1. Mean preservation (unbiased noise)
2. Variance proportional to 1/ε² (Laplace distribution property)
3. Distribution follows Laplace (verified using Kolmogorov-Smirnov test)

Property 5: Differential Privacy Noise Statistical Properties
- Laplace noise preserves the mean of the original value
- Variance is proportional to 1/ε² as expected for Laplace distribution
- The noise distribution follows a Laplace distribution
"""

import pytest
import numpy as np
from scipy import stats
from hypothesis import given, strategies as st, settings, assume

from core.encryption import add_laplace_noise


# ─── Test Strategies ────────────────────────────────────────────────────────

def valid_health_values() -> st.SearchStrategy[float]:
    """
    Generate valid health metric values.
    
    Generates values in the range [0.0, 100.0] which covers:
    - Mood scores (1-10)
    - Sleep hours (0-24)
    - Energy levels (1-10)
    - Heart rate (30-220 bpm)
    - SpO2 (70-100%)
    - Temperature (35-42°C)
    """
    return st.floats(
        min_value=0.0,
        max_value=100.0,
        allow_nan=False,
        allow_infinity=False,
    )


def valid_epsilon_values() -> st.SearchStrategy[float]:
    """
    Generate valid epsilon (privacy budget) values.
    
    Epsilon controls the privacy-utility tradeoff:
    - Lower epsilon = more privacy, more noise
    - Higher epsilon = less privacy, less noise
    
    Typical range: 0.1 to 10.0
    """
    return st.floats(
        min_value=0.1,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False,
    )


def valid_sample_counts() -> st.SearchStrategy[int]:
    """
    Generate valid sample counts for statistical testing.
    
    We need at least 1000 samples for reliable statistical tests.
    Upper limit is 10000 to keep tests reasonably fast.
    """
    return st.integers(min_value=1000, max_value=10000)


# ─── Property Tests ─────────────────────────────────────────────────────────

@given(
    value=valid_health_values(),
    epsilon=valid_epsilon_values(),
    num_samples=valid_sample_counts(),
)
@settings(max_examples=50, deadline=None)
def test_laplace_noise_preserves_mean(value: float, epsilon: float, num_samples: int):
    """
    **Property 5: Differential Privacy Noise Statistical Properties (Mean Preservation)**
    **Validates: Requirements 15.10**
    
    Property: For any value V and epsilon ε, the mean of many noisy samples
    should be close to the original value V (unbiased noise).
    
    Mathematical property:
        E[V + Laplace(0, b)] = V
        where b = sensitivity/ε
    
    We verify this by generating many samples and checking that the sample mean
    is within 3 standard errors of the true value (99.7% confidence interval).
    
    Standard error = σ/√n where σ = √(2b²) for Laplace distribution
    """
    # Generate multiple noisy samples
    noisy_values = [
        add_laplace_noise(value, epsilon=epsilon, sensitivity=1.0)
        for _ in range(num_samples)
    ]
    
    # Calculate sample mean
    sample_mean = np.mean(noisy_values)
    
    # Calculate expected standard error
    # For Laplace distribution: Var = 2b² where b = sensitivity/epsilon
    scale = 1.0 / epsilon  # b = sensitivity/epsilon
    variance = 2 * scale ** 2
    std_dev = np.sqrt(variance)
    standard_error = std_dev / np.sqrt(num_samples)
    
    # Mean should be within 3 standard errors (99.7% confidence)
    tolerance = 3 * standard_error
    
    assert abs(sample_mean - value) < tolerance, (
        f"Mean preservation failed!\n"
        f"Original value: {value}\n"
        f"Sample mean: {sample_mean}\n"
        f"Difference: {abs(sample_mean - value)}\n"
        f"Tolerance (3 SE): {tolerance}\n"
        f"Epsilon: {epsilon}\n"
        f"Num samples: {num_samples}\n"
        f"Standard error: {standard_error}"
    )


@given(
    value=valid_health_values(),
    epsilon=valid_epsilon_values(),
    num_samples=valid_sample_counts(),
)
@settings(max_examples=50, deadline=None)
def test_laplace_noise_variance_proportional_to_epsilon_squared(
    value: float, epsilon: float, num_samples: int
):
    """
    **Property 5: Differential Privacy Noise Statistical Properties (Variance)**
    **Validates: Requirements 15.10**
    
    Property: For any value V and epsilon ε, the variance of noisy samples
    should be proportional to 1/ε².
    
    Mathematical property:
        Var[V + Laplace(0, b)] = 2b² = 2(sensitivity/ε)²
    
    For sensitivity = 1.0:
        Expected variance = 2/ε²
    
    We allow 20% tolerance due to sampling variability.
    """
    # Generate multiple noisy samples
    noisy_values = [
        add_laplace_noise(value, epsilon=epsilon, sensitivity=1.0)
        for _ in range(num_samples)
    ]
    
    # Calculate sample variance
    sample_variance = np.var(noisy_values, ddof=1)  # Use sample variance (n-1)
    
    # Calculate expected variance for Laplace distribution
    # Var(Laplace(μ, b)) = 2b² where b = sensitivity/epsilon
    scale = 1.0 / epsilon
    expected_variance = 2 * scale ** 2
    
    # Allow 20% tolerance due to sampling variability
    # This is reasonable for sample sizes >= 1000
    tolerance = 0.20 * expected_variance
    
    assert abs(sample_variance - expected_variance) < tolerance, (
        f"Variance property failed!\n"
        f"Expected variance: {expected_variance}\n"
        f"Sample variance: {sample_variance}\n"
        f"Difference: {abs(sample_variance - expected_variance)}\n"
        f"Tolerance (20%): {tolerance}\n"
        f"Epsilon: {epsilon}\n"
        f"Scale (b): {scale}\n"
        f"Num samples: {num_samples}"
    )


@given(
    value=valid_health_values(),
    epsilon=valid_epsilon_values(),
    num_samples=valid_sample_counts(),
)
@settings(max_examples=30, deadline=None)
def test_laplace_noise_follows_laplace_distribution(
    value: float, epsilon: float, num_samples: int
):
    """
    **Property 5: Differential Privacy Noise Statistical Properties (Distribution)**
    **Validates: Requirements 15.10**
    
    Property: For any value V and epsilon ε, the noisy samples should follow
    a Laplace distribution centered at V with scale b = sensitivity/ε.
    
    We verify this using the Kolmogorov-Smirnov (K-S) test, which compares
    the empirical distribution of samples to the theoretical Laplace distribution.
    
    The K-S test returns a p-value:
    - p > 0.05: Cannot reject null hypothesis (samples follow Laplace)
    - p ≤ 0.05: Reject null hypothesis (samples don't follow Laplace)
    
    We use p > 0.01 as threshold to reduce false positives from sampling variability.
    """
    # Generate multiple noisy samples
    noisy_values = [
        add_laplace_noise(value, epsilon=epsilon, sensitivity=1.0)
        for _ in range(num_samples)
    ]
    
    # Calculate scale parameter for Laplace distribution
    scale = 1.0 / epsilon  # b = sensitivity/epsilon
    
    # Perform Kolmogorov-Smirnov test
    # Compare empirical distribution to theoretical Laplace(value, scale)
    ks_statistic, p_value = stats.kstest(
        noisy_values,
        lambda x: stats.laplace.cdf(x, loc=value, scale=scale)
    )
    
    # We use p > 0.01 as threshold (more lenient than typical 0.05)
    # to account for sampling variability in property-based testing
    assert p_value > 0.01, (
        f"Kolmogorov-Smirnov test failed!\n"
        f"The samples do not follow a Laplace distribution.\n"
        f"K-S statistic: {ks_statistic}\n"
        f"P-value: {p_value}\n"
        f"Threshold: 0.01\n"
        f"Original value: {value}\n"
        f"Epsilon: {epsilon}\n"
        f"Scale (b): {scale}\n"
        f"Num samples: {num_samples}\n"
        f"Sample mean: {np.mean(noisy_values)}\n"
        f"Sample std: {np.std(noisy_values)}"
    )


@given(
    value=valid_health_values(),
    epsilon=valid_epsilon_values(),
)
@settings(max_examples=100, deadline=None)
def test_laplace_noise_produces_different_values(value: float, epsilon: float):
    """
    **Property 5: Differential Privacy Noise Statistical Properties (Randomness)**
    **Validates: Requirements 15.10**
    
    Property: For any value V and epsilon ε, multiple calls to add_laplace_noise
    should produce different noisy values (noise is random, not deterministic).
    
    This verifies that the noise mechanism is actually adding randomness
    and not just returning the original value or a fixed offset.
    """
    # Generate multiple noisy samples
    noisy_values = [
        add_laplace_noise(value, epsilon=epsilon, sensitivity=1.0)
        for _ in range(10)
    ]
    
    # At least some values should differ from the original
    # (with very high probability for reasonable epsilon values)
    different_values = [v for v in noisy_values if abs(v - value) > 0.001]
    
    assert len(different_values) >= 8, (
        f"Noise mechanism appears deterministic!\n"
        f"Original value: {value}\n"
        f"Noisy values: {noisy_values}\n"
        f"Number of different values: {len(different_values)}\n"
        f"Expected at least 8 out of 10 to differ"
    )
    
    # Values should not all be identical
    unique_values = len(set(noisy_values))
    assert unique_values >= 8, (
        f"Noise mechanism produced too many identical values!\n"
        f"Original value: {value}\n"
        f"Noisy values: {noisy_values}\n"
        f"Unique values: {unique_values}\n"
        f"Expected at least 8 unique values out of 10"
    )


# ─── Edge Case Tests ────────────────────────────────────────────────────────

def test_zero_value_with_noise():
    """
    Test that zero values can have noise added correctly.
    
    This is an important edge case for health metrics that can be zero
    (e.g., pain level = 0, steps = 0).
    """
    value = 0.0
    epsilon = 1.0
    num_samples = 1000
    
    noisy_values = [
        add_laplace_noise(value, epsilon=epsilon, sensitivity=1.0)
        for _ in range(num_samples)
    ]
    
    # Mean should be close to zero
    sample_mean = np.mean(noisy_values)
    scale = 1.0 / epsilon
    std_error = np.sqrt(2 * scale ** 2) / np.sqrt(num_samples)
    
    assert abs(sample_mean) < 3 * std_error, (
        f"Mean preservation failed for zero value!\n"
        f"Sample mean: {sample_mean}\n"
        f"Expected: ~0.0\n"
        f"Tolerance: {3 * std_error}"
    )


def test_high_epsilon_produces_low_noise():
    """
    Test that high epsilon (low privacy) produces low noise.
    
    High epsilon means less privacy protection, so noise should be small.
    """
    value = 50.0
    epsilon = 10.0  # High epsilon = low noise
    num_samples = 1000
    
    noisy_values = [
        add_laplace_noise(value, epsilon=epsilon, sensitivity=1.0)
        for _ in range(num_samples)
    ]
    
    # With high epsilon, most values should be close to original
    close_values = [v for v in noisy_values if abs(v - value) < 1.0]
    
    # At least 60% of values should be within 1.0 of original
    assert len(close_values) >= 600, (
        f"High epsilon should produce low noise!\n"
        f"Epsilon: {epsilon}\n"
        f"Values within 1.0 of original: {len(close_values)}/1000\n"
        f"Expected at least 600"
    )


def test_low_epsilon_produces_high_noise():
    """
    Test that low epsilon (high privacy) produces high noise.
    
    Low epsilon means more privacy protection, so noise should be large.
    """
    value = 50.0
    epsilon = 0.1  # Low epsilon = high noise
    num_samples = 1000
    
    noisy_values = [
        add_laplace_noise(value, epsilon=epsilon, sensitivity=1.0)
        for _ in range(num_samples)
    ]
    
    # With low epsilon, values should be spread out
    far_values = [v for v in noisy_values if abs(v - value) > 5.0]
    
    # At least 40% of values should be more than 5.0 away from original
    assert len(far_values) >= 400, (
        f"Low epsilon should produce high noise!\n"
        f"Epsilon: {epsilon}\n"
        f"Values more than 5.0 from original: {len(far_values)}/1000\n"
        f"Expected at least 400"
    )


def test_different_sensitivity_scales_noise():
    """
    Test that different sensitivity values scale the noise appropriately.
    
    Higher sensitivity should produce more noise for the same epsilon.
    """
    value = 50.0
    epsilon = 1.0
    num_samples = 1000
    
    # Generate samples with sensitivity = 1.0
    noisy_values_s1 = [
        add_laplace_noise(value, epsilon=epsilon, sensitivity=1.0)
        for _ in range(num_samples)
    ]
    
    # Generate samples with sensitivity = 2.0
    noisy_values_s2 = [
        add_laplace_noise(value, epsilon=epsilon, sensitivity=2.0)
        for _ in range(num_samples)
    ]
    
    # Variance should be 4x larger for sensitivity = 2.0
    # (variance scales with sensitivity²)
    var_s1 = np.var(noisy_values_s1)
    var_s2 = np.var(noisy_values_s2)
    
    # Allow 30% tolerance due to sampling variability
    expected_ratio = 4.0
    actual_ratio = var_s2 / var_s1
    
    assert abs(actual_ratio - expected_ratio) < 0.3 * expected_ratio, (
        f"Sensitivity scaling failed!\n"
        f"Variance (sensitivity=1.0): {var_s1}\n"
        f"Variance (sensitivity=2.0): {var_s2}\n"
        f"Ratio: {actual_ratio}\n"
        f"Expected ratio: {expected_ratio}\n"
        f"Tolerance: {0.3 * expected_ratio}"
    )


def test_noise_is_symmetric_around_value():
    """
    Test that noise is symmetric around the original value.
    
    Laplace distribution is symmetric, so we should see roughly equal
    numbers of values above and below the original.
    """
    value = 50.0
    epsilon = 1.0
    num_samples = 10000  # Large sample for symmetry test
    
    noisy_values = [
        add_laplace_noise(value, epsilon=epsilon, sensitivity=1.0)
        for _ in range(num_samples)
    ]
    
    # Count values above and below original
    above = sum(1 for v in noisy_values if v > value)
    below = sum(1 for v in noisy_values if v < value)
    
    # Should be roughly 50/50 split (allow 5% deviation)
    expected = num_samples / 2
    tolerance = 0.05 * num_samples
    
    assert abs(above - expected) < tolerance, (
        f"Noise is not symmetric!\n"
        f"Values above original: {above}\n"
        f"Values below original: {below}\n"
        f"Expected: ~{expected} each\n"
        f"Tolerance: {tolerance}"
    )
