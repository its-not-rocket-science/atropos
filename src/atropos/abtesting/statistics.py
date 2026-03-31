"""Statistical functions for A/B testing analysis.

Provides implementations of common statistical tests for comparing variant
performance, with optional SciPy dependency for advanced functionality.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

try:
    from scipy import stats  # type: ignore

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def independent_t_test(
    sample_a: list[float],
    sample_b: list[float],
    equal_var: bool = True,
) -> dict[str, Any]:
    """Perform independent two-sample t-test.

    Args:
        sample_a: First sample values.
        sample_b: Second sample values.
        equal_var: Assume equal population variances (default True).

    Returns:
        Dictionary with t-statistic, p-value, degrees of freedom.
    """
    n_a, n_b = len(sample_a), len(sample_b)
    if n_a < 2 or n_b < 2:
        return {
            "t_statistic": None,
            "p_value": None,
            "degrees_of_freedom": None,
            "method": "insufficient_data",
            "error": "At least 2 samples required per group",
        }

    if SCIPY_AVAILABLE:
        # Use SciPy for accurate implementation
        t_stat, p_value = stats.ttest_ind(sample_a, sample_b, equal_var=equal_var)
        df_scipy = float(n_a + n_b - 2) if equal_var else float(n_a + n_b - 2)  # Approximate
        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "degrees_of_freedom": df_scipy,
            "method": "scipy_ttest_ind",
        }

    # Fallback pure Python implementation

    mean_a = statistics.mean(sample_a)
    mean_b = statistics.mean(sample_b)
    var_a = statistics.variance(sample_a) if n_a > 1 else 0.0
    var_b = statistics.variance(sample_b) if n_b > 1 else 0.0

    if equal_var:
        # Pooled variance t-test
        pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        std_error = math.sqrt(pooled_var * (1.0 / n_a + 1.0 / n_b))
        df: float = float(n_a + n_b - 2)
    else:
        # Welch's t-test (unequal variances)
        std_error = math.sqrt(var_a / n_a + var_b / n_b)
        # Welch–Satterthwaite equation for degrees of freedom
        df_numerator = (var_a / n_a + var_b / n_b) ** 2
        df_denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        df = df_numerator / df_denom if df_denom != 0 else 1.0

    if std_error == 0:
        # Samples are identical
        return {
            "t_statistic": 0.0,
            "p_value": 1.0,
            "degrees_of_freedom": df,
            "method": "pure_python_ttest",
        }

    t_stat = (mean_a - mean_b) / std_error
    # Two-tailed p-value approximation using t-distribution
    p_value = _t_distribution_two_tailed_p_value(t_stat, df)

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "degrees_of_freedom": df,
        "method": "pure_python_ttest",
    }


def _t_distribution_two_tailed_p_value(t_stat: float, df: float) -> float:
    """Approximate two-tailed p-value from t-distribution.

    Uses approximation formulas for t-distribution CDF.
    For better accuracy, SciPy should be installed.
    """
    if df <= 0:
        return 1.0

    # Approximate using normal distribution for large df
    if df > 30:
        # Use normal approximation
        z = abs(t_stat)
        # Abramowitz & Stegun approximation for normal CDF
        b1 = 0.319381530
        b2 = -0.356563782
        b3 = 1.781477937
        b4 = -1.821255978
        b5 = 1.330274429
        p = 0.2316419
        t = 1.0 / (1.0 + p * z)
        nd = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-z * z / 2.0)
        cdf = 1.0 - nd * (b1 * t + b2 * t**2 + b3 * t**3 + b4 * t**4 + b5 * t**5)
        return 2.0 * (1.0 - cdf)
    else:
        # Simple approximation for small df
        # This is less accurate; recommend SciPy for production use
        z = abs(t_stat)
        # Crude approximation
        approx_p = math.exp(-0.717 * z - 0.416 * z**2 / (df + 1))
        return min(2.0 * approx_p, 1.0)


def confidence_interval(
    sample: list[float],
    confidence_level: float = 0.95,
) -> tuple[float, float] | None:
    """Calculate confidence interval for sample mean.

    Args:
        sample: Sample values.
        confidence_level: Desired confidence level (0.0-1.0).

    Returns:
        (lower_bound, upper_bound) or None if insufficient data.
    """
    n = len(sample)
    if n < 2:
        return None

    mean = statistics.mean(sample)
    stderr = statistics.stdev(sample) / math.sqrt(n) if n > 1 else 0.0

    if SCIPY_AVAILABLE:
        # Use t-distribution critical value
        alpha = 1.0 - confidence_level
        t_critical = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
    else:
        # Approximate with normal distribution for large n, else use conservative estimate
        if n >= 30:
            # Z-values for common confidence levels
            z_values = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
            t_critical = z_values.get(confidence_level, 1.96)
        else:
            # Conservative approximation: use larger critical value for small samples
            t_critical = 2.0 + 4.0 / max(n, 1)

    margin = t_critical * stderr
    return (mean - margin, mean + margin)


def effect_size_cohens_d(
    sample_a: list[float],
    sample_b: list[float],
) -> float | None:
    """Calculate Cohen's d effect size.

    Cohen's d = (mean1 - mean2) / pooled_standard_deviation

    Args:
        sample_a: First sample values.
        sample_b: Second sample values.

    Returns:
        Cohen's d effect size, or None if insufficient data.
    """
    n_a, n_b = len(sample_a), len(sample_b)
    if n_a < 1 or n_b < 1:
        return None

    mean_a = statistics.mean(sample_a)
    mean_b = statistics.mean(sample_b)

    if n_a > 1:
        var_a = statistics.variance(sample_a)
    else:
        var_a = 0.0

    if n_b > 1:
        var_b = statistics.variance(sample_b)
    else:
        var_b = 0.0

    # Pooled standard deviation
    if n_a + n_b > 2:
        pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        pooled_std = math.sqrt(pooled_var)
    else:
        pooled_std = math.sqrt((var_a + var_b) / 2.0) if var_a + var_b > 0 else 1.0

    if pooled_std == 0:
        return 0.0 if mean_a == mean_b else float("inf")

    return (mean_a - mean_b) / pooled_std


def statistical_power(
    sample_a: list[float],
    sample_b: list[float],
    alpha: float = 0.05,
    effect_size: float | None = None,
) -> float | None:
    """Estimate statistical power of a test.

    Args:
        sample_a: First sample values.
        sample_b: Second sample values.
        alpha: Significance level (default 0.05).
        effect_size: Pre-specified effect size (optional).

    Returns:
        Estimated power (0.0-1.0) or None if insufficient data.
    """
    n_a, n_b = len(sample_a), len(sample_b)
    if n_a < 2 or n_b < 2:
        return None

    # Calculate effect size if not provided
    if effect_size is None:
        d = effect_size_cohens_d(sample_a, sample_b)
        if d is None:
            return None
        effect_size_abs = abs(d)
    else:
        effect_size_abs = abs(effect_size)

    # Simplified power calculation
    # For more accurate power analysis, SciPy is recommended
    n = min(n_a, n_b)  # Conservative: use smaller sample size
    if SCIPY_AVAILABLE:
        try:
            power = stats.power.tt_ind_solve_power(
                effect_size=effect_size_abs,
                nobs1=n,
                alpha=alpha,
                ratio=n_b / n_a if n_a > 0 else 1.0,
            )
            return float(power) if power is not None else None
        except Exception:
            # Fall back to approximation
            pass

    # Rough approximation: power ≈ 1 - β, where β decreases with effect size and n
    # This is a very rough heuristic
    approx_power = 1.0 - math.exp(-effect_size_abs * math.sqrt(n) / 2.0)
    return max(0.0, min(1.0, approx_power))


def sample_size_for_power(
    effect_size: float,
    power: float = 0.8,
    alpha: float = 0.05,
    ratio: float = 1.0,
) -> int:
    """Calculate required sample size per group for desired power.

    Args:
        effect_size: Expected Cohen's d effect size.
        power: Desired statistical power (default 0.8).
        alpha: Significance level (default 0.05).
        ratio: n2/n1 ratio (default 1.0 for equal group sizes).

    Returns:
        Required sample size for first group.
    """
    if SCIPY_AVAILABLE:
        try:
            n = stats.power.tt_ind_solve_power(
                effect_size=effect_size,
                power=power,
                alpha=alpha,
                ratio=ratio,
            )
            return int(math.ceil(n)) if n is not None else 100
        except Exception:
            pass

    # Approximation formula for two-sample t-test
    # n ≈ ( (Z_{1-α/2} + Z_{1-β})^2 * (σ1² + σ2²/κ) ) / δ²
    # Simplified assuming σ1=σ2=1 (standardized) and κ=ratio
    # For equal variances and ratio=1: n ≈ 16 / d² for 80% power, α=0.05
    z_alpha = 1.96  # For α=0.05, two-tailed
    z_beta = 0.84  # For power=0.8
    if power != 0.8:
        # Approximate Z for different power levels
        z_beta = {0.7: 0.52, 0.8: 0.84, 0.9: 1.28}.get(power, 0.84)

    n_per_group = ((z_alpha + z_beta) ** 2 * (1 + 1 / ratio)) / (effect_size**2)
    return max(10, int(math.ceil(n_per_group)))


def mann_whitney_u_test(
    sample_a: list[float],
    sample_b: list[float],
) -> dict[str, Any]:
    """Perform Mann-Whitney U test (non-parametric).

    Args:
        sample_a: First sample values.
        sample_b: Second sample values.

    Returns:
        Dictionary with U statistic, p-value, and effect size.
    """
    if SCIPY_AVAILABLE:
        u_stat, p_value = stats.mannwhitneyu(sample_a, sample_b, alternative="two-sided")
        return {
            "u_statistic": float(u_stat),
            "p_value": float(p_value),
            "method": "scipy_mannwhitneyu",
        }

    # Fallback pure Python implementation
    n_a, n_b = len(sample_a), len(sample_b)
    if n_a == 0 or n_b == 0:
        return {
            "u_statistic": None,
            "p_value": None,
            "method": "insufficient_data",
            "error": "Both samples must have at least 1 observation",
        }

    # Combine and rank all samples
    combined = [(val, "A", i) for i, val in enumerate(sample_a)]
    combined += [(val, "B", i) for i, val in enumerate(sample_b)]
    combined.sort(key=lambda x: x[0])

    # Assign ranks, handling ties
    ranks = []
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        # Tied values get average rank
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks.append((combined[k][1], combined[k][2], avg_rank))
        i = j

    # Sort ranks back to original order
    ranks.sort(key=lambda x: (x[0], x[1]))

    # Sum ranks for each group
    rank_sum_a = sum(rank for group, _, rank in ranks if group == "A")
    rank_sum_b = sum(rank for group, _, rank in ranks if group == "B")

    # Calculate U statistics
    u_a = rank_sum_a - n_a * (n_a + 1) / 2
    u_b = rank_sum_b - n_b * (n_b + 1) / 2

    u_stat = min(u_a, u_b)

    # Approximate p-value using normal approximation for large samples
    if n_a > 20 and n_b > 20:
        # Normal approximation
        mean_u = n_a * n_b / 2.0
        var_u = n_a * n_b * (n_a + n_b + 1) / 12.0
        if var_u > 0:
            z = (u_stat - mean_u) / math.sqrt(var_u)
            p_value = 2.0 * (1.0 - _normal_cdf(abs(z)))
        else:
            p_value = 1.0
    else:
        # For small samples, use conservative approximation
        # Exact p-value would require lookup table; this is approximate
        max_u = n_a * n_b
        p_value = 2.0 * (u_stat / max_u)  # Rough approximation

    return {
        "u_statistic": u_stat,
        "p_value": min(p_value, 1.0),
        "method": "pure_python_mannwhitneyu",
    }


def _normal_cdf(x: float) -> float:
    """Approximate cumulative distribution function for standard normal."""
    # Abramowitz & Stegun approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = 1 if x >= 0 else -1
    x = abs(x) / math.sqrt(2.0)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - ((((a5 * t + a4) * t) + a3) * t + a2) * t + a1 * t
    y = y * math.exp(-x * x)

    return 0.5 * (1.0 + sign * y)


def analyze_variant_comparison(
    variant_a_metrics: dict[str, float] | list[float],
    variant_b_metrics: dict[str, float] | list[float],
    test_type: str = "t-test",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Analyze comparison between two variants for a single metric.

    Args:
        variant_a_metrics: List of metric values for variant A.
        variant_b_metrics: List of metric values for variant B.
        test_type: Type of statistical test ("t-test" or "mann-whitney").
        alpha: Significance level.

    Returns:
        Comprehensive analysis results.
    """
    # Convert dict to list if needed (for backward compatibility)
    if isinstance(variant_a_metrics, dict):
        # Assume dict contains raw values
        sample_a = list(variant_a_metrics.values())
    else:
        sample_a = variant_a_metrics

    if isinstance(variant_b_metrics, dict):
        sample_b = list(variant_b_metrics.values())
    else:
        sample_b = variant_b_metrics

    if not sample_a or not sample_b:
        return {
            "error": "Insufficient data for analysis",
            "sample_sizes": {"a": len(sample_a), "b": len(sample_b)},
        }

    # Perform statistical test
    if test_type.lower() in ["t-test", "t_test", "ttest"]:
        test_result = independent_t_test(sample_a, sample_b)
    elif test_type.lower() in ["mann-whitney", "mannwhitney", "u-test"]:
        test_result = mann_whitney_u_test(sample_a, sample_b)
    else:
        return {"error": f"Unknown test type: {test_type}"}

    # Calculate additional statistics
    ci_a = confidence_interval(sample_a, 1.0 - alpha)
    ci_b = confidence_interval(sample_b, 1.0 - alpha)
    effect_size = effect_size_cohens_d(sample_a, sample_b)
    power = statistical_power(sample_a, sample_b, alpha, effect_size)

    # Determine if significant
    is_significant = (
        test_result.get("p_value") is not None
        and test_result["p_value"] < alpha
        and test_result.get("method") != "insufficient_data"
    )

    return {
        "test_type": test_type,
        "sample_sizes": {"a": len(sample_a), "b": len(sample_b)},
        "means": {
            "a": statistics.mean(sample_a) if sample_a else None,
            "b": statistics.mean(sample_b) if sample_b else None,
        },
        "standard_deviations": {
            "a": statistics.stdev(sample_a) if len(sample_a) > 1 else 0.0,
            "b": statistics.stdev(sample_b) if len(sample_b) > 1 else 0.0,
        },
        "confidence_intervals": {"a": ci_a, "b": ci_b},
        "test_result": test_result,
        "effect_size": effect_size,
        "statistical_power": power,
        "is_significant": is_significant,
        "p_value": test_result.get("p_value"),
        "recommendation": _generate_recommendation(
            is_significant,
            test_result.get("p_value"),
            effect_size,
            len(sample_a) + len(sample_b),
        ),
    }


def _generate_recommendation(
    is_significant: bool,
    p_value: float | None,
    effect_size: float | None,
    total_samples: int,
) -> str:
    """Generate recommendation based on analysis results."""
    if p_value is None or effect_size is None:
        return "Insufficient data for recommendation"

    if not is_significant:
        if total_samples < 100:
            return "Continue experiment: insufficient sample size"
        elif p_value < 0.1:  # Marginal significance
            return "Continue experiment: marginal significance detected"
        else:
            return "Consider stopping: no significant difference detected"

    # Significant result
    if abs(effect_size) < 0.2:
        return "Statistically significant but effect size is small"
    elif abs(effect_size) < 0.5:
        return "Moderate effect size detected"
    elif abs(effect_size) < 0.8:
        return "Large effect size detected"
    else:
        return "Very large effect size detected"
