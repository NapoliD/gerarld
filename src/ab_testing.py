"""
A/B Testing framework for recommendation experiments.
"""
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, List
from scipy import stats


class ABTest:
    """A/B testing framework for comparing recommendation methods."""

    def __init__(self, experiment_name: str, variants: List[str]):
        """
        Initialize A/B test.

        Args:
            experiment_name: Name of the experiment.
            variants: List of variant names (e.g., ['control', 'treatment']).
        """
        self.experiment_name = experiment_name
        self.variants = variants
        self.results = {variant: [] for variant in variants}

    def assign_variant(self, user_id: str) -> str:
        """
        Assign user to variant using consistent hashing.

        Args:
            user_id: User ID.

        Returns:
            Variant name.
        """
        # Use hash for consistent assignment
        hash_val = int(hashlib.md5(f"{user_id}_{self.experiment_name}".encode()).hexdigest(), 16)
        variant_idx = hash_val % len(self.variants)

        return self.variants[variant_idx]

    def track_conversion(self, user_id: str, variant: str, converted: bool, value: float = 0):
        """
        Track conversion for a user.

        Args:
            user_id: User ID.
            variant: Variant name.
            converted: Whether user converted.
            value: Optional conversion value.
        """
        self.results[variant].append({
            'user_id': user_id,
            'converted': converted,
            'value': value
        })

    def get_results(self) -> Dict:
        """
        Get experiment results.

        Returns:
            Dictionary with results per variant.
        """
        summary = {}

        for variant in self.variants:
            data = self.results[variant]

            if not data:
                summary[variant] = {
                    'users': 0,
                    'conversions': 0,
                    'conversion_rate': 0.0,
                    'avg_value': 0.0
                }
                continue

            df = pd.DataFrame(data)

            summary[variant] = {
                'users': len(df),
                'conversions': df['converted'].sum(),
                'conversion_rate': df['converted'].mean(),
                'avg_value': df['value'].mean(),
                'total_value': df['value'].sum()
            }

        return summary

    def statistical_significance(
        self,
        variant_a: str,
        variant_b: str,
        alpha: float = 0.05
    ) -> Dict:
        """
        Test statistical significance between two variants.

        Args:
            variant_a: First variant name.
            variant_b: Second variant name.
            alpha: Significance level.

        Returns:
            Statistical test results.
        """
        data_a = pd.DataFrame(self.results[variant_a])
        data_b = pd.DataFrame(self.results[variant_b])

        if len(data_a) == 0 or len(data_b) == 0:
            return {'error': 'Insufficient data'}

        # Conversion rate test (chi-square or z-test)
        conversions_a = data_a['converted'].sum()
        conversions_b = data_b['converted'].sum()
        total_a = len(data_a)
        total_b = len(data_b)

        # Z-test for proportions
        p_a = conversions_a / total_a
        p_b = conversions_b / total_b
        p_pooled = (conversions_a + conversions_b) / (total_a + total_b)

        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/total_a + 1/total_b))

        if se == 0:
            z_score = 0
        else:
            z_score = (p_b - p_a) / se

        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        # Effect size (relative uplift)
        uplift = ((p_b - p_a) / p_a * 100) if p_a > 0 else 0

        # Confidence interval (95%)
        ci = 1.96 * se
        ci_lower = (p_b - p_a) - ci
        ci_upper = (p_b - p_a) + ci

        return {
            'variant_a': variant_a,
            'variant_b': variant_b,
            'conversion_rate_a': p_a,
            'conversion_rate_b': p_b,
            'uplift_percent': uplift,
            'z_score': z_score,
            'p_value': p_value,
            'is_significant': p_value < alpha,
            'confidence_interval': (ci_lower, ci_upper),
            'sample_size_a': total_a,
            'sample_size_b': total_b
        }

    def print_summary(self):
        """Print experiment summary."""
        print("="*70)
        print(f"A/B TEST RESULTS: {self.experiment_name}")
        print("="*70)

        results = self.get_results()

        for variant in self.variants:
            r = results[variant]
            print(f"\n{variant.upper()}:")
            print(f"  Users: {r['users']:,}")
            print(f"  Conversions: {r['conversions']:,}")
            print(f"  Conversion Rate: {r['conversion_rate']:.2%}")
            print(f"  Avg Value: ${r['avg_value']:.2f}")
            print(f"  Total Value: ${r['total_value']:.2f}")

        # Statistical test (if 2 variants)
        if len(self.variants) == 2:
            sig = self.statistical_significance(self.variants[0], self.variants[1])

            if 'error' not in sig:
                print("\n" + "="*70)
                print("STATISTICAL SIGNIFICANCE TEST")
                print("="*70)
                print(f"\nUplift: {sig['uplift_percent']:.2f}%")
                print(f"P-value: {sig['p_value']:.4f}")
                print(f"Significant at α=0.05: {'YES' if sig['is_significant'] else 'NO'}")


class ExperimentTracker:
    """Track multiple experiments."""

    def __init__(self):
        """Initialize experiment tracker."""
        self.experiments = {}

    def create_experiment(self, name: str, variants: List[str]) -> ABTest:
        """
        Create new experiment.

        Args:
            name: Experiment name.
            variants: List of variant names.

        Returns:
            ABTest instance.
        """
        experiment = ABTest(name, variants)
        self.experiments[name] = experiment

        return experiment

    def get_experiment(self, name: str) -> ABTest:
        """Get experiment by name."""
        return self.experiments.get(name)

    def list_experiments(self) -> List[str]:
        """List all experiments."""
        return list(self.experiments.keys())


if __name__ == "__main__":
    # Example A/B test
    experiment = ABTest(
        experiment_name="recommendation_algorithm_v2",
        variants=['control', 'treatment']
    )

    # Simulate some data
    np.random.seed(42)

    for i in range(1000):
        user_id = f"user_{i}"
        variant = experiment.assign_variant(user_id)

        # Simulate conversions (treatment has slightly higher rate)
        if variant == 'control':
            converted = np.random.random() < 0.05
        else:
            converted = np.random.random() < 0.06  # 20% uplift

        value = np.random.uniform(50, 200) if converted else 0

        experiment.track_conversion(user_id, variant, converted, value)

    # Print results
    experiment.print_summary()
