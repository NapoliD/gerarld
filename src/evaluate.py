"""
Evaluation metrics for recommendation model.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split


class RecommenderEvaluator:
    """Evaluate recommendation model performance."""

    def __init__(self, loader, model):
        """
        Initialize evaluator.

        Args:
            loader: OlistDataLoader instance.
            model: OlistRecommender instance.
        """
        self.loader = loader
        self.model = model

    def train_test_split_by_time(
        self,
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split orders into train/test by time.

        Args:
            test_size: Fraction of recent orders for testing.

        Returns:
            Tuple of (train_orders, test_orders).
        """
        orders = self.loader.orders.copy()
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
        orders = orders.sort_values('order_purchase_timestamp')

        # Split by time
        split_idx = int(len(orders) * (1 - test_size))
        train_orders = orders.iloc[:split_idx]
        test_orders = orders.iloc[split_idx:]

        return train_orders, test_orders

    def precision_at_k(
        self,
        recommendations: List[str],
        ground_truth: List[str],
        k: int = 5
    ) -> float:
        """
        Calculate Precision@K.

        Args:
            recommendations: List of recommended product IDs.
            ground_truth: List of actual purchased product IDs.
            k: Number of recommendations to consider.

        Returns:
            Precision@K score.
        """
        if len(recommendations) == 0 or len(ground_truth) == 0:
            return 0.0

        # Take top K recommendations
        recs_at_k = recommendations[:k]

        # Count hits
        hits = len(set(recs_at_k) & set(ground_truth))

        # Precision@K = hits / k
        return hits / k

    def recall_at_k(
        self,
        recommendations: List[str],
        ground_truth: List[str],
        k: int = 5
    ) -> float:
        """
        Calculate Recall@K.

        Args:
            recommendations: List of recommended product IDs.
            ground_truth: List of actual purchased product IDs.
            k: Number of recommendations to consider.

        Returns:
            Recall@K score.
        """
        if len(recommendations) == 0 or len(ground_truth) == 0:
            return 0.0

        # Take top K recommendations
        recs_at_k = recommendations[:k]

        # Count hits
        hits = len(set(recs_at_k) & set(ground_truth))

        # Recall@K = hits / total relevant items
        return hits / len(ground_truth)

    def evaluate_model(
        self,
        test_orders: pd.DataFrame,
        k: int = 5,
        method: str = 'hybrid'
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            test_orders: DataFrame of test orders.
            k: Number of recommendations.
            method: Recommendation method to use.

        Returns:
            Dictionary of evaluation metrics.
        """
        print(f"Evaluating model with method='{method}' and k={k}...")

        precisions = []
        recalls = []
        evaluated_customers = 0

        # Get unique customers in test set who had prior purchases
        test_customers = test_orders['customer_id'].unique()

        for customer_id in test_customers:
            # Get customer's test purchases (ground truth)
            customer_test_orders = test_orders[test_orders['customer_id'] == customer_id]
            test_order_ids = customer_test_orders['order_id'].tolist()

            ground_truth_products = self.loader.order_items[
                self.loader.order_items['order_id'].isin(test_order_ids)
            ]['product_id'].unique().tolist()

            if len(ground_truth_products) == 0:
                continue

            # Generate recommendations
            try:
                recommendations = self.model.recommend(
                    customer_id,
                    top_k=k,
                    method=method
                )
                rec_product_ids = [rec['product_id'] for rec in recommendations]

                # Calculate metrics
                precision = self.precision_at_k(rec_product_ids, ground_truth_products, k)
                recall = self.recall_at_k(rec_product_ids, ground_truth_products, k)

                precisions.append(precision)
                recalls.append(recall)
                evaluated_customers += 1

            except Exception as e:
                # Skip customers that cause errors
                continue

        if evaluated_customers == 0:
            print("Warning: No customers could be evaluated")
            return {
                'precision_at_k': 0.0,
                'recall_at_k': 0.0,
                'evaluated_customers': 0
            }

        # Calculate average metrics
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)

        print(f"  Evaluated {evaluated_customers} customers")
        print(f"  Precision@{k}: {avg_precision:.4f}")
        print(f"  Recall@{k}: {avg_recall:.4f}")

        return {
            'precision_at_k': avg_precision,
            'recall_at_k': avg_recall,
            'evaluated_customers': evaluated_customers
        }

    def evaluate_baseline_vs_improved(self, test_orders: pd.DataFrame, k: int = 5):
        """
        Compare baseline (popularity) vs improved (hybrid) model.

        Args:
            test_orders: DataFrame of test orders.
            k: Number of recommendations.
        """
        print("\n" + "="*70)
        print("MODEL EVALUATION COMPARISON")
        print("="*70)

        # Evaluate baseline (popularity)
        print("\nBASELINE MODEL (Popularity):")
        baseline_metrics = self.evaluate_model(test_orders, k=k, method='popularity')

        # Evaluate improved (hybrid)
        print("\nIMPROVED MODEL (Hybrid: Co-purchase + Popularity):")
        improved_metrics = self.evaluate_model(test_orders, k=k, method='hybrid')

        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"\nBaseline Precision@{k}: {baseline_metrics['precision_at_k']:.4f}")
        print(f"Improved Precision@{k}: {improved_metrics['precision_at_k']:.4f}")
        print(f"Improvement: {(improved_metrics['precision_at_k'] - baseline_metrics['precision_at_k']):.4f}")

        print(f"\nBaseline Recall@{k}: {baseline_metrics['recall_at_k']:.4f}")
        print(f"Improved Recall@{k}: {improved_metrics['recall_at_k']:.4f}")
        print(f"Improvement: {(improved_metrics['recall_at_k'] - baseline_metrics['recall_at_k']):.4f}")

        return baseline_metrics, improved_metrics


if __name__ == "__main__":
    # Test evaluation
    import sys
    sys.path.append('..')
    from data_loader import OlistDataLoader
    from model import OlistRecommender

    print("Loading data...")
    loader = OlistDataLoader(data_dir='../datos')
    loader.load_all()

    print("\nSplitting train/test...")
    evaluator = RecommenderEvaluator(loader, None)
    train_orders, test_orders = evaluator.train_test_split_by_time(test_size=0.2)
    print(f"Train orders: {len(train_orders)}, Test orders: {len(test_orders)}")

    # Create temporary loader with only training data for model fitting
    train_loader = OlistDataLoader(data_dir='../datos')
    train_loader.load_all()
    train_loader.orders = train_orders

    print("\nTraining model on training set...")
    model = OlistRecommender(train_loader)
    model.fit(min_support=3)

    # Now evaluate on test set (but use full loader for getting product info)
    model.loader = loader
    evaluator = RecommenderEvaluator(loader, model)
    baseline_metrics, improved_metrics = evaluator.evaluate_baseline_vs_improved(test_orders, k=5)
