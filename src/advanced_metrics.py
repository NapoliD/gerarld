"""
Advanced evaluation metrics for recommendations.
Includes MAP@K, NDCG@K, Coverage, Diversity, and more.
"""
import numpy as np
import pandas as pd
from typing import List, Dict


def average_precision_at_k(recommendations: List, ground_truth: List, k: int = 5) -> float:
    """
    Calculate Average Precision at K.

    Args:
        recommendations: List of recommended items.
        ground_truth: List of relevant items.
        k: Number of recommendations to consider.

    Returns:
        AP@K score.
    """
    if len(ground_truth) == 0:
        return 0.0

    recommendations = recommendations[:k]
    score = 0.0
    num_hits = 0.0

    for i, rec in enumerate(recommendations):
        if rec in ground_truth:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(ground_truth), k)


def mean_average_precision_at_k(
    all_recommendations: List[List],
    all_ground_truth: List[List],
    k: int = 5
) -> float:
    """
    Calculate Mean Average Precision at K across all users.

    Args:
        all_recommendations: List of recommendation lists.
        all_ground_truth: List of ground truth lists.
        k: Number of recommendations to consider.

    Returns:
        MAP@K score.
    """
    scores = []

    for recs, truth in zip(all_recommendations, all_ground_truth):
        scores.append(average_precision_at_k(recs, truth, k))

    return np.mean(scores) if scores else 0.0


def dcg_at_k(relevances: List[float], k: int = 5) -> float:
    """
    Calculate Discounted Cumulative Gain at K.

    Args:
        relevances: List of relevance scores (binary or graded).
        k: Number of recommendations to consider.

    Returns:
        DCG@K score.
    """
    relevances = np.array(relevances)[:k]
    if len(relevances) == 0:
        return 0.0

    # DCG formula: sum(rel_i / log2(i + 1))
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return np.sum(relevances / discounts)


def ndcg_at_k(recommendations: List, ground_truth: List, k: int = 5) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.

    Args:
        recommendations: List of recommended items.
        ground_truth: List of relevant items.
        k: Number of recommendations to consider.

    Returns:
        NDCG@K score.
    """
    if len(ground_truth) == 0:
        return 0.0

    recommendations = recommendations[:k]

    # Binary relevance: 1 if recommended item is relevant, 0 otherwise
    relevances = [1 if rec in ground_truth else 0 for rec in recommendations]

    # Ideal DCG (all relevant items at top)
    ideal_relevances = [1] * min(len(ground_truth), k)

    dcg = dcg_at_k(relevances, k)
    idcg = dcg_at_k(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0


def catalog_coverage(
    all_recommendations: List[List],
    total_items: int
) -> float:
    """
    Calculate catalog coverage (% of items recommended).

    Args:
        all_recommendations: List of recommendation lists.
        total_items: Total number of items in catalog.

    Returns:
        Coverage percentage (0-1).
    """
    unique_recommended = set()

    for recs in all_recommendations:
        unique_recommended.update(recs)

    return len(unique_recommended) / total_items


def diversity_score(
    recommendations: List,
    item_categories: Dict[str, str]
) -> float:
    """
    Calculate diversity of recommendations (category-based).

    Args:
        recommendations: List of recommended items.
        item_categories: Dictionary mapping item IDs to categories.

    Returns:
        Diversity score (0-1). Higher = more diverse.
    """
    if len(recommendations) == 0:
        return 0.0

    categories = [item_categories.get(item, 'unknown') for item in recommendations]
    unique_categories = len(set(categories))

    return unique_categories / len(recommendations)


def novelty_score(
    recommendations: List,
    item_popularity: Dict[str, int],
    total_interactions: int
) -> float:
    """
    Calculate novelty score (recommends less popular items).

    Args:
        recommendations: List of recommended items.
        item_popularity: Dictionary of item purchase counts.
        total_interactions: Total number of purchases.

    Returns:
        Novelty score. Higher = more novel (less popular items).
    """
    if len(recommendations) == 0:
        return 0.0

    novelty = 0.0

    for item in recommendations:
        popularity = item_popularity.get(item, 0)
        probability = popularity / total_interactions if total_interactions > 0 else 0

        # Self-information: -log2(p)
        if probability > 0:
            novelty += -np.log2(probability)

    return novelty / len(recommendations)


class AdvancedEvaluator:
    """Comprehensive evaluator with advanced metrics."""

    def __init__(self, loader, model):
        """
        Initialize evaluator.

        Args:
            loader: OlistDataLoader instance.
            model: Recommender model instance.
        """
        self.loader = loader
        self.model = model

    def evaluate_comprehensive(
        self,
        test_orders: pd.DataFrame,
        k: int = 5,
        method: str = 'hybrid'
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation with multiple metrics.

        Args:
            test_orders: Test orders DataFrame.
            k: Number of recommendations.
            method: Recommendation method.

        Returns:
            Dictionary of all metrics.
        """
        print(f"Comprehensive evaluation (k={k}, method={method})...")

        all_recommendations = []
        all_ground_truth = []
        evaluated = 0

        # Get unique customers in test set
        test_customers = test_orders.merge(
            self.loader.customers[['customer_id', 'customer_unique_id']],
            on='customer_id'
        )['customer_unique_id'].unique()

        for customer_id in test_customers[:1000]:  # Limit for speed
            # Get ground truth
            customer_test = test_orders.merge(
                self.loader.customers[['customer_id', 'customer_unique_id']],
                on='customer_id'
            )
            customer_test = customer_test[customer_test['customer_unique_id'] == customer_id]

            if len(customer_test) == 0:
                continue

            test_order_ids = customer_test['order_id'].tolist()
            ground_truth = self.loader.order_items[
                self.loader.order_items['order_id'].isin(test_order_ids)
            ]['product_id'].unique().tolist()

            if len(ground_truth) == 0:
                continue

            # Get recommendations
            try:
                recs = self.model.recommend(customer_id, top_k=k, method=method)
                rec_ids = [r['product_id'] for r in recs]

                all_recommendations.append(rec_ids)
                all_ground_truth.append(ground_truth)
                evaluated += 1

            except:
                continue

        if evaluated == 0:
            return {}

        # Calculate metrics
        metrics = {}

        # Precision & Recall
        precisions = []
        recalls = []

        for recs, truth in zip(all_recommendations, all_ground_truth):
            hits = len(set(recs) & set(truth))
            precisions.append(hits / len(recs) if recs else 0)
            recalls.append(hits / len(truth) if truth else 0)

        metrics['precision@k'] = np.mean(precisions)
        metrics['recall@k'] = np.mean(recalls)

        # F1 Score
        if metrics['precision@k'] + metrics['recall@k'] > 0:
            metrics['f1@k'] = (
                2 * metrics['precision@k'] * metrics['recall@k'] /
                (metrics['precision@k'] + metrics['recall@k'])
            )
        else:
            metrics['f1@k'] = 0.0

        # MAP@K
        metrics['map@k'] = mean_average_precision_at_k(
            all_recommendations,
            all_ground_truth,
            k
        )

        # NDCG@K
        ndcg_scores = [
            ndcg_at_k(recs, truth, k)
            for recs, truth in zip(all_recommendations, all_ground_truth)
        ]
        metrics['ndcg@k'] = np.mean(ndcg_scores)

        # Coverage
        total_products = len(self.loader.products)
        metrics['coverage'] = catalog_coverage(all_recommendations, total_products)

        # Diversity
        product_categories = dict(
            zip(
                self.loader.products['product_id'],
                self.loader.products['product_category_name']
            )
        )

        diversity_scores = [
            diversity_score(recs, product_categories)
            for recs in all_recommendations
        ]
        metrics['diversity'] = np.mean(diversity_scores)

        # Novelty
        item_popularity = (
            self.loader.order_items
            .groupby('product_id')
            .size()
            .to_dict()
        )
        total_interactions = len(self.loader.order_items)

        novelty_scores = [
            novelty_score(recs, item_popularity, total_interactions)
            for recs in all_recommendations
        ]
        metrics['novelty'] = np.mean(novelty_scores)

        metrics['evaluated_customers'] = evaluated

        # Print results
        print(f"\nEvaluated {evaluated} customers:")
        print(f"  Precision@{k}: {metrics['precision@k']:.4f}")
        print(f"  Recall@{k}: {metrics['recall@k']:.4f}")
        print(f"  F1@{k}: {metrics['f1@k']:.4f}")
        print(f"  MAP@{k}: {metrics['map@k']:.4f}")
        print(f"  NDCG@{k}: {metrics['ndcg@k']:.4f}")
        print(f"  Coverage: {metrics['coverage']:.4f}")
        print(f"  Diversity: {metrics['diversity']:.4f}")
        print(f"  Novelty: {metrics['novelty']:.2f}")

        return metrics


if __name__ == "__main__":
    # Test metrics
    recs = ['A', 'B', 'C', 'D', 'E']
    truth = ['A', 'C', 'F']

    print(f"Recommendations: {recs}")
    print(f"Ground truth: {truth}")
    print(f"AP@5: {average_precision_at_k(recs, truth, 5):.4f}")
    print(f"NDCG@5: {ndcg_at_k(recs, truth, 5):.4f}")
