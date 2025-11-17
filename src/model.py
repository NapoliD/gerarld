"""
Recommendation model for Olist dataset.
Implements baseline popularity and collaborative filtering approaches.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import pickle
from pathlib import Path
from collections import defaultdict


class OlistRecommender:
    """Product recommendation system for Olist customers."""

    def __init__(self, loader):
        """
        Initialize recommender with data loader.

        Args:
            loader: OlistDataLoader instance with loaded data.
        """
        self.loader = loader
        self.popularity_scores = None
        self.co_purchase_matrix = None
        self.product_to_idx = None
        self.idx_to_product = None
        self.is_fitted = False

    def fit(self, min_support: int = 5):
        """
        Fit the recommendation model.

        Args:
            min_support: Minimum number of purchases for a product to be considered.
        """
        print("Fitting recommender model...")

        # Build popularity baseline
        self._build_popularity_model()

        # Build co-purchase matrix
        self._build_copurchase_model(min_support=min_support)

        self.is_fitted = True
        print("Model fitted successfully!")

    def _build_popularity_model(self):
        """Build popularity-based recommendation (baseline)."""
        # Count purchases per product
        product_counts = (
            self.loader.order_items
            .groupby('product_id')
            .size()
            .reset_index(name='purchase_count')
        )

        # Merge with product info
        if self.loader.products is not None:
            product_counts = product_counts.merge(
                self.loader.products[['product_id', 'product_category_name']],
                on='product_id',
                how='left'
            )

        # Sort by popularity
        self.popularity_scores = product_counts.sort_values(
            'purchase_count',
            ascending=False
        )

        print(f"  Built popularity model with {len(self.popularity_scores)} products")

    def _build_copurchase_model(self, min_support: int = 5):
        """
        Build co-purchase matrix (products bought together).

        Args:
            min_support: Minimum number of co-occurrences to consider.
        """
        # Get products per order
        order_products = (
            self.loader.order_items
            .groupby('order_id')['product_id']
            .apply(list)
            .reset_index()
        )

        # Filter out single-item orders for co-purchase
        order_products = order_products[
            order_products['product_id'].apply(len) > 1
        ]

        # Build co-occurrence matrix
        co_occur = defaultdict(lambda: defaultdict(int))

        for products in order_products['product_id']:
            # For each pair of products in the same order
            for i, prod1 in enumerate(products):
                for prod2 in products[i+1:]:
                    co_occur[prod1][prod2] += 1
                    co_occur[prod2][prod1] += 1

        # Filter by min_support and convert to DataFrame
        copurchase_data = []
        for prod1, prod2_dict in co_occur.items():
            for prod2, count in prod2_dict.items():
                if count >= min_support:
                    copurchase_data.append({
                        'product_id_1': prod1,
                        'product_id_2': prod2,
                        'co_purchase_count': count
                    })

        if len(copurchase_data) > 0:
            self.co_purchase_matrix = pd.DataFrame(copurchase_data)
            print(f"  Built co-purchase model with {len(self.co_purchase_matrix)} product pairs")
        else:
            self.co_purchase_matrix = pd.DataFrame(columns=['product_id_1', 'product_id_2', 'co_purchase_count'])
            print("  Warning: No co-purchase pairs found with minimum support")

    def recommend(
        self,
        customer_id: str,
        top_k: int = 5,
        method: str = 'hybrid'
    ) -> List[Dict]:
        """
        Generate product recommendations for a customer.

        Args:
            customer_id: Customer ID to recommend for.
            top_k: Number of recommendations to return.
            method: Recommendation method ('popularity', 'copurchase', or 'hybrid').

        Returns:
            List of recommended product dictionaries.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations. Call fit() first.")

        # Get customer's purchase history
        customer_orders = self.loader.orders[
            self.loader.orders['customer_id'] == customer_id
        ]

        if len(customer_orders) == 0:
            # New customer - use popularity
            return self._recommend_popular(top_k)

        # Get products the customer has already purchased
        order_ids = customer_orders['order_id'].tolist()
        purchased_products = self.loader.order_items[
            self.loader.order_items['order_id'].isin(order_ids)
        ]['product_id'].unique()

        if method == 'popularity':
            return self._recommend_popular(top_k, exclude=purchased_products)
        elif method == 'copurchase':
            return self._recommend_copurchase(purchased_products, top_k, exclude=purchased_products)
        elif method == 'hybrid':
            return self._recommend_hybrid(purchased_products, top_k, exclude=purchased_products)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _recommend_popular(self, top_k: int, exclude: np.ndarray = None) -> List[Dict]:
        """
        Recommend based on popularity.

        Args:
            top_k: Number of recommendations.
            exclude: Products to exclude from recommendations.

        Returns:
            List of recommended product dictionaries.
        """
        recommendations = self.popularity_scores.copy()

        # Exclude already purchased products
        if exclude is not None and len(exclude) > 0:
            recommendations = recommendations[
                ~recommendations['product_id'].isin(exclude)
            ]

        # Get top K
        recommendations = recommendations.head(top_k)

        return recommendations[['product_id', 'product_category_name', 'purchase_count']].to_dict('records')

    def _recommend_copurchase(
        self,
        purchased_products: np.ndarray,
        top_k: int,
        exclude: np.ndarray = None
    ) -> List[Dict]:
        """
        Recommend based on co-purchase patterns.

        Args:
            purchased_products: Products the customer has purchased.
            top_k: Number of recommendations.
            exclude: Products to exclude from recommendations.

        Returns:
            List of recommended product dictionaries.
        """
        if self.co_purchase_matrix is None or len(self.co_purchase_matrix) == 0:
            # Fallback to popularity if no co-purchase data
            return self._recommend_popular(top_k, exclude)

        # Find products co-purchased with customer's products
        copurchased = self.co_purchase_matrix[
            self.co_purchase_matrix['product_id_1'].isin(purchased_products)
        ]

        if len(copurchased) == 0:
            # No co-purchase data for this customer - use popularity
            return self._recommend_popular(top_k, exclude)

        # Aggregate scores for recommended products
        scores = (
            copurchased
            .groupby('product_id_2')['co_purchase_count']
            .sum()
            .reset_index()
            .rename(columns={'product_id_2': 'product_id', 'co_purchase_count': 'score'})
            .sort_values('score', ascending=False)
        )

        # Exclude already purchased products
        if exclude is not None and len(exclude) > 0:
            scores = scores[~scores['product_id'].isin(exclude)]

        # Get top K
        scores = scores.head(top_k)

        # Add product info
        if self.loader.products is not None:
            scores = scores.merge(
                self.loader.products[['product_id', 'product_category_name']],
                on='product_id',
                how='left'
            )

        return scores.to_dict('records')

    def _recommend_hybrid(
        self,
        purchased_products: np.ndarray,
        top_k: int,
        exclude: np.ndarray = None
    ) -> List[Dict]:
        """
        Hybrid recommendation combining co-purchase and popularity.

        Args:
            purchased_products: Products the customer has purchased.
            top_k: Number of recommendations.
            exclude: Products to exclude from recommendations.

        Returns:
            List of recommended product dictionaries.
        """
        # Get co-purchase recommendations
        copurchase_recs = self._recommend_copurchase(
            purchased_products,
            top_k=top_k * 2,  # Get more to have options
            exclude=exclude
        )

        # If we have enough co-purchase recommendations, use them
        if len(copurchase_recs) >= top_k:
            return copurchase_recs[:top_k]

        # Otherwise, fill with popularity
        copurchase_product_ids = [rec['product_id'] for rec in copurchase_recs]
        exclude_extended = list(exclude) + copurchase_product_ids if exclude is not None else copurchase_product_ids

        popularity_recs = self._recommend_popular(
            top_k=top_k - len(copurchase_recs),
            exclude=np.array(exclude_extended)
        )

        return copurchase_recs + popularity_recs

    def save(self, filepath: str):
        """Save model to file."""
        model_data = {
            'popularity_scores': self.popularity_scores,
            'co_purchase_matrix': self.co_purchase_matrix,
            'is_fitted': self.is_fitted
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.popularity_scores = model_data['popularity_scores']
        self.co_purchase_matrix = model_data['co_purchase_matrix']
        self.is_fitted = model_data['is_fitted']

        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.append('..')
    from data_loader import OlistDataLoader

    loader = OlistDataLoader(data_dir='../datos')
    loader.load_all()

    # Train model
    model = OlistRecommender(loader)
    model.fit(min_support=3)

    # Test recommendation
    sample_customer = loader.orders['customer_id'].iloc[100]
    print(f"\nTesting recommendations for customer: {sample_customer}")

    recommendations = model.recommend(sample_customer, top_k=5, method='hybrid')
    print("\nTop 5 Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. Product: {rec['product_id']}, Category: {rec.get('product_category_name', 'N/A')}")

    # Save model
    model.save('../models/recommender.pkl')
