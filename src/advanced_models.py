"""
Advanced recommendation models with multiple algorithms.
Includes SVD, content-based, temporal, ensemble, and more.
"""
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedRecommender:
    """Advanced recommendation system with multiple algorithms."""

    def __init__(self, loader):
        """
        Initialize advanced recommender.

        Args:
            loader: OlistDataLoader instance.
        """
        self.loader = loader
        self.user_item_matrix = None
        self.svd_U = None
        self.svd_sigma = None
        self.svd_Vt = None
        self.product_features = None
        self.content_similarity = None
        self.temporal_weights = None
        self.is_fitted = False

    def _build_user_item_matrix(self):
        """Build sparse user-item interaction matrix."""
        # Merge to get customer-product pairs
        items = self.loader.order_items.merge(
            self.loader.orders[['order_id', 'customer_id']],
            on='order_id',
            how='left'
        )

        items = items.merge(
            self.loader.customers[['customer_id', 'customer_unique_id']],
            on='customer_id',
            how='left'
        )

        # Create user-product interaction (purchase count)
        interactions = (
            items.groupby(['customer_unique_id', 'product_id'])
            .size()
            .reset_index(name='purchases')
        )

        # Create mappings
        self.user_to_idx = {user: idx for idx, user in enumerate(interactions['customer_unique_id'].unique())}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.product_to_idx = {prod: idx for idx, prod in enumerate(interactions['product_id'].unique())}
        self.idx_to_product = {idx: prod for prod, idx in self.product_to_idx.items()}

        # Build sparse matrix
        row = interactions['customer_unique_id'].map(self.user_to_idx).values
        col = interactions['product_id'].map(self.product_to_idx).values
        data = interactions['purchases'].values

        self.user_item_matrix = csr_matrix(
            (data, (row, col)),
            shape=(len(self.user_to_idx), len(self.product_to_idx))
        )

        logger.info(f"Built user-item matrix: {self.user_item_matrix.shape}")

    def fit_matrix_factorization(self, n_factors: int = 50):
        """
        Fit matrix factorization using SVD.

        Args:
            n_factors: Number of latent factors.
        """
        if self.user_item_matrix is None:
            self._build_user_item_matrix()

        logger.info(f"Fitting SVD with {n_factors} factors...")

        # Perform SVD
        U, sigma, Vt = svds(self.user_item_matrix.asfptype(), k=min(n_factors, min(self.user_item_matrix.shape) - 1))

        self.svd_U = U
        self.svd_sigma = sigma
        self.svd_Vt = Vt

        logger.info("SVD fitting complete")

    def _build_content_features(self):
        """Build content-based features for products."""
        products = self.loader.products.copy()

        # Use available numeric features
        feature_cols = []

        if 'product_weight_g' in products.columns:
            feature_cols.append('product_weight_g')
        if 'product_length_cm' in products.columns:
            feature_cols.append('product_length_cm')
        if 'product_height_cm' in products.columns:
            feature_cols.append('product_height_cm')
        if 'product_width_cm' in products.columns:
            feature_cols.append('product_width_cm')

        # Fill missing values
        for col in feature_cols:
            products[col] = products[col].fillna(products[col].median())

        # Get average price per product
        avg_prices = self.loader.order_items.groupby('product_id')['price'].mean().reset_index()
        avg_prices.columns = ['product_id', 'avg_price']
        products = products.merge(avg_prices, on='product_id', how='left')
        products['avg_price'] = products['avg_price'].fillna(products['avg_price'].median())

        feature_cols.append('avg_price')

        # One-hot encode category
        if 'product_category_name' in products.columns:
            category_dummies = pd.get_dummies(products['product_category_name'], prefix='cat')
            products = pd.concat([products, category_dummies], axis=1)
            feature_cols.extend(category_dummies.columns.tolist())

        # Extract features
        self.product_features = products[['product_id'] + feature_cols].set_index('product_id')

        # Standardize
        scaler = StandardScaler()
        self.product_features[feature_cols] = scaler.fit_transform(self.product_features[feature_cols])

        logger.info(f"Built content features: {self.product_features.shape}")

    def fit_content_based(self):
        """Fit content-based filtering model."""
        self._build_content_features()

        # Compute pairwise similarity
        self.content_similarity = cosine_similarity(self.product_features.values)

        logger.info("Content-based model fitted")

    def _compute_temporal_weights(self, half_life_days: int = 30):
        """
        Compute temporal decay weights for purchases.

        Args:
            half_life_days: Half-life for exponential decay.
        """
        orders = self.loader.orders.copy()
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])

        reference_date = orders['order_purchase_timestamp'].max()
        orders['days_ago'] = (reference_date - orders['order_purchase_timestamp']).dt.days

        # Exponential decay: weight = 0.5^(days_ago / half_life)
        orders['temporal_weight'] = 0.5 ** (orders['days_ago'] / half_life_days)

        self.temporal_weights = orders[['order_id', 'temporal_weight']]

        logger.info("Computed temporal weights")

    def recommend_content_based(
        self,
        customer_unique_id: str,
        top_k: int = 5,
        exclude_purchased: bool = True
    ) -> List[Dict]:
        """
        Generate content-based recommendations.

        Args:
            customer_unique_id: Customer unique ID.
            top_k: Number of recommendations.
            exclude_purchased: Whether to exclude already purchased products.

        Returns:
            List of recommended products.
        """
        if self.content_similarity is None:
            self.fit_content_based()

        # Get customer's purchase history
        customer_orders = self.loader.orders[
            self.loader.orders['customer_id'].isin(
                self.loader.customers[
                    self.loader.customers['customer_unique_id'] == customer_unique_id
                ]['customer_id']
            )
        ]

        if len(customer_orders) == 0:
            return []

        order_ids = customer_orders['order_id'].tolist()
        purchased_products = self.loader.order_items[
            self.loader.order_items['order_id'].isin(order_ids)
        ]['product_id'].unique()

        # Find similar products
        scores = np.zeros(len(self.product_features))

        for prod in purchased_products:
            if prod in self.product_features.index:
                prod_idx = self.product_features.index.get_loc(prod)
                scores += self.content_similarity[prod_idx]

        # Exclude purchased products
        if exclude_purchased:
            for prod in purchased_products:
                if prod in self.product_features.index:
                    prod_idx = self.product_features.index.get_loc(prod)
                    scores[prod_idx] = -np.inf

        # Get top K
        top_indices = np.argsort(scores)[-top_k:][::-1]
        recommendations = []

        for idx in top_indices:
            product_id = self.product_features.index[idx]
            recommendations.append({
                'product_id': product_id,
                'score': float(scores[idx]),
                'method': 'content_based'
            })

        return recommendations

    def recommend_category_aware(
        self,
        customer_unique_id: str,
        top_k: int = 5,
        diversity_factor: float = 0.3
    ) -> List[Dict]:
        """
        Generate category-aware recommendations with diversity.

        Args:
            customer_unique_id: Customer unique ID.
            top_k: Number of recommendations.
            diversity_factor: Weight for category diversity (0-1).

        Returns:
            List of recommended products.
        """
        # Get customer's category preferences
        customer_orders = self.loader.orders[
            self.loader.orders['customer_id'].isin(
                self.loader.customers[
                    self.loader.customers['customer_unique_id'] == customer_unique_id
                ]['customer_id']
            )
        ]

        if len(customer_orders) == 0:
            return []

        order_ids = customer_orders['order_id'].tolist()
        customer_items = self.loader.order_items[
            self.loader.order_items['order_id'].isin(order_ids)
        ]

        # Get categories
        customer_items = customer_items.merge(
            self.loader.products[['product_id', 'product_category_name']],
            on='product_id',
            how='left'
        )

        # Calculate category affinity
        category_counts = customer_items['product_category_name'].value_counts()
        preferred_categories = category_counts.head(3).index.tolist()

        # Get popular products from preferred categories
        category_products = self.loader.products[
            self.loader.products['product_category_name'].isin(preferred_categories)
        ]

        # Get popularity scores
        product_popularity = (
            self.loader.order_items
            .groupby('product_id')
            .size()
            .reset_index(name='popularity')
        )

        category_products = category_products.merge(
            product_popularity,
            on='product_id',
            how='left'
        )
        category_products['popularity'] = category_products['popularity'].fillna(0)

        # Exclude already purchased
        purchased = customer_items['product_id'].unique()
        category_products = category_products[~category_products['product_id'].isin(purchased)]

        # Diversify across categories
        recommendations = []
        max_per_category = max(1, top_k // len(preferred_categories))

        for category in preferred_categories:
            cat_products = category_products[
                category_products['product_category_name'] == category
            ].sort_values('popularity', ascending=False).head(max_per_category)

            for _, row in cat_products.iterrows():
                if len(recommendations) < top_k:
                    recommendations.append({
                        'product_id': row['product_id'],
                        'category': category,
                        'score': float(row['popularity']),
                        'method': 'category_aware'
                    })

        return recommendations[:top_k]

    def recommend_ensemble(
        self,
        customer_unique_id: str,
        top_k: int = 5,
        weights: Dict[str, float] = None
    ) -> List[Dict]:
        """
        Ensemble recommendations combining multiple methods.

        Args:
            customer_unique_id: Customer unique ID.
            top_k: Number of recommendations.
            weights: Weights for each method.

        Returns:
            List of recommended products.
        """
        if weights is None:
            weights = {
                'popularity': 0.3,
                'content': 0.3,
                'category': 0.4
            }

        # Get recommendations from each method
        from model import OlistRecommender
        base_model = OlistRecommender(self.loader)
        base_model.fit()

        popularity_recs = base_model.recommend(
            customer_unique_id,
            top_k=top_k * 2,
            method='popularity'
        )

        content_recs = self.recommend_content_based(
            customer_unique_id,
            top_k=top_k * 2
        )

        category_recs = self.recommend_category_aware(
            customer_unique_id,
            top_k=top_k * 2
        )

        # Combine scores
        product_scores = {}

        for rec in popularity_recs:
            pid = rec['product_id']
            score = rec.get('purchase_count', rec.get('score', 0))
            product_scores[pid] = product_scores.get(pid, 0) + weights['popularity'] * score

        for rec in content_recs:
            pid = rec['product_id']
            score = rec.get('score', 0)
            product_scores[pid] = product_scores.get(pid, 0) + weights['content'] * score

        for rec in category_recs:
            pid = rec['product_id']
            score = rec.get('score', 0)
            product_scores[pid] = product_scores.get(pid, 0) + weights['category'] * score

        # Sort by combined score
        sorted_products = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)

        recommendations = []
        for product_id, score in sorted_products[:top_k]:
            recommendations.append({
                'product_id': product_id,
                'score': score,
                'method': 'ensemble'
            })

        return recommendations

    def explain_recommendation(
        self,
        customer_unique_id: str,
        product_id: str
    ) -> Dict:
        """
        Explain why a product is recommended.

        Args:
            customer_unique_id: Customer unique ID.
            product_id: Product ID.

        Returns:
            Explanation dictionary.
        """
        # Get customer purchase history
        customer_orders = self.loader.orders[
            self.loader.orders['customer_id'].isin(
                self.loader.customers[
                    self.loader.customers['customer_unique_id'] == customer_unique_id
                ]['customer_id']
            )
        ]

        order_ids = customer_orders['order_id'].tolist()
        purchased_products = self.loader.order_items[
            self.loader.order_items['order_id'].isin(order_ids)
        ]['product_id'].unique()

        # Get product category
        product_info = self.loader.products[
            self.loader.products['product_id'] == product_id
        ]

        if len(product_info) == 0:
            return {'reason': 'Product not found'}

        product_category = product_info.iloc[0]['product_category_name']

        # Check if from frequently purchased category
        purchased_with_category = self.loader.order_items[
            self.loader.order_items['product_id'].isin(purchased_products)
        ].merge(
            self.loader.products[['product_id', 'product_category_name']],
            on='product_id'
        )

        category_counts = purchased_with_category['product_category_name'].value_counts()

        explanation = {
            'product_id': product_id,
            'product_category': product_category,
            'reasons': []
        }

        # Check popularity
        popularity = len(self.loader.order_items[
            self.loader.order_items['product_id'] == product_id
        ])

        if popularity > 100:
            explanation['reasons'].append(f"Popular product ({popularity} purchases)")

        # Check category match
        if product_category in category_counts.index:
            explanation['reasons'].append(
                f"You've purchased {category_counts[product_category]} items from {product_category} category"
            )

        # Check co-purchase
        # (simplified - would need full co-purchase matrix)

        if len(explanation['reasons']) == 0:
            explanation['reasons'].append("Trending product in our store")

        return explanation


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from data_loader import OlistDataLoader

    loader = OlistDataLoader(data_dir='../datos')
    loader.load_all()

    model = AdvancedRecommender(loader)

    # Test content-based
    print("\nTesting content-based recommendations...")
    customer_id = loader.customers['customer_unique_id'].iloc[10]
    content_recs = model.recommend_content_based(customer_id, top_k=5)
    print(f"Content-based recs: {len(content_recs)}")

    # Test category-aware
    print("\nTesting category-aware recommendations...")
    category_recs = model.recommend_category_aware(customer_id, top_k=5)
    print(f"Category-aware recs: {len(category_recs)}")
