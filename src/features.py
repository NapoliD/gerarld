"""
Feature engineering module for advanced recommendations.
Implements RFM, category affinity, and other customer features.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from datetime import datetime, timedelta


class FeatureEngineer:
    """Engineer features for recommendation models."""

    def __init__(self, loader):
        """
        Initialize feature engineer.

        Args:
            loader: OlistDataLoader instance.
        """
        self.loader = loader
        self.reference_date = None

    def compute_rfm_features(self, reference_date: str = None) -> pd.DataFrame:
        """
        Compute RFM (Recency, Frequency, Monetary) features.

        Args:
            reference_date: Reference date for recency calculation.
                          If None, uses latest order date.

        Returns:
            DataFrame with RFM features per customer.
        """
        # Merge orders with customers
        orders = self.loader.orders.copy()
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])

        orders = orders.merge(
            self.loader.customers[['customer_id', 'customer_unique_id']],
            on='customer_id',
            how='left'
        )

        # Set reference date
        if reference_date is None:
            self.reference_date = orders['order_purchase_timestamp'].max()
        else:
            self.reference_date = pd.to_datetime(reference_date)

        # Merge with order values
        order_values = self.loader.order_items.groupby('order_id')['price'].sum().reset_index()
        order_values.columns = ['order_id', 'order_value']
        orders = orders.merge(order_values, on='order_id', how='left')

        # Calculate RFM per customer_unique_id
        rfm = orders.groupby('customer_unique_id').agg({
            'order_purchase_timestamp': lambda x: (self.reference_date - x.max()).days,  # Recency
            'order_id': 'count',  # Frequency
            'order_value': 'sum'  # Monetary
        }).reset_index()

        rfm.columns = ['customer_unique_id', 'recency_days', 'frequency', 'monetary']

        # Add RFM scores (1-5 quintiles)
        rfm['recency_score'] = pd.qcut(rfm['recency_days'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
        rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        rfm['monetary_score'] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')

        # Convert to int
        rfm['recency_score'] = rfm['recency_score'].astype(int)
        rfm['frequency_score'] = rfm['frequency_score'].astype(int)
        rfm['monetary_score'] = rfm['monetary_score'].astype(int)

        # RFM combined score
        rfm['rfm_score'] = (
            rfm['recency_score'] * 100 +
            rfm['frequency_score'] * 10 +
            rfm['monetary_score']
        )

        return rfm

    def segment_customers_by_rfm(self, rfm: pd.DataFrame = None) -> pd.DataFrame:
        """
        Segment customers based on RFM scores.

        Args:
            rfm: RFM DataFrame. If None, computes it.

        Returns:
            DataFrame with customer segments.
        """
        if rfm is None:
            rfm = self.compute_rfm_features()

        rfm = rfm.copy()

        # Define segments based on RFM scores
        def assign_segment(row):
            r, f, m = row['recency_score'], row['frequency_score'], row['monetary_score']

            # Champions: Recent, frequent, high value
            if r >= 4 and f >= 4 and m >= 4:
                return 'Champions'

            # Loyal Customers: Frequent buyers
            elif f >= 4:
                return 'Loyal Customers'

            # Potential Loyalists: Recent customers with good frequency
            elif r >= 4 and f >= 2 and m >= 2:
                return 'Potential Loyalists'

            # New Customers: Recent but low frequency
            elif r >= 4 and f == 1:
                return 'New Customers'

            # At Risk: Used to purchase frequently but not recently
            elif r <= 2 and f >= 4:
                return 'At Risk'

            # Need Attention: Below average in recency and frequency
            elif r <= 3 and f <= 3:
                return 'Need Attention'

            # About to Sleep: Low recency, low frequency
            elif r <= 2 and f <= 2:
                return 'About to Sleep'

            # Can't Lose: High value but not recent
            elif r <= 2 and m >= 4:
                return "Can't Lose"

            else:
                return 'Other'

        rfm['segment'] = rfm.apply(assign_segment, axis=1)

        return rfm

    def compute_category_affinity(self) -> pd.DataFrame:
        """
        Compute customer affinity for product categories.

        Returns:
            DataFrame with category purchase counts per customer.
        """
        # Merge order items with orders and products
        items = self.loader.order_items.merge(
            self.loader.orders[['order_id', 'customer_id']],
            on='order_id',
            how='left'
        )

        items = items.merge(
            self.loader.products[['product_id', 'product_category_name']],
            on='product_id',
            how='left'
        )

        items = items.merge(
            self.loader.customers[['customer_id', 'customer_unique_id']],
            on='customer_id',
            how='left'
        )

        # Count purchases per category per customer
        category_affinity = (
            items.groupby(['customer_unique_id', 'product_category_name'])
            .size()
            .reset_index(name='purchase_count')
        )

        # Pivot to get categories as columns
        affinity_matrix = category_affinity.pivot(
            index='customer_unique_id',
            columns='product_category_name',
            values='purchase_count'
        ).fillna(0)

        return affinity_matrix

    def compute_price_sensitivity(self) -> pd.DataFrame:
        """
        Compute customer price sensitivity metrics.

        Returns:
            DataFrame with price range preferences per customer.
        """
        # Merge to get customer-level prices
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

        # Calculate price metrics per customer
        price_metrics = items.groupby('customer_unique_id')['price'].agg([
            ('avg_price', 'mean'),
            ('min_price', 'min'),
            ('max_price', 'max'),
            ('std_price', 'std')
        ]).reset_index()

        # Fill NaN std with 0 (single purchase customers)
        price_metrics['std_price'] = price_metrics['std_price'].fillna(0)

        # Add price range
        price_metrics['price_range'] = price_metrics['max_price'] - price_metrics['min_price']

        return price_metrics

    def compute_temporal_features(self) -> pd.DataFrame:
        """
        Compute temporal purchase patterns.

        Returns:
            DataFrame with temporal features per customer.
        """
        orders = self.loader.orders.copy()
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])

        orders = orders.merge(
            self.loader.customers[['customer_id', 'customer_unique_id']],
            on='customer_id',
            how='left'
        )

        # Extract temporal features
        orders['hour'] = orders['order_purchase_timestamp'].dt.hour
        orders['day_of_week'] = orders['order_purchase_timestamp'].dt.dayofweek
        orders['day_of_month'] = orders['order_purchase_timestamp'].dt.day
        orders['month'] = orders['order_purchase_timestamp'].dt.month

        # Aggregate by customer
        temporal_features = orders.groupby('customer_unique_id').agg({
            'hour': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.mean(),  # Preferred hour
            'day_of_week': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.mean(),  # Preferred day
            'month': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.mean()  # Preferred month
        }).reset_index()

        temporal_features.columns = ['customer_unique_id', 'preferred_hour', 'preferred_day_of_week', 'preferred_month']

        return temporal_features

    def compute_all_features(self) -> pd.DataFrame:
        """
        Compute all customer features and merge them.

        Returns:
            Comprehensive feature DataFrame.
        """
        print("Computing comprehensive customer features...")

        # RFM features
        rfm = self.segment_customers_by_rfm()
        print(f"  Computed RFM features: {len(rfm)} customers")

        # Price sensitivity
        price_features = self.compute_price_sensitivity()
        print(f"  Computed price sensitivity: {len(price_features)} customers")

        # Temporal features
        temporal_features = self.compute_temporal_features()
        print(f"  Computed temporal features: {len(temporal_features)} customers")

        # Merge all features
        features = rfm.merge(price_features, on='customer_unique_id', how='left')
        features = features.merge(temporal_features, on='customer_unique_id', how='left')

        print(f"  Total features per customer: {features.shape[1] - 1}")

        return features

    def get_customer_features(self, customer_unique_id: str) -> Dict:
        """
        Get all features for a specific customer.

        Args:
            customer_unique_id: Unique customer ID.

        Returns:
            Dictionary of customer features.
        """
        features = self.compute_all_features()

        customer_data = features[features['customer_unique_id'] == customer_unique_id]

        if len(customer_data) == 0:
            return {}

        return customer_data.iloc[0].to_dict()


if __name__ == "__main__":
    # Test feature engineering
    import sys
    sys.path.append('..')
    from data_loader import OlistDataLoader

    loader = OlistDataLoader(data_dir='../datos')
    loader.load_all()

    engineer = FeatureEngineer(loader)

    # Compute RFM
    print("\nComputing RFM features...")
    rfm = engineer.compute_rfm_features()
    print(rfm.head())

    # Segment customers
    print("\nSegmenting customers...")
    segments = engineer.segment_customers_by_rfm(rfm)
    print(segments['segment'].value_counts())

    # All features
    print("\nComputing all features...")
    all_features = engineer.compute_all_features()
    print(all_features.head())
