"""
Analytics module for computing business KPIs and insights.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple


class OlistAnalytics:
    """Compute analytics and KPIs for Olist dataset."""

    def __init__(self, loader):
        """
        Initialize analytics with a data loader.

        Args:
            loader: OlistDataLoader instance with loaded data.
        """
        self.loader = loader

    def compute_all_metrics(self) -> Dict:
        """
        Compute all key metrics.

        Returns:
            Dictionary with all computed metrics.
        """
        metrics = {}

        # Basic metrics
        metrics['total_orders'] = len(self.loader.orders)
        metrics['total_customers'] = self.loader.customers['customer_unique_id'].nunique()
        metrics['total_gmv'] = self.loader.order_items['price'].sum()
        metrics['avg_order_value'] = self.loader.order_items.groupby('order_id')['price'].sum().mean()

        # Order status metrics
        metrics['delivery_rate'] = (
            self.loader.orders['order_status'].value_counts()['delivered'] /
            len(self.loader.orders) * 100
        )

        # Repeat purchase metrics
        repeat_metrics = self._compute_repeat_purchase_rate()
        metrics.update(repeat_metrics)

        # Category metrics
        metrics['top_categories'] = self._get_top_categories()

        # Review metrics
        if self.loader.reviews is not None:
            metrics['avg_review_score'] = self.loader.reviews['review_score'].mean()
            metrics['review_distribution'] = self.loader.reviews['review_score'].value_counts().sort_index().to_dict()

        # Time between orders
        metrics['avg_days_between_orders'] = self._compute_avg_days_between_orders()

        return metrics

    def _compute_repeat_purchase_rate(self) -> Dict:
        """Compute repeat purchase metrics."""
        # Merge orders with customers to get customer_unique_id
        orders_with_unique = self.loader.orders.merge(
            self.loader.customers[['customer_id', 'customer_unique_id']],
            on='customer_id',
            how='left'
        )

        # Get unique customer IDs
        customer_orders = (
            orders_with_unique
            .groupby('customer_unique_id')
            .size()
            .reset_index(name='order_count')
        )

        total_customers = len(customer_orders)
        repeat_customers = len(customer_orders[customer_orders['order_count'] > 1])

        return {
            'repeat_purchase_rate': (repeat_customers / total_customers * 100),
            'repeat_customers': repeat_customers,
            'one_time_customers': total_customers - repeat_customers,
            'avg_orders_per_customer': customer_orders['order_count'].mean(),
            'max_orders_per_customer': customer_orders['order_count'].max()
        }

    def _get_top_categories(self, top_n: int = 10) -> pd.DataFrame:
        """Get top product categories by orders and GMV."""
        # Merge order items with products
        items_products = self.loader.order_items.merge(
            self.loader.products[['product_id', 'product_category_name']],
            on='product_id',
            how='left'
        )

        # Translate categories if available
        if self.loader.category_translation is not None:
            items_products = items_products.merge(
                self.loader.category_translation,
                on='product_category_name',
                how='left'
            )
            category_col = 'product_category_name_english'
        else:
            category_col = 'product_category_name'

        # Compute metrics by category
        category_metrics = (
            items_products
            .groupby(category_col)
            .agg(
                total_orders=('order_id', 'nunique'),
                total_items=('order_id', 'count'),
                total_gmv=('price', 'sum'),
                avg_price=('price', 'mean')
            )
            .sort_values('total_gmv', ascending=False)
            .head(top_n)
            .reset_index()
        )

        return category_metrics

    def _compute_avg_days_between_orders(self) -> float:
        """Compute average days between orders for repeat customers."""
        # Convert timestamp to datetime and merge with customers
        orders = self.loader.orders.copy()
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
        orders = orders.merge(
            self.loader.customers[['customer_id', 'customer_unique_id']],
            on='customer_id',
            how='left'
        )

        # Get customers with multiple orders
        repeat_customers = (
            orders
            .groupby('customer_unique_id')['order_id']
            .count()
            .loc[lambda x: x > 1]
            .index
        )

        if len(repeat_customers) == 0:
            return np.nan

        # Calculate time between consecutive orders
        time_diffs = []
        for customer_unique_id in repeat_customers:
            customer_orders = (
                orders[orders['customer_unique_id'] == customer_unique_id]
                .sort_values('order_purchase_timestamp')
            )

            if len(customer_orders) >= 2:
                dates = customer_orders['order_purchase_timestamp'].values
                diffs = np.diff(dates).astype('timedelta64[D]').astype(int)
                time_diffs.extend(diffs)

        if len(time_diffs) == 0:
            return np.nan

        return np.mean(time_diffs)

    def get_customer_purchase_history(self, customer_id: str) -> pd.DataFrame:
        """
        Get purchase history for a specific customer.

        Args:
            customer_id: Customer ID to look up.

        Returns:
            DataFrame with customer's order history.
        """
        customer_orders = self.loader.orders[
            self.loader.orders['customer_id'] == customer_id
        ].copy()

        if len(customer_orders) == 0:
            return pd.DataFrame()

        # Get items for these orders
        order_ids = customer_orders['order_id'].tolist()
        items = self.loader.order_items[
            self.loader.order_items['order_id'].isin(order_ids)
        ]

        # Merge with products
        items = items.merge(
            self.loader.products[['product_id', 'product_category_name']],
            on='product_id',
            how='left'
        )

        return items

    def print_summary(self):
        """Print a formatted summary of key metrics."""
        metrics = self.compute_all_metrics()

        print("="*70)
        print("OLIST ANALYTICS SUMMARY")
        print("="*70)

        print(f"\nOVERALL METRICS:")
        print(f"  Total Orders: {metrics['total_orders']:,}")
        print(f"  Total Customers: {metrics['total_customers']:,}")
        print(f"  Total GMV: ${metrics['total_gmv']:,.2f}")
        print(f"  Avg Order Value: ${metrics['avg_order_value']:.2f}")
        print(f"  Delivery Rate: {metrics['delivery_rate']:.2f}%")

        print(f"\nCUSTOMER RETENTION:")
        print(f"  Repeat Purchase Rate: {metrics['repeat_purchase_rate']:.2f}%")
        print(f"  Repeat Customers: {metrics['repeat_customers']:,}")
        print(f"  One-Time Customers: {metrics['one_time_customers']:,}")
        print(f"  Avg Orders per Customer: {metrics['avg_orders_per_customer']:.2f}")
        print(f"  Max Orders per Customer: {metrics['max_orders_per_customer']}")
        print(f"  Avg Days Between Orders: {metrics['avg_days_between_orders']:.1f}")

        print(f"\nREVIEWS:")
        print(f"  Average Review Score: {metrics['avg_review_score']:.2f}/5.0")

        print(f"\nTOP 10 CATEGORIES BY GMV:")
        top_cats = metrics['top_categories']
        for idx, row in top_cats.iterrows():
            cat_name = row['product_category_name_english'] if 'product_category_name_english' in row else row['product_category_name']
            print(f"  {idx+1}. {cat_name}: ${row['total_gmv']:,.0f} ({row['total_orders']:,} orders)")

        print("\n" + "="*70)


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.append('..')
    from data_loader import OlistDataLoader

    loader = OlistDataLoader(data_dir='../datos')
    loader.load_all()

    analytics = OlistAnalytics(loader)
    analytics.print_summary()
