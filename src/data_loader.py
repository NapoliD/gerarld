"""
Data loader for Olist dataset.
Loads CSV files and provides easy access to dataframes.
"""
import os
import pandas as pd
from pathlib import Path


class OlistDataLoader:
    """Load and manage Olist e-commerce dataset."""

    def __init__(self, data_dir: str = None):
        """
        Initialize data loader.

        Args:
            data_dir: Path to directory containing CSV files.
                     If None, uses DATA_DIR env variable or './datos'.
        """
        if data_dir is None:
            data_dir = os.getenv('DATA_DIR', './datos')
        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")

        # Initialize dataframes as None
        self.orders = None
        self.order_items = None
        self.products = None
        self.customers = None
        self.payments = None
        self.reviews = None
        self.sellers = None
        self.category_translation = None

    def load_all(self):
        """Load all available CSV files."""
        self.orders = self._load_csv('olist_orders_dataset.csv')
        self.order_items = self._load_csv('olist_order_items_dataset.csv')
        self.products = self._load_csv('olist_products_dataset.csv')
        self.customers = self._load_csv('olist_customers_dataset.csv')
        self.payments = self._load_csv('olist_order_payments_dataset.csv')
        self.reviews = self._load_csv('olist_order_reviews_dataset.csv')
        self.sellers = self._load_csv('olist_sellers_dataset.csv')
        self.category_translation = self._load_csv('product_category_name_translation.csv')

        return self

    def _load_csv(self, filename: str) -> pd.DataFrame:
        """
        Load a single CSV file.

        Args:
            filename: Name of the CSV file to load.

        Returns:
            DataFrame with the loaded data, or None if file not found.
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            print(f"Warning: {filename} not found")
            return None

        try:
            df = pd.read_csv(filepath)
            print(f"Loaded {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None

    def get_merged_data(self):
        """
        Get merged dataset with all relevant information.

        Returns:
            DataFrame with orders, items, products, customers merged.
        """
        if any(df is None for df in [self.orders, self.order_items, self.products, self.customers]):
            raise ValueError("Not all required datasets are loaded. Call load_all() first.")

        # Merge order items with products
        data = self.order_items.merge(
            self.products,
            on='product_id',
            how='left'
        )

        # Merge with orders
        data = data.merge(
            self.orders,
            on='order_id',
            how='left'
        )

        # Merge with customers
        data = data.merge(
            self.customers,
            on='customer_id',
            how='left'
        )

        # Translate category names if available
        if self.category_translation is not None:
            data = data.merge(
                self.category_translation,
                on='product_category_name',
                how='left'
            )

        return data

    def get_orders_with_reviews(self):
        """Get orders merged with review scores."""
        if self.orders is None or self.reviews is None:
            raise ValueError("Orders and reviews must be loaded first.")

        return self.orders.merge(
            self.reviews[['order_id', 'review_score']],
            on='order_id',
            how='left'
        )


if __name__ == "__main__":
    # Quick test
    loader = OlistDataLoader()
    loader.load_all()
    print("\nDatasets loaded successfully!")

    if loader.orders is not None:
        print(f"\nOrders date range: {loader.orders['order_purchase_timestamp'].min()} to {loader.orders['order_purchase_timestamp'].max()}")
