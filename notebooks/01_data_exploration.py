"""
Initial data exploration script.
Quick analysis of the Olist dataset structure.
"""
import sys
sys.path.append('..')

from src.data_loader import OlistDataLoader
import pandas as pd

# Load data
print("="*60)
print("OLIST DATASET - INITIAL EXPLORATION")
print("="*60)

loader = OlistDataLoader(data_dir='../datos')
loader.load_all()

print("\n" + "="*60)
print("DATASET OVERVIEW")
print("="*60)

# Orders overview
if loader.orders is not None:
    print(f"\nORDERS:")
    print(f"  Total orders: {len(loader.orders):,}")
    print(f"  Columns: {list(loader.orders.columns)}")
    print(f"  Date range: {loader.orders['order_purchase_timestamp'].min()} to {loader.orders['order_purchase_timestamp'].max()}")
    print(f"  Order statuses: {loader.orders['order_status'].value_counts().to_dict()}")

# Order items overview
if loader.order_items is not None:
    print(f"\nORDER ITEMS:")
    print(f"  Total items: {len(loader.order_items):,}")
    print(f"  Unique products: {loader.order_items['product_id'].nunique():,}")
    print(f"  Columns: {list(loader.order_items.columns)}")

# Products overview
if loader.products is not None:
    print(f"\nPRODUCTS:")
    print(f"  Total products: {len(loader.products):,}")
    print(f"  Unique categories: {loader.products['product_category_name'].nunique()}")
    print(f"  Columns: {list(loader.products.columns)}")

# Customers overview
if loader.customers is not None:
    print(f"\nCUSTOMERS:")
    print(f"  Total customers: {len(loader.customers):,}")
    print(f"  Unique cities: {loader.customers['customer_city'].nunique()}")
    print(f"  Columns: {list(loader.customers.columns)}")

# Reviews overview
if loader.reviews is not None:
    print(f"\nREVIEWS:")
    print(f"  Total reviews: {len(loader.reviews):,}")
    print(f"  Columns: {list(loader.reviews.columns)}")
    print(f"  Review score distribution:")
    print(loader.reviews['review_score'].value_counts().sort_index())

print("\n" + "="*60)
print("MISSING VALUES CHECK")
print("="*60)

for name, df in [('orders', loader.orders), ('order_items', loader.order_items),
                  ('products', loader.products), ('customers', loader.customers)]:
    if df is not None:
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"\n{name.upper()}:")
            print(missing[missing > 0])

print("\nExploration complete!")
