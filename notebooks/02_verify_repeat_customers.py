"""
Verify repeat customer logic - investigate customer_id vs customer_unique_id.
"""
import sys
sys.path.append('..')

from src.data_loader import OlistDataLoader
import pandas as pd

loader = OlistDataLoader(data_dir='../datos')
loader.load_all()

print("="*60)
print("CUSTOMER ID INVESTIGATION")
print("="*60)

# Check customer_id
print(f"\nTotal orders: {len(loader.orders)}")
print(f"Unique customer_id: {loader.orders['customer_id'].nunique()}")

# Merge with customers to get customer_unique_id
orders_with_unique = loader.orders.merge(
    loader.customers[['customer_id', 'customer_unique_id']],
    on='customer_id',
    how='left'
)

print(f"Unique customer_unique_id: {orders_with_unique['customer_unique_id'].nunique()}")

# Check for repeat customers using customer_unique_id
customer_order_counts = (
    orders_with_unique
    .groupby('customer_unique_id')
    .size()
    .reset_index(name='order_count')
)

print(f"\nOrder count distribution:")
print(customer_order_counts['order_count'].value_counts().sort_index().head(10))

repeat_customers = customer_order_counts[customer_order_counts['order_count'] > 1]
print(f"\nRepeat customers (using customer_unique_id): {len(repeat_customers)}")
print(f"Repeat purchase rate: {len(repeat_customers) / len(customer_order_counts) * 100:.2f}%")

# Show some examples
if len(repeat_customers) > 0:
    print(f"\nTop 5 customers by order count:")
    print(repeat_customers.sort_values('order_count', ascending=False).head())
