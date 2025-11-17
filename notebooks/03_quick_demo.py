"""
Quick demo script showing all components working together.
Run this after setup to verify everything works.
"""
import sys
sys.path.append('..')

from src.data_loader import OlistDataLoader
from src.analytics import OlistAnalytics
from src.model import OlistRecommender

print("="*70)
print("OLIST RECOMMENDATION SYSTEM - QUICK DEMO")
print("="*70)

# 1. Load Data
print("\n1. Loading data...")
loader = OlistDataLoader(data_dir='../datos')
loader.load_all()
print(f"   Loaded {len(loader.orders)} orders")

# 2. Analytics
print("\n2. Computing analytics...")
analytics = OlistAnalytics(loader)
metrics = analytics.compute_all_metrics()
print(f"   Total GMV: ${metrics['total_gmv']:,.2f}")
print(f"   Repeat Purchase Rate: {metrics['repeat_purchase_rate']:.2f}%")
print(f"   Avg Review Score: {metrics['avg_review_score']:.2f}/5.0")

# 3. Train Model
print("\n3. Training recommendation model...")
model = OlistRecommender(loader)
model.fit(min_support=3)
print(f"   Model trained with {len(model.popularity_scores)} products")

# 4. Generate Recommendations
print("\n4. Generating recommendations...")
sample_customer = loader.orders['customer_id'].iloc[50]
recommendations = model.recommend(sample_customer, top_k=5, method='hybrid')

print(f"\n   Top 5 recommendations for customer {sample_customer[:16]}...")
for i, rec in enumerate(recommendations, 1):
    category = rec.get('product_category_name', 'N/A')
    print(f"   {i}. {category}")

# 5. Summary
print("\n" + "="*70)
print("DEMO COMPLETE - All components working!")
print("="*70)
print("\nNext steps:")
print("  - Run full analytics: python -m src.main --analytics")
print("  - Train & save model: python -m src.main --train")
print("  - Get recommendations: python -m src.main --customer_id <ID> --top_k 5")
print("  - Run tests: pytest -v")
print("\n")
