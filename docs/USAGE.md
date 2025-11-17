# Usage Guide

Comprehensive usage examples for all project features.

---

## Table of Contents

1. [AI Chatbot](#1-ai-chatbot)
2. [Analytics & Business Insights](#2-analytics--business-insights)
3. [Churn Prediction](#3-churn-prediction)
4. [Recommendation Engine](#4-recommendation-engine)
5. [FastAPI Server](#5-fastapi-server)
6. [CLI Commands](#6-cli-commands)

---

## 1. AI Chatbot

### Interactive Mode (Recommended)

```bash
# Start the chatbot
python data_chatbot.py

# Interactive session
You: What CSV files are available?
You: Analyze olist_products_dataset.csv
You: Check data quality of olist_orders_dataset.csv
You: Suggest improvements for olist_customers_dataset.csv
You: quit
```

### Programmatic Usage

```python
from data_chatbot import DataAnalysisChatbot

# Initialize chatbot
chatbot = DataAnalysisChatbot(data_dir='datos')

# Ask questions
response = chatbot.chat("What CSV files are available?")
print(response)

response = chatbot.chat("Analyze olist_products_dataset.csv")
print(response)

response = chatbot.chat("Check data quality")
print(response)
```

### Example Questions

**List Files:**
```
What CSV files are available?
List all datasets
Show me the data files
```

**Analyze Data:**
```
Analyze olist_products_dataset.csv
Show me the structure of customers.csv
What's in the orders dataset?
```

**Check Quality:**
```
Check data quality of olist_products_dataset.csv
What are the data issues in customers.csv?
Find problems in orders.csv
```

**Get Recommendations:**
```
Suggest improvements for olist_products_dataset.csv
How can I fix the data quality issues?
Recommend data transformations
```

### Advanced: Pandas Agent

For complex queries, use the Pandas agent version:

```bash
python data_chatbot_pandas.py
```

```
You: Show me the top 10 heaviest products in olist_products_dataset.csv
You: Calculate correlation between price and weight
You: Find products with missing categories
```

---

## 2. Analytics & Business Insights

### Run Full Analytics

```bash
python -m src.main --analytics --data_dir datos
```

**Output includes:**
- Total GMV and order counts
- Repeat purchase rate
- Delivery success rate
- Average review scores
- Top categories by revenue
- Customer distribution
- Temporal trends

### Programmatic Analytics

```python
from src.data_loader import OlistDataLoader
from src.analytics import analyze_business_metrics

# Load data
loader = OlistDataLoader(data_dir='datos')
customers_df = loader.load_customers()
orders_df = loader.load_orders()
order_items_df = loader.load_order_items()

# Get metrics
metrics = analyze_business_metrics(
    customers_df,
    orders_df,
    order_items_df
)

print(f"GMV: ${metrics['total_gmv']:,.2f}")
print(f"Repeat Rate: {metrics['repeat_purchase_rate']:.2%}")
print(f"Delivery Rate: {metrics['delivery_success_rate']:.2%}")
```

### Custom Analysis

```python
from src.data_loader import OlistDataLoader

loader = OlistDataLoader(data_dir='datos')

# Customer analysis
customers = loader.load_customers()
print(f"Total customers: {len(customers)}")
print(f"Unique states: {customers['customer_state'].nunique()}")

# Order analysis
orders = loader.load_orders()
print(f"Total orders: {len(orders)}")
print(f"Date range: {orders['order_purchase_timestamp'].min()} to {orders['order_purchase_timestamp'].max()}")

# Revenue analysis
items = loader.load_order_items()
print(f"Total revenue: ${items['price'].sum():,.2f}")
print(f"Average order value: ${items.groupby('order_id')['price'].sum().mean():,.2f}")
```

---

## 3. Churn Prediction

### Train Model

```bash
python -m src.main --train --data_dir datos
```

### Use Trained Model

```python
from src.churn_predictor import ChurnPredictor
from src.data_loader import OlistDataLoader

# Load data
loader = OlistDataLoader(data_dir='datos')
df = loader.prepare_churn_dataset()

# Initialize and train predictor
predictor = ChurnPredictor()
predictor.train(df)

# Make predictions
predictions = predictor.predict(df)

# Get at-risk customers
at_risk = df[predictions == 1]
print(f"At-risk customers: {len(at_risk)}")

# Get feature importance
importance = predictor.get_feature_importance()
print("\nTop features:")
for feature, score in importance[:10]:
    print(f"  {feature}: {score:.4f}")

# Save model
predictor.save_model('models/churn_model.pkl')
```

### Predict for New Customer

```python
import pandas as pd
from src.churn_predictor import ChurnPredictor

# Load trained model
predictor = ChurnPredictor.load_model('models/churn_model.pkl')

# New customer data
new_customer = pd.DataFrame({
    'recency_days': [30],
    'frequency': [1],
    'monetary_value': [150.00],
    'avg_review_score': [4.5],
    'delivery_success_rate': [1.0],
    # ... other features
})

# Predict
prediction = predictor.predict(new_customer)
probability = predictor.predict_proba(new_customer)

if prediction[0] == 1:
    print(f"⚠️  At risk of churn ({probability[0][1]:.1%} probability)")
else:
    print(f"✓ Likely to return ({probability[0][0]:.1%} probability)")
```

---

## 4. Recommendation Engine

### Get Recommendations

```bash
python -m src.main --customer_id <CUSTOMER_ID> --top_k 10 --data_dir datos
```

### Programmatic Usage

```python
from src.recommendation_engine import RecommendationEngine

# Initialize engine
engine = RecommendationEngine(data_dir='datos')

# Get recommendations for a customer
recommendations = engine.recommend_for_customer(
    customer_id='abc123def456',
    n_recommendations=10
)

print("Recommended products:")
for i, (product_id, score) in enumerate(recommendations, 1):
    print(f"{i}. Product {product_id} (score: {score:.3f})")
```

### Different Recommendation Methods

```python
# Popularity-based
recs_pop = engine.recommend_by_popularity(n=10)

# Co-purchase based
recs_copurchase = engine.recommend_by_copurchase(
    customer_id='abc123',
    n=10
)

# Hybrid (default)
recs_hybrid = engine.recommend_hybrid(
    customer_id='abc123',
    n=10
)
```

### Train Custom Model

```python
from src.recommendation_engine import RecommendationEngine

engine = RecommendationEngine(data_dir='datos')

# Train with custom parameters
engine.train(
    min_support=5,  # Minimum co-purchases
    min_confidence=0.1  # Minimum confidence score
)

# Save model
engine.save_model('models/recommendation_model.pkl')
```

---

## 5. FastAPI Server

### Start Server

```bash
# Development mode
python src/api.py

# Production mode with uvicorn
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

#### 1. Health Check
```bash
curl http://localhost:8000/health
```

#### 2. Get Analytics
```bash
curl http://localhost:8000/analytics/customer_distribution
curl http://localhost:8000/analytics/top_categories
curl http://localhost:8000/analytics/review_distribution
```

#### 3. Get Recommendations
```bash
curl http://localhost:8000/recommendations/<customer_id>/10
```

#### 4. Predict Churn
```bash
curl -X POST http://localhost:8000/churn/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "abc123def456"}'
```

### Python Client

```python
import requests

BASE_URL = "http://localhost:8000"

# Get analytics
response = requests.get(f"{BASE_URL}/analytics/customer_distribution")
print(response.json())

# Get recommendations
customer_id = "abc123def456"
response = requests.get(f"{BASE_URL}/recommendations/{customer_id}/10")
recommendations = response.json()
print(recommendations)

# Predict churn
response = requests.post(
    f"{BASE_URL}/churn/predict",
    json={"customer_id": customer_id}
)
prediction = response.json()
print(f"Churn risk: {prediction['churn_probability']:.1%}")
```

### API Documentation

Interactive API docs available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 6. CLI Commands

### Analytics Commands

```bash
# Full analytics report
python -m src.main --analytics

# With custom data directory
python -m src.main --analytics --data_dir /path/to/data
```

### Model Training

```bash
# Train churn model
python -m src.main --train

# Train recommendation model
python -m src.main --train --model recommendation
```

### Model Evaluation

```bash
# Evaluate churn model
python -m src.main --evaluate

# Evaluate with metrics
python -m src.main --evaluate --metrics precision recall f1
```

### Generate Recommendations

```bash
# Get recommendations for customer
python -m src.main --customer_id <ID> --top_k 5

# Different methods
python -m src.main --customer_id <ID> --method popularity
python -m src.main --customer_id <ID> --method copurchase
python -m src.main --customer_id <ID> --method hybrid
```

### Combined Operations

```bash
# Train and evaluate
python -m src.main --train --evaluate

# Analytics and recommendations
python -m src.main --analytics --customer_id <ID> --top_k 10
```

---

## Example Workflows

### Workflow 1: Initial Data Exploration

```bash
# 1. Run chatbot to understand data
python data_chatbot.py
You: What CSV files are available?
You: Analyze olist_orders_dataset.csv
You: Check data quality
You: quit

# 2. Run business analytics
python -m src.main --analytics

# 3. Review outputs in outputs/eda/
```

### Workflow 2: Build Churn Model

```python
from src.data_loader import OlistDataLoader
from src.churn_predictor import ChurnPredictor

# 1. Load and prepare data
loader = OlistDataLoader(data_dir='datos')
df = loader.prepare_churn_dataset()

# 2. Train model
predictor = ChurnPredictor()
predictor.train(df)

# 3. Evaluate
metrics = predictor.evaluate(df)
print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"Precision: {metrics['precision']:.2%}")

# 4. Save model
predictor.save_model('models/churn_model.pkl')
```

### Workflow 3: Deploy API

```bash
# 1. Train models
python -m src.main --train

# 2. Start API server
python src/api.py

# 3. Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/analytics/customer_distribution

# 4. Access docs at http://localhost:8000/docs
```

---

## Tips & Best Practices

### For Chatbot

- Start with "What CSV files are available?" to see available data
- Be specific in questions (mention exact filenames)
- Ask for quality checks before making changes
- Use recommendations as starting points, not final solutions

### For Analytics

- Run analytics regularly to track metrics
- Compare time periods to identify trends
- Focus on actionable metrics (repeat rate, churn, AOV)
- Visualize results for better insights

### For ML Models

- Retrain models periodically with new data
- Validate on holdout set before deployment
- Monitor performance metrics in production
- Use cross-validation for reliable estimates

### For API

- Enable rate limiting for production
- Add authentication for security
- Log all requests for monitoring
- Cache frequent queries for performance

---

## Need More Help?

- **Installation issues**: See [Installation Guide](INSTALLATION.md)
- **Common questions**: Check [FAQ](FAQ.md)
- **Chatbot details**: Read [Chatbot Guide](CHATBOT_GUIDE.md)
- **Business insights**: Review [Analytics Summary](ANALYTICS_SUMMARY.md)

---

**Ready to analyze your data! 📊**
