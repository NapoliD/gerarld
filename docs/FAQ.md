# FAQ - Frequently Asked Questions
## Olist E-Commerce Recommendation System

---

## Table of Contents
- [General](#general)
- [Data Analysis](#data-analysis)
- [Recommendation Model](#recommendation-model)
- [Technical Implementation](#technical-implementation)
- [API and Deployment](#api-and-deployment)
- [Evaluation and Metrics](#evaluation-and-metrics)
- [Common Issues](#common-issues)

---

## General

### What is this project?
An enterprise-grade product recommendation system built for the Olist Brazilian e-commerce dataset. It includes data analysis, multiple recommendation algorithms, churn prediction, REST API, and visualizations.

### How long did it take to complete?
- **Basic version**: 2-3 hours (analytics, baseline model, CLI)
- **Full advanced version**: 6-8 hours (23+ enterprise features)

### What technologies were used?
- **Python 3.9+**: Main language
- **Pandas/NumPy**: Data processing
- **Scikit-learn**: Machine Learning
- **FastAPI**: REST API
- **Docker**: Containerization
- **Pytest**: Testing

### What is the project structure?
```
datos/          # CSV data files
src/            # Source code (15+ modules)
models/         # Saved models
outputs/        # Visualizations
tests/          # Unit tests
notebooks/      # Exploration scripts
```

---

## Data Analysis

### What are the main KPIs?
| Metric | Value |
|---------|-------|
| Total GMV | $13.6M |
| Orders | 99,441 |
| Unique customers | 96,096 |
| Repeat purchase rate | **3.12%** |
| Delivery rate | 97.02% |
| Average review | 4.09/5.0 |

### Why is the repeat purchase rate so low?
**96.88% of customers make only one purchase**. Possible reasons:
1. **Marketplace dynamics**: Olist is an aggregator, customers don't develop brand loyalty
2. **Price-shopping behavior**: Customers seek one-time deals
3. **Lack of retention mechanisms**: No robust loyalty programs
4. **Limited variety**: May not offer enough breadth for repeat purchases

### What are the most important insights?
1. **Retention crisis**: High satisfaction but zero loyalty
2. **Health & Beauty opportunity**: Leading category with subscription potential
3. **77-day window**: Average time between purchases for repeat customers
4. **Satisfaction paradox**: 97% successful delivery + 4.09 reviews, but customers don't return

### Which categories generate the most revenue?
Top 5:
1. Health & Beauty: $1.26M
2. Watches & Gifts: $1.21M
3. Bed, Bath & Table: $1.04M
4. Sports & Leisure: $988K
5. Computers & Accessories: $912K

### How can I generate visualizations?
```python
from src.data_loader import OlistDataLoader
from src.analytics import OlistAnalytics
from src.visualizations import OlistVisualizer

loader = OlistDataLoader(data_dir='datos')
loader.load_all()

analytics = OlistAnalytics(loader)
viz = OlistVisualizer(output_dir='outputs')
viz.create_all_visualizations(loader, analytics)
```

**Output**: 6 high-resolution PNG charts in `outputs/` folder

---

## Recommendation Model

### What algorithms are implemented?
**7+ recommendation methods:**
1. **Popularity-based**: Most popular products
2. **Co-purchase**: "Customers who bought X also bought Y"
3. **Content-based**: Product feature similarity
4. **Category-aware**: Based on preferred categories
5. **SVD (Matrix Factorization)**: User-item matrix factorization
6. **Temporal decay**: Weights by recency
7. **Ensemble**: Combination of multiple methods

### Which is the best algorithm?
**Recommendation: Ensemble Method** (hybrid)
- Combines popularity + co-purchase + content + category
- Configurable weights (default: 30/30/40)
- Fallback for cold-start
- Balance between personalization and coverage

### Why is Precision@K almost 0?
**It's expected and honest** given that:
- 96.88% of customers have only 1 order (no personalization signal)
- Predicting exact product repurchase is extremely difficult
- Many purchases are one-time (gifts, specific needs)
- Limited dataset (2016-2018)

**But the model is still useful** because:
- Provides sensible recommendations (popular + co-purchased)
- Useful for cold-start scenarios
- Foundation for A/B testing
- Can be improved with more features

### How can I improve model performance?
**Additional data needed:**
- Demographic information (age, location, income)
- Browsing behavior and search queries
- Product features (text, images)
- Longer time horizon with more repeat customers
- Collaborative filtering with larger user base

### What is cold start and how is it handled?
**Cold start**: New customer without purchase history.

**Implemented strategies:**
1. Fallback to global popularity
2. Category-based recommendations (if there's initial signal)
3. Preference quiz (conceptual)
4. Rapid adaptation with first clicks

### How does the co-purchase method work?
1. Analyzes products bought together in the same order
2. Builds affinity matrix with minimum support threshold
3. Recommends based on customer's purchase history
4. If customer bought product A, recommends products frequently bought with A

### How do I train the model?
```bash
# Train basic model
python -m src.main --train --data_dir datos

# Model is saved to models/recommender.pkl
```

### How do I get recommendations?
```bash
# CLI
python -m src.main --customer_id 2c3642e1392097fb4af76a76fec16a46 --top_k 5

# Python
from src.model import RecommenderModel
model = RecommenderModel()
model.load('models/recommender.pkl')
recs = model.recommend(customer_id, top_k=5)
```

---

## Technical Implementation

### Why Pandas and not Spark?
**Technical decisions:**
- Dataset (~100K orders) fits in memory
- Faster development
- Easier to test and debug
- Sufficient for this use case

**When to use Spark:**
- Millions+ of orders
- Distributed processing needed
- Complex real-time ETL

### Why pickle and not joblib?
- Simple and built-in serialization
- Sufficient for model size
- Easy to inspect
- Widely compatible

### How is the code organized?
**Modular design:**
- `data_loader.py`: Data loading
- `analytics.py`: Business KPIs
- `model.py`: Basic models
- `advanced_models.py`: Advanced algorithms
- `evaluate.py`: Evaluation metrics
- `features.py`: Feature engineering (RFM)
- `churn_prediction.py`: Churn prediction
- `api.py`: REST API
- `main.py`: CLI

### Are there unit tests?
**Yes, 30+ tests:**
```bash
pytest -v                 # Run all
pytest tests/test_model.py -v   # Specific tests
pytest --cov=src tests/   # With coverage
```

**Coverage target**: >80%

### What are RFM features?
**RFM = Recency, Frequency, Monetary**
- **Recency**: Days since last purchase
- **Frequency**: Number of purchases
- **Monetary**: Total value spent

**Generated segments:**
- Champions
- Loyal Customers
- Potential Loyalists
- New Customers
- At Risk
- Need Attention
- About to Sleep
- Can't Lose

### How does churn prediction work?
**Model:** Gradient Boosting Classifier

**Features used:**
- RFM scores
- Temporal patterns (preferred hour, day, month)
- Time since last purchase
- Purchase rate

**Prediction window**: 90 days

**Usage:**
```python
from src.churn_prediction import ChurnPredictor

predictor = ChurnPredictor(loader)
metrics = predictor.train(prediction_window_days=90)
risk = predictor.predict_churn_risk(customer_id)
# Output: 'Low', 'Medium', or 'High'
```

---

## API and Deployment

### How do I start the API?
```bash
# Option 1: Local
uvicorn src.api:app --reload

# Option 2: Docker
docker-compose up -d

# API will be at http://localhost:8000
```

### What endpoints are available?
```
GET  /health                          # Health check
GET  /                                # Root
GET  /recommend/{customer_id}         # Simple recommendations
POST /recommend                       # Detailed request
GET  /recommend/advanced/{customer_id} # Advanced models
GET  /features/{customer_id}          # Customer features (RFM)
GET  /explain/{customer_id}/{product} # Explainability
GET  /stats                           # Dataset statistics
```

### How do I test the endpoints?
```bash
# Health check
curl http://localhost:8000/health

# Recommendations
curl "http://localhost:8000/recommend/customer_123?top_k=5"

# Advanced recommendations
curl "http://localhost:8000/recommend/advanced/customer_123?method=ensemble"

# Customer features
curl http://localhost:8000/features/customer_123
```

### Is there interactive API documentation?
**Yes, 2 interfaces:**
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Both automatically generated by FastAPI.

### How do I deploy to production?
**Options:**
1. **AWS ECS + ALB**: Scalable containers
2. **Google Cloud Run**: Serverless with Docker
3. **Azure Container Instances**: Simple containers
4. **Kubernetes**: For complex orchestration
5. **Heroku**: Quick deployment (demo)

**Included files**: `Dockerfile` and `docker-compose.yml`

### What is the A/B Testing framework?
```python
from src.ab_testing import ABTest

# Create experiment
experiment = ABTest("new_algorithm", ["control", "treatment"])

# Assign variant (consistent hashing)
variant = experiment.assign_variant(user_id)

# Track conversion
experiment.track_conversion(user_id, variant, converted=True)

# View results
experiment.print_summary()
# Output: Uplift, p-value, confidence interval
```

---

## Evaluation and Metrics

### What metrics are used?
**Basic metrics:**
- Precision@K
- Recall@K

**Advanced metrics:**
- MAP@K (Mean Average Precision)
- NDCG@K (Normalized Discounted Cumulative Gain)
- Coverage (% of catalog recommended)
- Diversity (category variety)
- Novelty (self-information)
- F1@K (harmonic mean of precision/recall)

### How is the model evaluated?
**Temporal Train/Test Split:**
- 80% training (earlier orders)
- 20% testing (most recent orders)
- Time-based split ensures realistic evaluation

```bash
python -m src.main --evaluate --data_dir datos
```

### What is MAP@K and why is it important?
**MAP@K = Mean Average Precision at K**

Metric that considers:
1. Whether recommended items are relevant
2. The order of recommendations (ranking)
3. Average across all users

**Better than Precision@K** because it penalizes poor rankings.

### How do I compare with baselines?
**Implemented baselines:**
1. Random
2. Global popularity
3. Same-category popularity
4. Item-item collaborative filtering

**Comparison:**
```python
from src.evaluate import Evaluator

evaluator = Evaluator(loader, model)
results = evaluator.compare_baselines()
```

### What does "Coverage" mean?
**Coverage** = Percentage of catalog that the model recommends

- Low coverage: Only recommends few popular products
- High coverage: Diversity of recommended products

**Desirable**: Balance between precision and coverage (trade-off)

---

## Common Issues

### Error: Module not found
```bash
# Make sure you're in the project root
cd Gerarld
pip install -r requirements.txt
```

### Error: Data not found
```bash
# Verify that CSVs exist
ls datos/
# You should see: olist_orders_dataset.csv, etc.

# Use --data_dir if they're in another location
python -m src.main --analytics --data_dir /path/to/data
```

### Error: API won't start
```bash
# Install FastAPI dependencies
pip install fastapi uvicorn pydantic
```

### Model takes too long to train
**Optimizations:**
1. Reduce dataset size (subset for testing)
2. Adjust co-purchase parameters (min_support)
3. Use caching for intermediate results
4. Consider using sparse matrices

### MemoryError when loading data
```python
# Option 1: Load only necessary columns
loader = OlistDataLoader(data_dir='datos')
loader.load_orders(usecols=['order_id', 'customer_id'])

# Option 2: Use chunks
for chunk in pd.read_csv('datos/orders.csv', chunksize=10000):
    process(chunk)
```

### How do I debug recommendations?
```python
from src.advanced_models import AdvancedRecommender

model = AdvancedRecommender(loader)

# Explainability
explanation = model.explain_recommendation(customer_id, product_id)
print(explanation)
# Output: {
#   'reasons': ['popular', 'category_match'],
#   'confidence': 0.85,
#   'similar_products': [...]
# }
```

### API returns error 500
**Checklist:**
1. Is the model trained? (`models/recommender.pkl` exists?)
2. Is data loaded correctly?
3. Does customer ID exist in the dataset?
4. Are there logs with the specific error?

```bash
# View detailed logs
uvicorn src.api:app --reload --log-level debug
```

---

## Business Questions

### What is the expected ROI of this system?
**Measurable impacts:**
1. **Conversion increase**: 5-15% with personalized recommendations
2. **AOV increase**: 10-20% with effective cross-sell
3. **Retention**: Reducing churn from 96.88% to 90% = millions in revenue
4. **Efficiency**: Automation vs manual curation

**Approximate calculation** (96K customers, AOV $138):
- 1% improvement in repeat purchase = 960 customers
- 960 × $138 = $132,480 additional revenue
- 5% improvement = $662,400 additional revenue

### What business actions are recommended?
**Immediate (30 days):**
1. Launch second-purchase campaign
2. Repeat purchase rate dashboard as primary KPI
3. Subscription pilot in Health & Beauty

**Medium-term (90 days):**
1. Loyalty program with points
2. Personalization engine with browsing data
3. Brand awareness in customer journey

### How does it integrate with existing systems?
**Integration options:**
1. **REST API**: Calls from frontend/backend
2. **Batch processing**: Pre-compute recommendations offline
3. **Event-driven**: Kafka/Kinesis for real-time
4. **Embedded**: Export model for in-app scoring

### How is it monitored in production?
**Metrics to track:**
1. API latency (p50, p95, p99)
2. Error rate (4xx, 5xx)
3. Precision@K in production (online metrics)
4. Click-through rate (CTR) of recommendations
5. Conversion rate of recommended products
6. Catalog coverage

**Tools:**
- Structured logs (already implemented)
- Prometheus + Grafana
- Datadog / New Relic
- A/B testing framework (included)

---

## Additional Resources

### Where can I find more information?
- **Quick setup**: `QUICKSTART.md`
- **Advanced guide**: `README_ADVANCED.md`
- **Implementation details**: `IMPLEMENTATION_SUMMARY.md`
- **Business insights**: `ANALYTICS_SUMMARY.md`
- **API documentation**: http://localhost:8000/docs (with API running)

### How do I contribute or extend the project?
**Suggested next improvements:**
1. Deep learning (NCF, transformers)
2. Real-time streaming (Kafka + Flink)
3. GraphQL API
4. Multi-armed bandits
5. Reinforcement learning
6. Edge deployment

**Modular structure** facilitates extension - just add new module in `src/`

### Are there usage examples?
**Yes, several scripts in `notebooks/`:**
- `01_data_exploration.py`: Data exploration
- `02_verify_repeat_customers.py`: Metrics validation
- `03_quick_demo.py`: Quick feature demo

---

## Contact and Support

### Who do I contact for technical questions?
This project was developed as part of a Data Science technical challenge. For questions:
1. Review this FAQ first
2. Consult documentation in MD files
3. Review tests for usage examples
4. Inspect source code (well documented)

### Where do I report bugs?
For this specific project, create a document with:
1. Error description
2. Steps to reproduce
3. Expected vs actual behavior
4. Relevant logs
5. Environment (OS, Python version, etc.)

---

**Last updated**: November 2025
