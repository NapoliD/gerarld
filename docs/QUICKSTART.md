# Quick Start Guide - Olist Advanced Recommendation System

## 5-Minute Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Analytics
```bash
python -m src.main --analytics --data_dir datos
```

### 3. Train Model
```bash
python -m src.main --train --data_dir datos
```

### 4. Get Recommendations
```bash
python -m src.main --customer_id 2c3642e1392097fb4af76a76fec16a46 --top_k 5 --data_dir datos
```

---

## Advanced Features (10 Minutes)

### Visualizations
```python
from src.data_loader import OlistDataLoader
from src.visualizations import OlistVisualizer
from src.analytics import OlistAnalytics

loader = OlistDataLoader(data_dir='datos')
loader.load_all()

analytics = OlistAnalytics(loader)
viz = OlistVisualizer(output_dir='outputs')
viz.create_all_visualizations(loader, analytics)

# Check outputs/ folder for 6 charts!
```

### Customer Segmentation
```python
from src.features import FeatureEngineer

engineer = FeatureEngineer(loader)
segments = engineer.segment_customers_by_rfm()
print(segments['segment'].value_counts())
```

### Churn Prediction
```python
from src.churn_prediction import ChurnPredictor

predictor = ChurnPredictor(loader)
metrics = predictor.train(prediction_window_days=90)
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
```

### Market Basket Analysis
```python
from src.market_basket import MarketBasketAnalyzer

analyzer = MarketBasketAnalyzer(loader)
rules = analyzer.generate_association_rules(min_confidence=0.3)
print(rules.head())
```

---

## API Server (5 Minutes)

### Start Server
```bash
uvicorn src.api:app --reload
```

### Test Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Get recommendations
curl "http://localhost:8000/recommend/customer_123?top_k=5"

# Advanced recommendations
curl "http://localhost:8000/recommend/advanced/customer_123?method=ensemble"

# Customer features
curl http://localhost:8000/features/customer_123
```

### Interactive Docs
Open browser: `http://localhost:8000/docs`

---

## Docker (3 Minutes)

```bash
docker-compose up -d
# API runs on http://localhost:8000
```

---

## Common Commands

### Analytics
```bash
python -m src.main --analytics
```

### Train Model
```bash
python -m src.main --train
```

### Evaluate Model
```bash
python -m src.main --evaluate
```

### Generate Recommendations
```bash
python -m src.main --customer_id <ID> --top_k 5 --method hybrid
```

---

## File Locations

**Data**: `datos/*.csv`
**Models**: `models/recommender.pkl`
**Visualizations**: `outputs/*.png`
**Logs**: Console output

---

## Troubleshooting

### Issue: Module not found
```bash
# Make sure you're in the project root
cd Gerarld
pip install -r requirements.txt
```

### Issue: Data not found
```bash
# Check data directory
ls datos/
# Should show CSV files
```

### Issue: API won't start
```bash
# Install FastAPI dependencies
pip install fastapi uvicorn pydantic
```

---

## Next Steps

1. ✅ Run analytics to understand the data
2. ✅ Train the model
3. ✅ Test recommendations
4. ✅ Generate visualizations
5. ✅ Try advanced features (churn, segmentation)
6. ✅ Start the API server
7. ✅ Read `README_ADVANCED.md` for full documentation

---

## Need Help?

- **Documentation**: See `README_ADVANCED.md`
- **Implementation Details**: See `IMPLEMENTATION_SUMMARY.md`
- **Business Insights**: See `ANALYTICS_SUMMARY.md`
- **API Docs**: Run API and visit `/docs`

---

Enjoy exploring the advanced recommendation system!
