 Recommended Approach

🎯 General Strategy

This challenge evaluates 3 pillars: Analytics (30%), Modeling (30%), and Code Quality (30%). I recommend balancing your time between them and prioritizing simplicity and clarity over complexity.

🗓️ Recommended Stages (2–3 hours)

**Stage 1: Setup & Exploration (30 min)**

1. Download and review the data

   * Prioritize: orders, order_items, products, customers
   * Explore structure, missing values, relationships between tables

2. Project setup

   * Create folder structure according to the template
   * Set up a virtual environment
   * Basic `requirements.txt`: pandas, numpy, scikit-learn, pytest

---

**Stage 2: Analytics (45 min)**

Key goals:

* Top categories by orders and GMV (Gross Merchandise Value)
* Repeat purchase rate
* Average time between orders
* Review score distribution

Tips:

* Use simple joins between dataframes
* Compute aggregated metrics
* Identify 2 non-obvious insights with business impact
* Example insight: “70% of customers never make a repeat purchase → retention opportunity”

---

**Stage 3: Modeling (45–60 min)**

Recommendation: Start with **RECOMMENDATION** (easier than prediction).

Suggested approach:

1. Simple baseline: Popularity-based (best-selling products)
2. Improvement: Co-purchase (customers who bought X also bought Y)
3. Metric: Precision@K or MAP@K
4. Evaluation: Train/test split by date or by customer

Code structure:

```text
src/
├── data_loader.py    # Load CSVs
├── model.py          # RecommenderModel class
├── evaluate.py       # precision_at_k()
└── main.py           # CLI
```

---

**Stage 4: Production Code (30 min)**

1. Working CLI:

```bash
python -m src.main --customer_id <ID> --top_k 5
```

2. A simple test:

```python
def test_model_returns_correct_number():
    model = RecommenderModel()
    recs = model.recommend(customer_id, top_k=5)
    assert len(recs) == 5
```

3. Save the model:

   * Model pickle or JSON with the co-purchase matrix

---

**Stage 5: Documentation (15 min)**

1. `README.md`: setup, how to run, how to test
2. Analytics summary (1 page): KPIs + 2 insights + simple visualizations

---

✨ Key Differentiators

To stand out:

* ✅ Modular code with well-defined classes
* ✅ Insights with clear business impact
* ✅ Well-justified evaluation metric
* ✅ Tests that validate critical logic
* ✅ Clear, reproducible README

Avoid:

* ❌ Notebooks as the only deliverable
* ❌ Complex models without a baseline
* ❌ Over-engineering (KISS principle)

---

🚀 Quick Wins

1. Analytics: Review score distribution plot + top categories table
2. Model: Popularity baseline + co-purchase matrix
3. Code: Template-based structure + 2–3 basic tests
4. Docs: README with copy-paste commands + summary with clear bullets

  - Implementar el modelo de recomendación

  - Configurar los tests
