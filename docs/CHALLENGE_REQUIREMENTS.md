# **Data Scientist – Analytics, Modeling & Code**

## **Summary**

A focused take-home that evaluates three things in one coherent task:

1. **Analytics & communication**
2. **Modeling** (Recommendation or Prediction)
3. **Production-quality Python** (modular code + tests + CLI)

## The **Data**

The dataset includes ~100k orders between 2016 and 2018 from Olist Store marketplaces in Brazil. It covers multiple dimensions such as product, category, customer, reviews, payments, and freight. Data are anonymized (no real company references).

[online_store_orders.zip](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f03af478-18a9-423e-9759-d03c96b44321/online_store_orders.zip)

### **Files typically include:**

- olist_orders_dataset.csv
- olist_order_items_dataset.csv
- olist_products_dataset.csv
- olist_customers_dataset.csv
- olist_order_payments_dataset.csv *(optional)*
- olist_order_reviews_dataset.csv *(optional)*
- product_category_name_translation.csv *(optional)*
- olist_sellers_dataset.csv *(optional)*
- olist_geolocation_dataset.csv *(optional)*

👉 You don’t need to use every file—focus on those most useful for your analysis and modeling.

## **🎯 Goals (what we’re testing)**

- **Analytics:** compute meaningful metrics, find non-obvious insights, explain business implications.
- **Modeling:** build a simple, defensible recommendation or prediction model; evaluate with a clear metric.
- **Production-Ready Python:** clean, modular code with a small CLI and at least one unit test.

## **📝 Part 1 — Analytics**

Compute descriptive analytics such as:

- Top product categories by orders and GMV
- Repeat purchase rate per customer; average time between orders
- Conversion view: share of delivered orders vs overall; review score distribution

**Deliver:** A max 1-page summary with ≥2 non-trivial insights and why they matter.

Include at least one KPI (e.g., repeat purchase rate) and consider plots/tables for clarity.

## **🤖 Part 2 — Modeling (pick one)**

- **Recommendation:** Recommend K products for a target customer_id (baseline popularity, co-purchase, or simple ML).
- **Prediction:** Predict a target like 5-star review or on-time delivery.

**Requirements:**

- Put logic in Python modules (no notebook-only).
- Include an evaluation metric (precision@K, MAP@K, ROC-AUC, etc.) and justify it.
- Start with a simple baseline, then improve if time allows.
- Briefly explain your modeling choices and tradeoffs.
- Heavy tuning/deep learning is not expected.

## **⚙️ Part 3 — Production-Ready Code**

Provide a CLI:

```jsx
python -m src.main --customer_id <ID> --top_k 5
```

It should load (or train) the model and print recommendations (or run your prediction path).

**Requirements:**

- Add at least one unit test with pytest.
- Print/log evaluation results in CLI.
- Structure code so an API could be added later.
- Saving/loading a model artifact (pickle/JSON) is a plus.

## **📦 Deliverables**

1. **Codebase** (structured Python project; notebook optional for exploration only).
2. **Analytics Summary** (≤1 page, PDF or markdown).
3. **README** (setup, how to run analytics/model, how to use the CLI, how to run tests).

## **✅ Evaluation Criteria**

- **Analytics & Insights (30%)**: Clear KPIs, correct logic, ≥2 non-obvious insights with business impact.
- **Modeling & Metrics (30%)**: Simple, sensible baseline or model; appropriate metric; honest discussion of tradeoffs.
- **Code Quality (30%)**: Modular, testable, documented; not notebook-only; minimal but meaningful tests.
- **Practicality (10%)**: Reasonable choices for 2-hour scope; easy to extend to an API; reproducible runs.
- **Communication (bonus):** Clarity of explanations in summary/README.

## **🛠 Starter Project (optional)**

We provide a scaffold so you can focus on substance, not wiring.
You may extend or modify it as you see fit—structure matters more than sticking to template.

```jsx
ds_challenge_olist/
├─ data/                     # Put CSVs here (or set DATA_DIR)
├─ notebooks/                # Optional exploration
├─ src/
│  ├─ data_loader.py         # Loads Olist CSVs (glob-tolerant)
│  ├─ model.py               # Baseline recommender / predictor
│  ├─ evaluate.py            # Metrics (e.g. precision@K)
│  └─ main.py                # CLI entrypoint
├─ tests/
│  └─ test_model.py          # Minimal pytest
├─ requirements.txt
└─ README.md
```

**How to run**

```jsx
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export DATA_DIR=./data  # or absolute path to Olist CSVs
python -m src.main --customer_id <some_customer_hash> --top_k 5
pytest -q
```

## **📤 Submission**

- Share a GitHub repo (preferred) or zipped folder with code + summary + README.
- Do not include large data files.
- Note any assumptions or shortcuts you took due to timebox.

## **⏱ Time Expectations**

This challenge is scoped to be completable in **~2-3 hours for a baseline solution**.

- We do **not** expect a production-ready system within that time — focus on clarity, structure, and showing how you approach problems.
- If you’d like to polish your work (e.g., add tests, refine the model, improve the README), **you may take more time**, but this is optional.
- If you run out of time, submit what you have — **quality and communication matter more than completeness**.