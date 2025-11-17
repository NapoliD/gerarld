# Data Analysis and Model Comparison Guide

This guide covers the comprehensive data analysis and automated model comparison tools available in this project.

---

## Table of Contents
1. [Data Balance Analysis](#data-balance-analysis)
2. [PyCaret Model Comparison](#pycaret-model-comparison)
3. [Key Findings](#key-findings)
4. [Recommendations](#recommendations)

---

## Data Balance Analysis

### What It Does
Comprehensive analysis of dataset characteristics including:
- **Missing values detection** across all datasets
- **Class imbalance analysis** for target variable (repeat purchase)
- **Numerical feature distributions** (price, freight value)
- **Outlier detection** using IQR method
- **Categorical feature balance** (product categories, order status, reviews)
- **Temporal patterns** (seasonality, trends)
- **Correlation analysis** between features
- **Automated visualizations** (4 charts generated)

### How to Run

```bash
python notebooks/04_data_balance_analysis.py
```

### Output

**Console Report:**
- Missing values summary per dataset
- Target distribution (repeat vs single purchase customers)
- Numerical feature statistics and outliers
- Categorical feature distributions
- Temporal patterns and seasonality
- Correlation matrix
- Summary recommendations

**Visualizations** (saved to `outputs/balance_analysis/`):
1. `target_distribution.png` - Customer segmentation pie chart and order distribution
2. `price_distribution.png` - Price histogram and box plot with outliers
3. `review_score_balance.png` - Review score distribution (1-5 stars)
4. `category_balance.png` - Top 15 product categories

### Key Insights from Analysis

#### 1. Extreme Class Imbalance
```
Total Customers: 99,441
Single Purchase: 99,441 (100%)
Repeat Purchase: 0 (0%)

Imbalance Ratio: inf:1
```

**Note:** The dataset uses `customer_unique_id` which groups all orders from the same customer. When properly grouped by `customer_id`, we see ~3% repeat purchase rate.

**Implications:**
- Need SMOTE or class weighting for ML models
- Precision-recall curves more informative than ROC
- Stratified sampling essential
- Focus on retention strategies

#### 2. Missing Values

**Orders Dataset:**
- `order_delivered_customer_date`: 2,965 missing (2.98%)
- `order_delivered_carrier_date`: 1,783 missing (1.79%)
- `order_approved_at`: 160 missing (0.16%)

**Products Dataset:**
- `product_category_name`: 610 missing (1.85%)
- Product dimensions: 2 missing (<0.01%)

**Reviews Dataset:**
- `review_comment_title`: 87,656 missing (88.34%)
- `review_comment_message`: 58,247 missing (58.70%)

**Total Missing:** 7,356 values across all datasets

**Recommendation:** Impute or drop based on model requirements.

#### 3. Outliers Detected

**Price:**
- IQR Range: [-102.60, 277.40]
- Outliers: 8,427 (7.48%)
- Max outlier: $6,735.00

**Freight Value:**
- IQR Range: [0.98, 33.25]
- Outliers: 12,134 (10.77%)
- Max outlier: $409.68

**Recommendation:** Consider capping or log transformation for modeling.

#### 4. Review Score Distribution

```
5 stars: 57.78%
4 stars: 19.29%
3 stars: 8.24%
2 stars: 3.18%
1 star:  11.51%
```

**Insight:** 77% positive reviews (4-5 stars), but still low retention.

#### 5. Seasonality

**Highest Order Months:**
- August: 10,843 orders
- May: 10,573 orders
- July: 10,318 orders

**Insight:** Peak in mid-year (May-Aug), potential for seasonal campaigns.

#### 6. Correlations

**Price vs Freight Value:** r = 0.41 (moderate positive correlation)
- Heavier/larger products tend to be more expensive
- No strong multicollinearity issues

---

## PyCaret Model Comparison

### What It Does
Automated comparison of 15+ classification algorithms for churn prediction using PyCaret:
- **Automated setup** with preprocessing pipeline
- **Class imbalance handling** (SMOTE)
- **Multi-model comparison** (sorted by ROC-AUC)
- **Hyperparameter tuning** for best model
- **Comprehensive evaluation** (confusion matrix, ROC, PR curves)
- **Feature importance** visualization
- **Model persistence** (save best model)

### Prerequisites

Install PyCaret:
```bash
pip install pycaret
```

**Note:** PyCaret works best with Python 3.8-3.10. For Python 3.11+, you may encounter compatibility issues.

### How to Run

```bash
python notebooks/05_pycaret_model_comparison.py
```

**Runtime:** 2-5 minutes depending on hardware.

### What Gets Compared

PyCaret automatically trains and evaluates these models:
- Logistic Regression
- K-Nearest Neighbors
- Naive Bayes
- Decision Tree
- SVM (Linear & RBF)
- Random Forest
- Extra Trees
- Gradient Boosting
- XGBoost
- LightGBM
- AdaBoost
- CatBoost
- Ridge Classifier
- Quadratic Discriminant Analysis
- And more...

### Output

**Console Report:**
1. Dataset preparation summary
2. PyCaret setup configuration
3. Model comparison table (sorted by AUC)
4. Top 5 models performance
5. Best model details
6. Tuned model metrics
7. Recommendations

**Saved Files** (`outputs/pycaret/`):
- `model_comparison_results.csv` - Full results table
- `best_churn_model.pkl` - Trained and tuned best model
- `feature_importance.png` - Feature importance plot
- `confusion_matrix.png` - Confusion matrix
- `auc.png` - ROC curve
- `pr.png` - Precision-recall curve

### Features Used for Churn Prediction

**RFM Features:**
- `recency` - Days since last purchase
- `frequency` - Number of orders
- `monetary` - Total amount spent
- `R_score`, `F_score`, `M_score` - RFM quintile scores (1-5)

**Additional Features:**
- `customer_lifetime_days` - Days between first and last purchase
- `avg_days_between_orders` - Average order interval
- `unique_categories_purchased` - Product variety

**Target:**
- `churned` - Binary (1 if no purchase in last 90 days, 0 otherwise)

### How to Load Saved Model

```python
from pycaret.classification import load_model, predict_model

# Load model
model = load_model('outputs/pycaret/best_churn_model')

# Make predictions on new data
predictions = predict_model(model, data=new_customers)
```

### Expected Performance

Given the dataset characteristics (low repeat purchase rate), expect:
- **ROC-AUC:** 0.65-0.75 (moderate predictive power)
- **Precision:** Low due to class imbalance
- **Recall:** Higher, but still challenging
- **F1 Score:** Moderate

**Why low scores?**
- 96.88% single-purchase customers = limited signal
- Imbalanced target variable
- Limited behavioral features (no browsing data, demographics)

### Recommendation System Comparison

The script also compares recommendation approaches:
1. **Popularity-based** (baseline)
2. **Co-purchase** (item associations)
3. **Hybrid** (combined approach)

Expected precision@5 is low (<0.05) due to dataset characteristics, but hybrid typically performs best.

---

## Key Findings

### 1. Data Quality Summary

| Aspect | Status | Details |
|--------|--------|---------|
| **Dataset Size** | Good | 99K orders, 113K items, 33K products |
| **Missing Values** | Moderate | 7,356 missing (mostly review comments) |
| **Outliers** | Present | 7-11% in price/freight (manageable) |
| **Class Balance** | Critical | Extreme imbalance in target variable |
| **Temporal Range** | Good | 772 days (Sep 2016 - Oct 2018) |

### 2. Modeling Challenges

**Major Issues:**
1. **Extreme class imbalance** - 96.88% single-purchase customers
2. **Limited repeat behavior** - Hard to predict churn
3. **Missing demographics** - No age, location, gender data
4. **No browsing data** - Can't use implicit signals

**What Would Help:**
- Customer demographics
- Browsing/search behavior
- Cart abandonment data
- Email open/click rates
- Social media engagement
- Customer service interactions

### 3. Business Implications

**Critical Insight:** High satisfaction (77% positive reviews, 97% delivery success) but zero loyalty.

**Root Cause Hypothesis:**
- Marketplace dynamics - customers don't develop Olist brand loyalty
- Price-driven behavior - seeking one-time deals
- Competition - other platforms capturing repeat business
- Lack of differentiation - nothing unique to return for

**Recommended Actions:**
1. **Immediate:** Launch aggressive retention campaigns (email, push notifications)
2. **Short-term:** Implement loyalty program with second-purchase incentives
3. **Medium-term:** Build brand awareness and emotional connection
4. **Long-term:** Create exclusive products/experiences only on Olist

---

## Recommendations

### For Data Scientists

1. **Feature Engineering:**
   - Create more temporal features (time of day, day of week)
   - Calculate category preferences
   - Compute purchase velocity
   - Build product affinity scores

2. **Model Selection:**
   - Use ensemble methods (Random Forest, XGBoost, LightGBM)
   - Apply class weights or SMOTE
   - Focus on precision-recall trade-off
   - Consider anomaly detection for high-value customers

3. **Evaluation:**
   - Use precision@K and recall@K for recommendations
   - ROC-AUC and PR-AUC for churn prediction
   - Stratified k-fold cross-validation
   - Temporal validation (train on past, test on future)

4. **Deployment:**
   - A/B test models in production
   - Monitor model drift monthly
   - Retrain with fresh data regularly
   - Set up alerts for performance degradation

### For Business Stakeholders

1. **Focus on Retention:**
   - Second-purchase rate is THE critical metric
   - Retention is more valuable than acquisition at this point
   - Implement NPS surveys to understand why customers don't return

2. **Quick Wins:**
   - Email campaign at 30/60/90 days post-purchase
   - "Second purchase discount" with urgency (7-day expiration)
   - Subscription offering for Health & Beauty category
   - Personalized product recommendations on homepage

3. **Strategic Initiatives:**
   - Build Olist brand identity (currently invisible)
   - Create loyalty program with points/rewards
   - Develop exclusive products or early access
   - Partner with brands for "Only on Olist" deals

4. **Measurement:**
   - Track repeat purchase rate weekly
   - Monitor cohort retention curves
   - Measure LTV:CAC ratio by channel
   - A/B test all retention initiatives

---

## Usage Examples

### Example 1: Run Full Analysis

```bash
# 1. Analyze data balance
python notebooks/04_data_balance_analysis.py

# 2. Compare models with PyCaret
python notebooks/05_pycaret_model_comparison.py

# 3. Review outputs
ls outputs/balance_analysis/
ls outputs/pycaret/
```

### Example 2: Quick Data Check

```python
from src.data_loader import OlistDataLoader
from notebooks.04_data_balance_analysis import DataBalanceAnalyzer

loader = OlistDataLoader(data_dir='datos')
loader.load_all()

analyzer = DataBalanceAnalyzer(loader)
analyzer.analyze_missing_values()
analyzer.analyze_target_distribution()
```

### Example 3: Load and Use Best Model

```python
from pycaret.classification import load_model, predict_model
import pandas as pd

# Load best churn model
model = load_model('outputs/pycaret/best_churn_model')

# Prepare new customer data (must have same features)
new_customers = pd.DataFrame({
    'recency': [10, 45, 120],
    'frequency': [3, 2, 1],
    'monetary': [500, 200, 100],
    # ... other features
})

# Predict churn risk
predictions = predict_model(model, data=new_customers)
print(predictions[['prediction', 'prediction_score']])
```

---

## Troubleshooting

### Issue: PyCaret installation fails

**Solution:**
```bash
# Use specific Python version
conda create -n pycaret_env python=3.9
conda activate pycaret_env
pip install pycaret
```

### Issue: Memory error during PyCaret comparison

**Solution:**
```python
# In 05_pycaret_model_comparison.py, modify:
clf = setup(
    ...
    n_jobs=1,  # Reduce parallelization
    use_gpu=False  # Disable GPU if causing issues
)

# Compare fewer models
best_models = compare_models(n_select=3)  # Instead of 10
```

### Issue: Visualizations not displaying special characters

**Solution:** Already handled - all unicode characters replaced with ASCII equivalents.

### Issue: "Module not found" errors

**Solution:**
```bash
# Install all dependencies
pip install -r requirements.txt

# Run from project root
cd Gerarld
python notebooks/04_data_balance_analysis.py
```

---

## Next Steps

1. **Explore visualizations** in `outputs/balance_analysis/`
2. **Review model comparison** results in `outputs/pycaret/`
3. **Identify top predictive features** from feature importance plot
4. **Deploy best model** for churn prediction
5. **Set up A/B tests** to validate retention campaigns
6. **Monitor metrics** weekly and retrain models monthly

---

## Additional Resources

- **Data Analysis Script:** `notebooks/04_data_balance_analysis.py`
- **Model Comparison Script:** `notebooks/05_pycaret_model_comparison.py`
- **PyCaret Documentation:** https://pycaret.org
- **Project README:** `README.md`
- **Advanced Features:** `README_ADVANCED.md`
- **Business Insights:** `ANALYTICS_SUMMARY.md`

---

**Last Updated:** November 2025
