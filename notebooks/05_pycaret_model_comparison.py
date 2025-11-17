"""
PyCaret Model Comparison
Compare multiple classification models for churn prediction using PyCaret.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import OlistDataLoader
from src.features import FeatureEngineer

print("="*80)
print("PYCARET MODEL COMPARISON FOR CHURN PREDICTION")
print("="*80)

# Try to import PyCaret
try:
    from pycaret.classification import *
    print("[OK] PyCaret imported successfully")
except ImportError:
    print("\n[!] PyCaret not installed!")
    print("\nTo install PyCaret, run:")
    print("  pip install pycaret")
    print("\nAlternatively, run:")
    print("  pip install pycaret[full]  # for all dependencies")
    print("\nNote: PyCaret works best with Python 3.8-3.10")
    sys.exit(1)


def prepare_churn_dataset(loader: OlistDataLoader, prediction_window_days=90):
    """
    Prepare dataset for churn prediction.

    Args:
        loader: OlistDataLoader instance
        prediction_window_days: Days to predict churn

    Returns:
        DataFrame with features and target variable
    """
    print("\n" + "-"*80)
    print("PREPARING CHURN PREDICTION DATASET")
    print("-"*80)

    # Get customer features
    engineer = FeatureEngineer(loader)
    rfm_features = engineer.calculate_rfm()

    # Convert timestamp
    orders = loader.orders.copy()
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])

    # Calculate churn target
    max_date = orders['order_purchase_timestamp'].max()
    last_purchase = orders.groupby('customer_id')['order_purchase_timestamp'].max()

    # Define churn: no purchase in last N days
    days_since_purchase = (max_date - last_purchase).dt.days
    churned = (days_since_purchase > prediction_window_days).astype(int)

    # Merge with RFM features
    data = rfm_features.join(churned.rename('churned'))

    # Add additional features
    customer_orders = orders.groupby('customer_id').agg({
        'order_id': 'count',
        'order_purchase_timestamp': ['min', 'max']
    })
    customer_orders.columns = ['total_orders', 'first_purchase', 'last_purchase']

    # Calculate customer lifetime (days)
    customer_orders['customer_lifetime_days'] = (
        customer_orders['last_purchase'] - customer_orders['first_purchase']
    ).dt.days

    # Calculate average days between orders
    customer_orders['avg_days_between_orders'] = (
        customer_orders['customer_lifetime_days'] / customer_orders['total_orders'].clip(lower=1)
    )

    # Merge all features
    data = data.join(customer_orders[['customer_lifetime_days', 'avg_days_between_orders']])

    # Fill NaN values
    data['customer_lifetime_days'] = data['customer_lifetime_days'].fillna(0)
    data['avg_days_between_orders'] = data['avg_days_between_orders'].fillna(0)

    # Add category preferences
    order_items = loader.order_items.merge(
        loader.products[['product_id', 'product_category_name']],
        on='product_id',
        how='left'
    )
    order_items = order_items.merge(
        loader.orders[['order_id', 'customer_id']],
        on='order_id',
        how='left'
    )

    # Count unique categories per customer
    unique_categories = order_items.groupby('customer_id')['product_category_name'].nunique()
    data = data.join(unique_categories.rename('unique_categories_purchased'))
    data['unique_categories_purchased'] = data['unique_categories_purchased'].fillna(0)

    # Remove any remaining NaN
    data = data.fillna(data.median())

    print(f"\n[OK] Dataset prepared:")
    print(f"  Total samples: {len(data):,}")
    print(f"  Features: {len(data.columns) - 1}")
    print(f"  Target variable: churned")
    print(f"\n  Class distribution:")
    print(f"    Not churned (0): {(data['churned'] == 0).sum():,} ({(data['churned'] == 0).sum()/len(data)*100:.1f}%)")
    print(f"    Churned (1): {(data['churned'] == 1).sum():,} ({(data['churned'] == 1).sum()/len(data)*100:.1f}%)")

    return data


def run_pycaret_comparison(data, output_dir='outputs/pycaret'):
    """
    Run PyCaret model comparison.

    Args:
        data: DataFrame with features and target
        output_dir: Directory to save results
    """
    print("\n" + "-"*80)
    print("PYCARET SETUP")
    print("-"*80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Setup PyCaret
    print("\nInitializing PyCaret environment...")
    print("This may take a minute...")

    clf = setup(
        data=data,
        target='churned',
        session_id=42,
        train_size=0.8,
        fix_imbalance=True,  # Important: handle class imbalance
        remove_multicollinearity=True,
        multicollinearity_threshold=0.95,
        normalize=True,
        transformation=True,
        ignore_low_variance=True,
        silent=True,
        verbose=False,
        html=False
    )

    print("\n[OK] PyCaret setup complete!")

    # Compare all models
    print("\n" + "-"*80)
    print("COMPARING MODELS")
    print("-"*80)
    print("\nThis will train and evaluate 15+ models...")
    print("Please wait (this may take 2-5 minutes)...\n")

    best_models = compare_models(
        n_select=10,  # Select top 10 models
        sort='AUC',   # Sort by ROC-AUC
        verbose=True
    )

    # Get comparison results
    results = pull()

    # Save results
    results_file = output_path / 'model_comparison_results.csv'
    results.to_csv(results_file)
    print(f"\n[OK] Results saved to: {results_file}")

    # Display top 5 models
    print("\n" + "-"*80)
    print("TOP 5 MODELS")
    print("-"*80)
    print("\n", results.head())

    # Create and evaluate best model
    print("\n" + "-"*80)
    print("CREATING BEST MODEL")
    print("-"*80)

    best_model = create_model(best_models[0], verbose=False)

    # Tune best model
    print("\nTuning hyperparameters...")
    tuned_model = tune_model(best_model, n_iter=10, optimize='AUC', verbose=False)

    # Evaluate on test set
    print("\n" + "-"*80)
    print("FINAL MODEL EVALUATION")
    print("-"*80)

    # Get predictions on test set
    predictions = predict_model(tuned_model)

    # Get evaluation metrics
    print("\nHold-out Test Set Performance:")
    evaluate_model(tuned_model)

    # Plot feature importance
    print("\nFeature Importance:")
    try:
        plot_model(tuned_model, plot='feature', save=True)
        print(f"[OK] Feature importance plot saved")
    except:
        print("[!]  Feature importance plot not available for this model type")

    # Plot confusion matrix
    print("\nConfusion Matrix:")
    plot_model(tuned_model, plot='confusion_matrix', save=True)
    print(f"[OK] Confusion matrix saved")

    # Plot ROC curve
    print("\nROC Curve:")
    plot_model(tuned_model, plot='auc', save=True)
    print(f"[OK] ROC curve saved")

    # Plot precision-recall curve
    print("\nPrecision-Recall Curve:")
    plot_model(tuned_model, plot='pr', save=True)
    print(f"[OK] Precision-recall curve saved")

    # Save final model
    model_file = output_path / 'best_churn_model'
    save_model(tuned_model, str(model_file))
    print(f"\n[OK] Best model saved to: {model_file}.pkl")

    # Get final metrics
    final_results = pull()

    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)

    print(f"\nBest Model: {best_models[0].__class__.__name__}")
    print(f"\nKey Metrics (Test Set):")
    print(f"  Accuracy: {final_results['Accuracy'][0]:.4f}")
    print(f"  ROC-AUC: {final_results['AUC'][0]:.4f}")
    print(f"  Precision: {final_results['Prec.'][0]:.4f}")
    print(f"  Recall: {final_results['Recall'][0]:.4f}")
    print(f"  F1 Score: {final_results['F1'][0]:.4f}")

    # Compare with baseline
    print("\n" + "-"*80)
    print("COMPARISON WITH BASELINE")
    print("-"*80)

    baseline_accuracy = (data['churned'] == 0).sum() / len(data)
    improvement = (final_results['Accuracy'][0] - baseline_accuracy) * 100

    print(f"\nBaseline (always predict majority class): {baseline_accuracy:.4f}")
    print(f"Best Model Accuracy: {final_results['Accuracy'][0]:.4f}")
    print(f"Improvement: {improvement:+.2f} percentage points")

    # Recommendations
    print("\n" + "-"*80)
    print("RECOMMENDATIONS")
    print("-"*80)

    print("\n1. Model Selection:")
    print(f"   - Best performing: {best_models[0].__class__.__name__}")
    print(f"   - Consider top 3 models for ensemble")

    print("\n2. Class Imbalance:")
    print(f"   - Used SMOTE for balancing (fix_imbalance=True)")
    print(f"   - Monitor precision-recall, not just accuracy")

    print("\n3. Feature Engineering:")
    print(f"   - RFM features are critical")
    print(f"   - Customer lifetime and purchase frequency matter")

    print("\n4. Next Steps:")
    print(f"   - Deploy best model for churn prediction")
    print(f"   - Set up monitoring for model drift")
    print(f"   - A/B test retention campaigns on high-risk customers")
    print(f"   - Retrain monthly with new data")

    print("\n" + "="*80)

    return tuned_model, results


def compare_recommendation_approaches(loader: OlistDataLoader):
    """
    Compare different recommendation approaches.
    Note: This is a simplified version since recommendations are harder to automate with PyCaret.
    """
    print("\n" + "="*80)
    print("RECOMMENDATION SYSTEM COMPARISON")
    print("="*80)

    from src.model import RecommenderModel
    from src.evaluate import Evaluator

    print("\nThis compares our implemented recommendation approaches:")
    print("  1. Popularity-based (baseline)")
    print("  2. Co-purchase based")
    print("  3. Hybrid approach")

    # Train model
    model = RecommenderModel()
    model.fit(loader)

    # Evaluate
    evaluator = Evaluator(loader, model)

    print("\n" + "-"*80)
    print("BASELINE EVALUATION")
    print("-"*80)

    metrics_pop = evaluator.evaluate_model(method='popularity', top_k=5)
    print(f"\nPopularity-based:")
    print(f"  Precision@5: {metrics_pop['precision@5']:.4f}")
    print(f"  Recall@5: {metrics_pop['recall@5']:.4f}")

    print("\n" + "-"*80)
    print("CO-PURCHASE EVALUATION")
    print("-"*80)

    metrics_copurchase = evaluator.evaluate_model(method='copurchase', top_k=5)
    print(f"\nCo-purchase based:")
    print(f"  Precision@5: {metrics_copurchase['precision@5']:.4f}")
    print(f"  Recall@5: {metrics_copurchase['recall@5']:.4f}")

    print("\n" + "-"*80)
    print("HYBRID EVALUATION")
    print("-"*80)

    metrics_hybrid = evaluator.evaluate_model(method='hybrid', top_k=5)
    print(f"\nHybrid approach:")
    print(f"  Precision@5: {metrics_hybrid['precision@5']:.4f}")
    print(f"  Recall@5: {metrics_hybrid['recall@5']:.4f}")

    print("\n" + "-"*80)
    print("RECOMMENDATION")
    print("-"*80)

    best_method = max(
        [('Popularity', metrics_pop['precision@5']),
         ('Co-purchase', metrics_copurchase['precision@5']),
         ('Hybrid', metrics_hybrid['precision@5'])],
        key=lambda x: x[1]
    )

    print(f"\nBest performing method: {best_method[0]}")
    print(f"Precision@5: {best_method[1]:.4f}")

    print("\nNote: Low precision is expected due to:")
    print("  - 96.88% single-purchase customers")
    print("  - Limited repeat purchase data")
    print("  - One-time purchase behavior")


def main():
    """Run complete PyCaret model comparison."""
    print("\n" + "="*80)
    print("STARTING PYCARET MODEL COMPARISON")
    print("="*80)

    # Load data
    print("\nLoading data...")
    loader = OlistDataLoader(data_dir='datos')
    loader.load_all()

    # Prepare churn dataset
    churn_data = prepare_churn_dataset(loader, prediction_window_days=90)

    # Run PyCaret comparison
    print("\n" + "="*80)
    print("PART 1: CHURN PREDICTION MODEL COMPARISON")
    print("="*80)

    tuned_model, results = run_pycaret_comparison(churn_data)

    # Compare recommendation approaches
    print("\n" + "="*80)
    print("PART 2: RECOMMENDATION SYSTEM COMPARISON")
    print("="*80)

    compare_recommendation_approaches(loader)

    print("\n" + "="*80)
    print("[OK] ANALYSIS COMPLETE!")
    print("="*80)

    print("\nGenerated outputs:")
    print("  - Model comparison results: outputs/pycaret/model_comparison_results.csv")
    print("  - Best model: outputs/pycaret/best_churn_model.pkl")
    print("  - Visualizations: outputs/pycaret/*.png")

    print("\nTo load the best model later:")
    print("  from pycaret.classification import load_model")
    print("  model = load_model('outputs/pycaret/best_churn_model')")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
