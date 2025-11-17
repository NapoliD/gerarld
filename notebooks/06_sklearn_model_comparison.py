"""
Scikit-Learn Model Comparison (PyCaret Alternative)
Compare multiple classification models for churn prediction using scikit-learn.
Works with Python 3.12+
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

# Machine learning imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)

# Import all classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Imbalanced learning
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("SCIKIT-LEARN MODEL COMPARISON FOR CHURN PREDICTION")
print("="*80)
print(f"\nPython version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
print("[OK] All libraries imported successfully\n")


def prepare_churn_dataset(loader: OlistDataLoader, prediction_window_days=90):
    """Prepare dataset for churn prediction."""
    print("-"*80)
    print("PREPARING CHURN PREDICTION DATASET")
    print("-"*80)

    # Get customer features
    engineer = FeatureEngineer(loader)
    rfm_features = engineer.compute_rfm_features()

    # Convert timestamp
    orders = loader.orders.copy()
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])

    # Merge orders with customers to get customer_unique_id
    orders = orders.merge(
        loader.customers[['customer_id', 'customer_unique_id']],
        on='customer_id',
        how='left'
    )

    # Calculate churn target using customer_unique_id
    max_date = orders['order_purchase_timestamp'].max()
    last_purchase = orders.groupby('customer_unique_id')['order_purchase_timestamp'].max()

    # Define churn: no purchase in last N days
    days_since_purchase = (max_date - last_purchase).dt.days
    churned = (days_since_purchase > prediction_window_days).astype(int)

    # Merge with RFM features (both use customer_unique_id as index)
    data = rfm_features.join(churned.rename('churned'), how='inner')

    # Add additional features using customer_unique_id
    customer_orders = orders.groupby('customer_unique_id').agg({
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
    data = data.join(customer_orders[['customer_lifetime_days', 'avg_days_between_orders']], how='left')

    # Fill NaN values
    data['customer_lifetime_days'] = data['customer_lifetime_days'].fillna(0)
    data['avg_days_between_orders'] = data['avg_days_between_orders'].fillna(0)

    # Add category preferences using customer_unique_id
    order_items = loader.order_items.merge(
        loader.products[['product_id', 'product_category_name']],
        on='product_id',
        how='left'
    )
    order_items = order_items.merge(
        orders[['order_id', 'customer_unique_id']],
        on='order_id',
        how='left'
    )

    # Count unique categories per customer
    unique_categories = order_items.groupby('customer_unique_id')['product_category_name'].nunique()
    data = data.join(unique_categories.rename('unique_categories_purchased'), how='left')
    data['unique_categories_purchased'] = data['unique_categories_purchased'].fillna(0)

    # Remove any remaining NaN (only for numerical columns)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

    print(f"\n[OK] Dataset prepared:")
    print(f"  Total samples: {len(data):,}")
    print(f"  Features: {len(data.columns) - 1}")
    print(f"  Target variable: churned")
    print(f"\n  Class distribution:")
    print(f"    Not churned (0): {(data['churned'] == 0).sum():,} ({(data['churned'] == 0).sum()/len(data)*100:.1f}%)")
    print(f"    Churned (1): {(data['churned'] == 1).sum():,} ({(data['churned'] == 1).sum()/len(data)*100:.1f}%)")

    return data


def get_models():
    """Get dictionary of models to compare."""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42, algorithm='SAMME'),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=42),
    }
    return models


def compare_models(X_train, X_test, y_train, y_test, use_smote=True):
    """Compare multiple models and return results."""
    print("\n" + "-"*80)
    print("COMPARING MODELS")
    print("-"*80)

    if use_smote:
        print("\n[OK] Using SMOTE to handle class imbalance")

    models = get_models()
    results = []

    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\nTraining {name}...", end=" ")

        try:
            if use_smote:
                # Create pipeline with SMOTE
                pipeline = ImbPipeline([
                    ('scaler', StandardScaler()),
                    ('smote', SMOTE(random_state=42)),
                    ('classifier', model)
                ])
            else:
                # Just scaler and model
                from sklearn.pipeline import Pipeline
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', model)
                ])

            # Cross-validation
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

            # Train on full training set
            pipeline.fit(X_train, y_train)

            # Predictions
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'ROC-AUC': roc_auc,
                'CV ROC-AUC Mean': cv_scores.mean(),
                'CV ROC-AUC Std': cv_scores.std(),
                'Pipeline': pipeline
            })

            print(f"[OK] ROC-AUC: {roc_auc:.4f}")

        except Exception as e:
            print(f"[FAILED] {str(e)}")
            continue

    # Create results dataframe
    results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)

    return results_df


def plot_results(results_df, output_dir='outputs/sklearn_comparison'):
    """Create visualizations of model comparison."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Model comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    metrics = ['Accuracy', 'Precision', 'Recall', 'ROC-AUC']

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        data = results_df.sort_values(metric, ascending=True)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(data)))

        ax.barh(data['Model'], data[metric], color=colors)
        ax.set_xlabel(metric, fontsize=12)
        ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)

        # Add values on bars
        for i, v in enumerate(data[metric]):
            ax.text(v, i, f' {v:.3f}', va='center')

    plt.tight_layout()
    plt.savefig(output_path / 'model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path / 'model_comparison.png'}")
    plt.close()

    # 2. CV scores plot
    plt.figure(figsize=(12, 6))

    data = results_df.sort_values('CV ROC-AUC Mean', ascending=False)
    x = range(len(data))

    plt.errorbar(x, data['CV ROC-AUC Mean'], yerr=data['CV ROC-AUC Std'],
                 fmt='o', capsize=5, capthick=2, markersize=8)
    plt.xticks(x, data['Model'], rotation=45, ha='right')
    plt.ylabel('ROC-AUC Score', fontsize=12)
    plt.title('Cross-Validation ROC-AUC Scores (Mean ± Std)', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'cv_scores.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path / 'cv_scores.png'}")
    plt.close()


def evaluate_best_model(best_pipeline, X_test, y_test, feature_names, output_dir='outputs/sklearn_comparison'):
    """Detailed evaluation of best model."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "-"*80)
    print("BEST MODEL DETAILED EVALUATION")
    print("-"*80)

    # Predictions
    y_pred = best_pipeline.predict(X_test)
    y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Churned', 'Churned'],
                yticklabels=['Not Churned', 'Churned'])
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path / 'confusion_matrix.png'}")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'roc_curve.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path / 'roc_curve.png'}")
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkgreen', lw=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path / 'precision_recall_curve.png'}")
    plt.close()

    # Feature importance (if available)
    try:
        # Get the classifier from the pipeline
        classifier = best_pipeline.named_steps['classifier']

        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            indices = np.argsort(importances)[::-1][:15]  # Top 15

            plt.figure(figsize=(10, 6))
            plt.barh(range(len(indices)), importances[indices], color='steelblue')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Feature Importance', fontsize=12)
            plt.title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
            print(f"[OK] Saved: {output_path / 'feature_importance.png'}")
            plt.close()
        elif hasattr(classifier, 'coef_'):
            importances = np.abs(classifier.coef_[0])
            indices = np.argsort(importances)[::-1][:15]  # Top 15

            plt.figure(figsize=(10, 6))
            plt.barh(range(len(indices)), importances[indices], color='steelblue')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Coefficient Magnitude', fontsize=12)
            plt.title('Top 15 Feature Coefficients (Absolute)', fontsize=14, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
            print(f"[OK] Saved: {output_path / 'feature_importance.png'}")
            plt.close()
    except:
        print("[!] Feature importance not available for this model type")


def main():
    """Run complete model comparison."""
    print("\nStarting Model Comparison...")

    # Load data
    print("\nLoading data...")
    loader = OlistDataLoader(data_dir='datos')
    loader.load_all()

    # Prepare dataset
    data = prepare_churn_dataset(loader, prediction_window_days=90)

    # Split features and target
    X = data.drop('churned', axis=1)
    y = data['churned']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n[OK] Train/Test Split:")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Test samples: {len(X_test):,}")

    # Compare models
    results_df = compare_models(X_train, X_test, y_train, y_test, use_smote=True)

    # Display results
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print("\n", results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']].to_string(index=False))

    # Save results
    output_path = Path('outputs/sklearn_comparison')
    output_path.mkdir(parents=True, exist_ok=True)

    results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'CV ROC-AUC Mean', 'CV ROC-AUC Std']].to_csv(
        output_path / 'model_comparison_results.csv', index=False
    )
    print(f"\n[OK] Results saved to: {output_path / 'model_comparison_results.csv'}")

    # Plot results
    plot_results(results_df)

    # Evaluate best model
    best_model_row = results_df.iloc[0]
    best_model_name = best_model_row['Model']
    best_pipeline = best_model_row['Pipeline']

    print(f"\n[OK] Best Model: {best_model_name}")
    print(f"  ROC-AUC: {best_model_row['ROC-AUC']:.4f}")

    evaluate_best_model(best_pipeline, X_test, y_test, X.columns.tolist())

    # Save best model
    import joblib
    model_file = output_path / 'best_model.pkl'
    joblib.dump(best_pipeline, model_file)
    print(f"\n[OK] Best model saved to: {model_file}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\nBest Model: {best_model_name}")
    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {best_model_row['Accuracy']:.4f}")
    print(f"  Precision: {best_model_row['Precision']:.4f}")
    print(f"  Recall: {best_model_row['Recall']:.4f}")
    print(f"  F1 Score: {best_model_row['F1']:.4f}")
    print(f"  ROC-AUC: {best_model_row['ROC-AUC']:.4f}")

    print("\nGenerated outputs:")
    print("  - Model comparison results: outputs/sklearn_comparison/model_comparison_results.csv")
    print("  - Best model: outputs/sklearn_comparison/best_model.pkl")
    print("  - Visualizations: outputs/sklearn_comparison/*.png")

    print("\nTo load the best model later:")
    print("  import joblib")
    print("  model = joblib.load('outputs/sklearn_comparison/best_model.pkl')")
    print("  predictions = model.predict(new_data)")

    print("\n" + "="*80)
    print("[OK] ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
