"""
Churn prediction model to identify customers at risk of not returning.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pickle
from typing import Dict, Tuple


class ChurnPredictor:
    """Predict customer churn (will they return to purchase?)."""

    def __init__(self, loader):
        """
        Initialize churn predictor.

        Args:
            loader: OlistDataLoader instance.
        """
        self.loader = loader
        self.model = None
        self.feature_names = None
        self.is_fitted = False

    def _prepare_churn_dataset(self, prediction_window_days: int = 90) -> pd.DataFrame:
        """
        Prepare dataset for churn prediction.

        Args:
            prediction_window_days: Days to look ahead for churn.

        Returns:
            DataFrame with features and churn labels.
        """
        orders = self.loader.orders.copy()
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])

        # Merge with customers
        orders = orders.merge(
            self.loader.customers[['customer_id', 'customer_unique_id']],
            on='customer_id',
            how='left'
        )

        # Merge with order values
        order_values = self.loader.order_items.groupby('order_id')['price'].sum().reset_index()
        order_values.columns = ['order_id', 'order_value']
        orders = orders.merge(order_values, on='order_id', how='left')

        # Sort by date
        orders = orders.sort_values('order_purchase_timestamp')

        # For each customer, take all but last purchase for features
        # Use last purchase to determine churn
        customer_features = []

        for customer_id in orders['customer_unique_id'].unique():
            customer_orders = orders[orders['customer_unique_id'] == customer_id]

            if len(customer_orders) < 2:
                # Can't compute churn for single-purchase customers
                continue

            # Take all but last order for feature engineering
            train_orders = customer_orders.iloc[:-1]
            test_order = customer_orders.iloc[-1]

            # Features
            features = {}
            features['customer_unique_id'] = customer_id

            # Recency (days since last purchase before test order)
            features['recency_days'] = (
                test_order['order_purchase_timestamp'] - train_orders['order_purchase_timestamp'].max()
            ).days

            # Frequency
            features['frequency'] = len(train_orders)

            # Monetary
            features['monetary'] = train_orders['order_value'].sum()
            features['avg_order_value'] = train_orders['order_value'].mean()

            # Time between purchases
            if len(train_orders) > 1:
                time_diffs = train_orders['order_purchase_timestamp'].diff().dt.days.dropna()
                features['avg_days_between_orders'] = time_diffs.mean()
                features['std_days_between_orders'] = time_diffs.std() if len(time_diffs) > 1 else 0
            else:
                features['avg_days_between_orders'] = 0
                features['std_days_between_orders'] = 0

            # Target: Did customer return within prediction_window_days after last purchase?
            days_since_last = (
                orders['order_purchase_timestamp'].max() - test_order['order_purchase_timestamp']
            ).days

            # Churn = 1 if customer didn't return within window, 0 if they did
            features['churn'] = 1 if days_since_last > prediction_window_days else 0
            features['days_since_last'] = days_since_last

            customer_features.append(features)

        df = pd.DataFrame(customer_features)

        # Fill NaN
        df = df.fillna(0)

        return df

    def train(self, prediction_window_days: int = 90, test_size: float = 0.2):
        """
        Train churn prediction model.

        Args:
            prediction_window_days: Days to define churn window.
            test_size: Test set size.

        Returns:
            Dictionary with metrics.
        """
        print(f"Preparing churn dataset (window={prediction_window_days} days)...")
        df = self._prepare_churn_dataset(prediction_window_days)

        print(f"Dataset size: {len(df)} customers")
        print(f"Churn rate: {df['churn'].mean():.2%}")

        # Prepare features and target
        feature_cols = [
            'recency_days', 'frequency', 'monetary', 'avg_order_value',
            'avg_days_between_orders', 'std_days_between_orders'
        ]

        X = df[feature_cols]
        y = df['churn']

        self.feature_names = feature_cols

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"\nTraining churn model...")
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

        # Train model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )

        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Will Return', 'Will Churn']))

        auc = roc_auc_score(y_test, y_prob)
        print(f"\nROC-AUC Score: {auc:.4f}")

        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nFeature Importance:")
        print(importance)

        self.is_fitted = True

        return {
            'roc_auc': auc,
            'feature_importance': importance.to_dict('records')
        }

    def predict_churn_risk(self, customer_unique_id: str) -> Dict:
        """
        Predict churn risk for a customer.

        Args:
            customer_unique_id: Customer unique ID.

        Returns:
            Prediction dictionary with risk score and features.
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")

        # Get customer features
        from features import FeatureEngineer

        engineer = FeatureEngineer(self.loader)
        customer_features = engineer.get_customer_features(customer_unique_id)

        if not customer_features:
            return {'error': 'Customer not found'}

        # Prepare features
        X = pd.DataFrame([{
            'recency_days': customer_features.get('recency_days', 0),
            'frequency': customer_features.get('frequency', 0),
            'monetary': customer_features.get('monetary', 0),
            'avg_order_value': customer_features.get('avg_price', 0),
            'avg_days_between_orders': 0,  # Would need to compute
            'std_days_between_orders': 0
        }])

        # Predict
        churn_prob = self.model.predict_proba(X)[0, 1]
        churn_prediction = self.model.predict(X)[0]

        # Risk level
        if churn_prob < 0.3:
            risk_level = 'Low'
        elif churn_prob < 0.6:
            risk_level = 'Medium'
        else:
            risk_level = 'High'

        return {
            'customer_unique_id': customer_unique_id,
            'churn_probability': float(churn_prob),
            'will_churn': bool(churn_prediction),
            'risk_level': risk_level,
            'segment': customer_features.get('segment', 'Unknown')
        }

    def get_at_risk_customers(self, threshold: float = 0.6, top_n: int = 100) -> pd.DataFrame:
        """
        Get list of customers at high risk of churning.

        Args:
            threshold: Probability threshold for high risk.
            top_n: Number of customers to return.

        Returns:
            DataFrame of at-risk customers.
        """
        if not self.is_fitted:
            raise ValueError("Model not trained.")

        # This would require batch prediction on all customers
        # Simplified version here
        print(f"Finding top {top_n} customers at risk (threshold > {threshold})...")

        # Would implement full batch prediction
        return pd.DataFrame()


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from data_loader import OlistDataLoader

    loader = OlistDataLoader(data_dir='../datos')
    loader.load_all()

    predictor = ChurnPredictor(loader)
    metrics = predictor.train(prediction_window_days=90)

    print("\nModel trained successfully!")
