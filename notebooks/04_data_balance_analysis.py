"""
Comprehensive Data Balance and Distribution Analysis
Analyzes dataset characteristics, distributions, and balance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import OlistDataLoader
from src.analytics import OlistAnalytics

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class DataBalanceAnalyzer:
    """Analyze data distribution and balance."""

    def __init__(self, loader: OlistDataLoader):
        self.loader = loader

    def analyze_missing_values(self):
        """Analyze missing values in all datasets."""
        print("\n" + "="*80)
        print("MISSING VALUES ANALYSIS")
        print("="*80)

        datasets = {
            'Orders': self.loader.orders,
            'Order Items': self.loader.order_items,
            'Products': self.loader.products,
            'Customers': self.loader.customers,
            'Reviews': self.loader.reviews,
            'Payments': self.loader.payments,
        }

        for name, df in datasets.items():
            if df is not None:
                missing = df.isnull().sum()
                missing_pct = (missing / len(df) * 100).round(2)

                missing_df = pd.DataFrame({
                    'Missing Count': missing,
                    'Missing %': missing_pct
                })
                missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)

                if len(missing_df) > 0:
                    print(f"\n{name} Dataset - Missing Values:")
                    print(missing_df)
                else:
                    print(f"\n{name} Dataset: No missing values OK")

    def analyze_target_distribution(self):
        """Analyze distribution of target variable (repeat purchase)."""
        print("\n" + "="*80)
        print("TARGET VARIABLE DISTRIBUTION (Repeat Purchase)")
        print("="*80)

        # Calculate repeat purchases per customer
        customer_orders = self.loader.orders.groupby('customer_id').size()

        repeat_customers = (customer_orders > 1).sum()
        single_customers = (customer_orders == 1).sum()
        total_customers = len(customer_orders)

        print(f"\nTotal Customers: {total_customers:,}")
        print(f"Single Purchase Customers: {single_customers:,} ({single_customers/total_customers*100:.2f}%)")
        print(f"Repeat Purchase Customers: {repeat_customers:,} ({repeat_customers/total_customers*100:.2f}%)")

        print(f"\n[!] CLASS IMBALANCE DETECTED!")
        print(f"   Imbalance Ratio: {single_customers/repeat_customers:.1f}:1")
        print(f"   This is a HIGHLY IMBALANCED dataset")

        # Distribution of number of orders
        print("\n" + "-"*80)
        print("Distribution of Orders per Customer:")
        print(customer_orders.value_counts().head(10))

        return customer_orders

    def analyze_numerical_distributions(self):
        """Analyze distribution of numerical features."""
        print("\n" + "="*80)
        print("NUMERICAL FEATURES DISTRIBUTION")
        print("="*80)

        # Merge data for analysis
        data = self.loader.order_items.merge(
            self.loader.orders[['order_id', 'order_purchase_timestamp']],
            on='order_id',
            how='left'
        )

        # Price distribution
        print("\nPrice Distribution:")
        print(data['price'].describe())

        # Freight distribution
        print("\nFreight Value Distribution:")
        print(data['freight_value'].describe())

        # Detect outliers using IQR method
        self._detect_outliers(data, 'price')
        self._detect_outliers(data, 'freight_value')

    def _detect_outliers(self, df, column):
        """Detect outliers using IQR method."""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outlier_pct = len(outliers) / len(df) * 100

        print(f"\n{column} - Outlier Analysis:")
        print(f"  IQR Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
        print(f"  Outliers: {len(outliers):,} ({outlier_pct:.2f}%)")
        print(f"  Min outlier: {outliers[column].min():.2f}")
        print(f"  Max outlier: {outliers[column].max():.2f}")

    def analyze_categorical_balance(self):
        """Analyze distribution of categorical variables."""
        print("\n" + "="*80)
        print("CATEGORICAL FEATURES DISTRIBUTION")
        print("="*80)

        # Product categories
        if self.loader.products is not None:
            print("\nTop 10 Product Categories (by count):")
            cat_dist = self.loader.products['product_category_name'].value_counts().head(10)
            print(cat_dist)

            total_products = len(self.loader.products)
            top_cat_pct = cat_dist.iloc[0] / total_products * 100
            print(f"\nMost common category represents {top_cat_pct:.2f}% of products")

        # Order status
        if self.loader.orders is not None:
            print("\nOrder Status Distribution:")
            status_dist = self.loader.orders['order_status'].value_counts()
            print(status_dist)

            status_pct = (status_dist / len(self.loader.orders) * 100).round(2)
            print("\nStatus Percentages:")
            print(status_pct)

        # Review scores
        if self.loader.reviews is not None:
            print("\nReview Score Distribution:")
            review_dist = self.loader.reviews['review_score'].value_counts().sort_index()
            print(review_dist)

            review_pct = (review_dist / len(self.loader.reviews) * 100).round(2)
            print("\nReview Score Percentages:")
            print(review_pct)

    def analyze_temporal_patterns(self):
        """Analyze temporal patterns in the data."""
        print("\n" + "="*80)
        print("TEMPORAL PATTERNS ANALYSIS")
        print("="*80)

        # Convert timestamp
        self.loader.orders['order_purchase_timestamp'] = pd.to_datetime(
            self.loader.orders['order_purchase_timestamp']
        )

        # Date range
        min_date = self.loader.orders['order_purchase_timestamp'].min()
        max_date = self.loader.orders['order_purchase_timestamp'].max()

        print(f"\nData Time Range:")
        print(f"  Start: {min_date}")
        print(f"  End: {max_date}")
        print(f"  Duration: {(max_date - min_date).days} days")

        # Orders by month
        self.loader.orders['year_month'] = self.loader.orders['order_purchase_timestamp'].dt.to_period('M')
        monthly_orders = self.loader.orders.groupby('year_month').size()

        print(f"\nMonthly Order Statistics:")
        print(f"  Average: {monthly_orders.mean():.0f} orders/month")
        print(f"  Min: {monthly_orders.min()} orders ({monthly_orders.idxmin()})")
        print(f"  Max: {monthly_orders.max()} orders ({monthly_orders.idxmax()})")

        # Check for seasonality
        self.loader.orders['month'] = self.loader.orders['order_purchase_timestamp'].dt.month
        seasonal = self.loader.orders.groupby('month').size()

        print("\nOrders by Month (seasonality check):")
        print(seasonal.sort_values(ascending=False).head())

    def analyze_correlations(self):
        """Analyze correlations between numerical features."""
        print("\n" + "="*80)
        print("CORRELATION ANALYSIS")
        print("="*80)

        # Merge relevant data
        data = self.loader.order_items.merge(
            self.loader.orders[['order_id', 'order_purchase_timestamp']],
            on='order_id',
            how='left'
        )

        # Select numerical columns
        numerical_cols = ['price', 'freight_value']

        if len(numerical_cols) > 1:
            corr_matrix = data[numerical_cols].corr()
            print("\nCorrelation Matrix:")
            print(corr_matrix)

            # Find strong correlations
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.5:
                        strong_corr.append({
                            'Feature 1': corr_matrix.columns[i],
                            'Feature 2': corr_matrix.columns[j],
                            'Correlation': corr_matrix.iloc[i, j]
                        })

            if strong_corr:
                print("\nStrong Correlations (|r| > 0.5):")
                print(pd.DataFrame(strong_corr))
            else:
                print("\nNo strong correlations found (|r| > 0.5)")

    def create_visualizations(self, output_dir='outputs/balance_analysis'):
        """Create visualizations for data analysis."""
        print("\n" + "="*80)
        print("CREATING VISUALIZATIONS")
        print("="*80)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. Target distribution (repeat purchase)
        customer_orders = self.loader.orders.groupby('customer_id').size()

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Pie chart
        repeat_vs_single = pd.Series({
            'Single Purchase': (customer_orders == 1).sum(),
            'Repeat Purchase': (customer_orders > 1).sum()
        })
        axes[0].pie(repeat_vs_single, labels=repeat_vs_single.index, autopct='%1.1f%%', startangle=90)
        axes[0].set_title('Customer Segmentation: Single vs Repeat', fontsize=14, fontweight='bold')

        # Bar chart of orders per customer
        order_counts = customer_orders.value_counts().head(10).sort_index()
        axes[1].bar(order_counts.index, order_counts.values, color='steelblue')
        axes[1].set_xlabel('Number of Orders', fontsize=12)
        axes[1].set_ylabel('Number of Customers', fontsize=12)
        axes[1].set_title('Distribution of Orders per Customer', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'target_distribution.png', dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_path / 'target_distribution.png'}")
        plt.close()

        # 2. Price distribution with outliers
        data = self.loader.order_items

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Histogram
        axes[0].hist(data['price'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Price', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Price Distribution', fontsize=14, fontweight='bold')
        axes[0].axvline(data['price'].median(), color='red', linestyle='--', label=f'Median: ${data["price"].median():.2f}')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        # Box plot
        axes[1].boxplot(data['price'], vert=True)
        axes[1].set_ylabel('Price', fontsize=12)
        axes[1].set_title('Price Distribution - Box Plot (Outliers)', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'price_distribution.png', dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_path / 'price_distribution.png'}")
        plt.close()

        # 3. Review score distribution
        if self.loader.reviews is not None:
            review_dist = self.loader.reviews['review_score'].value_counts().sort_index()

            plt.figure(figsize=(10, 6))
            colors = ['#d32f2f', '#f57c00', '#fbc02d', '#afb42b', '#388e3c']
            plt.bar(review_dist.index, review_dist.values, color=colors, edgecolor='black')
            plt.xlabel('Review Score', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.title('Review Score Distribution', fontsize=14, fontweight='bold')
            plt.xticks(review_dist.index)
            plt.grid(axis='y', alpha=0.3)

            # Add percentages on top of bars
            for i, v in enumerate(review_dist.values):
                pct = v / review_dist.sum() * 100
                plt.text(review_dist.index[i], v, f'{pct:.1f}%', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(output_path / 'review_score_balance.png', dpi=300, bbox_inches='tight')
            print(f"[OK] Saved: {output_path / 'review_score_balance.png'}")
            plt.close()

        # 4. Top categories balance
        if self.loader.products is not None:
            top_categories = self.loader.products['product_category_name'].value_counts().head(15)

            plt.figure(figsize=(12, 8))
            plt.barh(range(len(top_categories)), top_categories.values, color='steelblue')
            plt.yticks(range(len(top_categories)), top_categories.index)
            plt.xlabel('Number of Products', fontsize=12)
            plt.title('Top 15 Product Categories - Balance Analysis', fontsize=14, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)

            # Add counts on bars
            for i, v in enumerate(top_categories.values):
                plt.text(v, i, f' {v}', va='center')

            plt.tight_layout()
            plt.savefig(output_path / 'category_balance.png', dpi=300, bbox_inches='tight')
            print(f"[OK] Saved: {output_path / 'category_balance.png'}")
            plt.close()

        print(f"\n[OK] All visualizations saved to: {output_path}")

    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        print("\n" + "="*80)
        print("DATA BALANCE & QUALITY SUMMARY REPORT")
        print("="*80)

        # Dataset sizes
        print("\nDATASET SIZES:")
        print(f"  Orders: {len(self.loader.orders):,} rows")
        print(f"  Order Items: {len(self.loader.order_items):,} rows")
        print(f"  Products: {len(self.loader.products):,} rows")
        print(f"  Customers: {len(self.loader.customers):,} rows")

        # Class balance
        customer_orders = self.loader.orders.groupby('customer_id').size()
        repeat_rate = (customer_orders > 1).sum() / len(customer_orders) * 100

        print("\nCLASS BALANCE:")
        print(f"  Repeat Purchase Rate: {repeat_rate:.2f}%")
        print(f"  [!] HIGHLY IMBALANCED - May need:")
        print(f"     - SMOTE or oversampling")
        print(f"     - Class weights in models")
        print(f"     - Stratified sampling")

        # Data quality
        total_missing = sum([
            self.loader.orders.isnull().sum().sum(),
            self.loader.order_items.isnull().sum().sum(),
            self.loader.products.isnull().sum().sum(),
        ])

        print("\nDATA QUALITY:")
        print(f"  Total Missing Values: {total_missing}")
        print(f"  Status: {'Good' if total_missing < 1000 else 'Needs attention'}")

        print("\nRECOMMENDATIONS:")
        if repeat_rate < 10:
            print("  1. Use class weights or SMOTE for imbalanced target")
        print("  2. Consider stratified train/test split")
        print("  3. Use precision-recall curves instead of ROC for evaluation")
        print("  4. Monitor minority class performance closely")
        print("  5. Consider ensemble methods for better performance")

        print("\n" + "="*80)


def main():
    """Run complete data balance analysis."""
    print("Starting Comprehensive Data Balance Analysis...")
    print("="*80)

    # Load data
    loader = OlistDataLoader(data_dir='datos')
    loader.load_all()

    # Create analyzer
    analyzer = DataBalanceAnalyzer(loader)

    # Run all analyses
    analyzer.analyze_missing_values()
    analyzer.analyze_target_distribution()
    analyzer.analyze_numerical_distributions()
    analyzer.analyze_categorical_balance()
    analyzer.analyze_temporal_patterns()
    analyzer.analyze_correlations()

    # Create visualizations
    analyzer.create_visualizations()

    # Generate summary
    analyzer.generate_summary_report()

    print("\n[OK] Analysis complete!")


if __name__ == '__main__':
    main()
