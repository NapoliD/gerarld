"""
Visualization module for analytics insights.
Creates publication-quality charts for reports.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional


class OlistVisualizer:
    """Create visualizations for Olist analytics."""

    def __init__(self, output_dir: str = './outputs'):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save plots.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10

    def plot_review_distribution(self, reviews_df: pd.DataFrame, save: bool = True):
        """Plot distribution of review scores."""
        fig, ax = plt.subplots(figsize=(10, 6))

        review_counts = reviews_df['review_score'].value_counts().sort_index()

        # Bar chart
        bars = ax.bar(review_counts.index, review_counts.values,
                      color=sns.color_palette('RdYlGn', len(review_counts)),
                      edgecolor='black', linewidth=1.2)

        # Add percentage labels
        total = review_counts.sum()
        for bar, count in zip(bars, review_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count:,}\n({count/total*100:.1f}%)',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xlabel('Review Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Reviews', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Review Scores', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks([1, 2, 3, 4, 5])

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'review_distribution.png', dpi=300, bbox_inches='tight')
            print(f"Saved: {self.output_dir / 'review_distribution.png'}")

        return fig

    def plot_top_categories(self, category_metrics: pd.DataFrame, top_n: int = 10, save: bool = True):
        """Plot top categories by GMV."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Get top N
        top_cats = category_metrics.head(top_n).copy()

        # Determine column name for category
        if 'product_category_name_english' in top_cats.columns:
            cat_col = 'product_category_name_english'
        else:
            cat_col = 'product_category_name'

        # Create horizontal bar chart
        y_pos = np.arange(len(top_cats))
        bars = ax.barh(y_pos, top_cats['total_gmv'],
                       color=sns.color_palette('viridis', len(top_cats)),
                       edgecolor='black', linewidth=1.2)

        # Add value labels
        for i, (bar, val, orders) in enumerate(zip(bars, top_cats['total_gmv'], top_cats['total_orders'])):
            ax.text(val, bar.get_y() + bar.get_height()/2,
                   f' ${val:,.0f} ({orders:,} orders)',
                   va='center', fontsize=9, fontweight='bold')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_cats[cat_col].str.replace('_', ' ').str.title())
        ax.set_xlabel('Total GMV ($)', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Product Categories by GMV', fontsize=14, fontweight='bold', pad=20)
        ax.invert_yaxis()

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'top_categories_gmv.png', dpi=300, bbox_inches='tight')
            print(f"Saved: {self.output_dir / 'top_categories_gmv.png'}")

        return fig

    def plot_orders_over_time(self, orders_df: pd.DataFrame, save: bool = True):
        """Plot orders over time."""
        fig, ax = plt.subplots(figsize=(14, 6))

        # Convert to datetime
        orders = orders_df.copy()
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])

        # Aggregate by month
        orders['year_month'] = orders['order_purchase_timestamp'].dt.to_period('M')
        monthly_orders = orders.groupby('year_month').size()

        # Plot
        x = range(len(monthly_orders))
        ax.plot(x, monthly_orders.values, marker='o', linewidth=2, markersize=6,
               color='#2E86AB', label='Monthly Orders')

        # Add trend line
        z = np.polyfit(x, monthly_orders.values, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), "--", color='red', alpha=0.8, linewidth=2, label='Trend')

        ax.set_xticks(x[::2])  # Show every other month
        ax.set_xticklabels([str(m) for m in monthly_orders.index[::2]], rotation=45, ha='right')
        ax.set_xlabel('Month', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Orders', fontsize=12, fontweight='bold')
        ax.set_title('Orders Over Time (Monthly)', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'orders_over_time.png', dpi=300, bbox_inches='tight')
            print(f"Saved: {self.output_dir / 'orders_over_time.png'}")

        return fig

    def plot_retention_cohort(self, orders_df: pd.DataFrame, customers_df: pd.DataFrame, save: bool = True):
        """Plot retention cohort analysis."""
        # Merge orders with customers
        orders = orders_df.copy()
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])

        orders = orders.merge(
            customers_df[['customer_id', 'customer_unique_id']],
            on='customer_id',
            how='left'
        )

        # Get first order date for each customer
        customer_first_order = (
            orders.groupby('customer_unique_id')['order_purchase_timestamp']
            .min()
            .reset_index()
            .rename(columns={'order_purchase_timestamp': 'first_order_date'})
        )

        orders = orders.merge(customer_first_order, on='customer_unique_id')
        orders['cohort'] = orders['first_order_date'].dt.to_period('M')
        orders['order_period'] = orders['order_purchase_timestamp'].dt.to_period('M')

        # Calculate periods since first order
        orders['period_number'] = (
            (orders['order_period'] - orders['cohort']).apply(lambda x: x.n)
        )

        # Create cohort table
        cohort_data = (
            orders.groupby(['cohort', 'period_number'])['customer_unique_id']
            .nunique()
            .reset_index()
        )

        cohort_pivot = cohort_data.pivot(
            index='cohort',
            columns='period_number',
            values='customer_unique_id'
        )

        # Calculate retention percentages
        cohort_size = cohort_pivot.iloc[:, 0]
        retention = cohort_pivot.divide(cohort_size, axis=0) * 100

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(14, 10))

        sns.heatmap(retention.iloc[:, :12],  # First 12 months
                   annot=True, fmt='.0f', cmap='RdYlGn',
                   cbar_kws={'label': 'Retention %'},
                   ax=ax, vmin=0, vmax=100)

        ax.set_title('Customer Retention Cohort Analysis (First 12 Months)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Months Since First Purchase', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cohort (First Purchase Month)', fontsize=12, fontweight='bold')

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'retention_cohort.png', dpi=300, bbox_inches='tight')
            print(f"Saved: {self.output_dir / 'retention_cohort.png'}")

        return fig

    def plot_price_distribution(self, order_items_df: pd.DataFrame, save: bool = True):
        """Plot distribution of product prices."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        prices = order_items_df['price'].dropna()

        # Histogram
        ax1.hist(prices, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Price ($)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title('Price Distribution', fontsize=12, fontweight='bold')
        ax1.axvline(prices.median(), color='red', linestyle='--', linewidth=2, label=f'Median: ${prices.median():.2f}')
        ax1.legend()

        # Box plot (log scale for better visualization)
        ax2.boxplot(prices, vert=False)
        ax2.set_xlabel('Price ($)', fontsize=11, fontweight='bold')
        ax2.set_title('Price Distribution (Box Plot)', fontsize=12, fontweight='bold')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'price_distribution.png', dpi=300, bbox_inches='tight')
            print(f"Saved: {self.output_dir / 'price_distribution.png'}")

        return fig

    def plot_customer_segments(self, segment_data: pd.DataFrame, save: bool = True):
        """Plot customer segmentation."""
        fig, ax = plt.subplots(figsize=(10, 6))

        segments = segment_data['segment'].value_counts()

        # Pie chart with custom colors
        colors = sns.color_palette('Set2', len(segments))
        wedges, texts, autotexts = ax.pie(
            segments.values,
            labels=segments.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            explode=[0.05] * len(segments),
            textprops={'fontsize': 11, 'fontweight': 'bold'}
        )

        # Add count labels
        for i, (text, count) in enumerate(zip(texts, segments.values)):
            text.set_text(f'{text.get_text()}\n({count:,} customers)')

        ax.set_title('Customer Segmentation', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'customer_segments.png', dpi=300, bbox_inches='tight')
            print(f"Saved: {self.output_dir / 'customer_segments.png'}")

        return fig

    def create_all_visualizations(self, loader, analytics):
        """Create all standard visualizations."""
        print("Creating all visualizations...")

        # 1. Review distribution
        if loader.reviews is not None:
            self.plot_review_distribution(loader.reviews)

        # 2. Top categories
        metrics = analytics.compute_all_metrics()
        self.plot_top_categories(metrics['top_categories'])

        # 3. Orders over time
        self.plot_orders_over_time(loader.orders)

        # 4. Retention cohort
        self.plot_retention_cohort(loader.orders, loader.customers)

        # 5. Price distribution
        self.plot_price_distribution(loader.order_items)

        print(f"\nAll visualizations saved to: {self.output_dir}")


if __name__ == "__main__":
    # Test visualizations
    import sys
    sys.path.append('..')
    from data_loader import OlistDataLoader
    from analytics import OlistAnalytics

    loader = OlistDataLoader(data_dir='../datos')
    loader.load_all()

    analytics = OlistAnalytics(loader)

    visualizer = OlistVisualizer(output_dir='../outputs')
    visualizer.create_all_visualizations(loader, analytics)
