"""
LangChain Tools for Data Analysis Chatbot
Custom tools for analyzing Olist data and generating insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, Any, List
import json

from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field

from .data_loader import OlistDataLoader
from .analytics import OlistAnalytics
from .features import FeatureEngineer


class DataAnalysisTool(BaseTool):
    """Tool for analyzing data distributions and statistics."""

    name: str = "analyze_data_distribution"
    description: str = """
    Analyzes the distribution of data in the Olist dataset.
    Use this to get statistics about:
    - Customer behavior (orders, repeat purchases)
    - Product categories
    - Prices and revenue
    - Reviews and ratings
    - Temporal patterns

    Input should be one of: 'customers', 'products', 'orders', 'reviews', 'revenue', 'temporal'
    Returns detailed statistics and insights.
    """

    loader: OlistDataLoader = Field(default=None, exclude=True)

    def __init__(self, loader: OlistDataLoader):
        super().__init__(loader=loader)

    def _run(self, analysis_type: str) -> str:
        """Execute the analysis."""
        try:
            if analysis_type.lower() == 'customers':
                return self._analyze_customers()
            elif analysis_type.lower() == 'products':
                return self._analyze_products()
            elif analysis_type.lower() == 'orders':
                return self._analyze_orders()
            elif analysis_type.lower() == 'reviews':
                return self._analyze_reviews()
            elif analysis_type.lower() == 'revenue':
                return self._analyze_revenue()
            elif analysis_type.lower() == 'temporal':
                return self._analyze_temporal()
            else:
                return f"Unknown analysis type: {analysis_type}. Use: customers, products, orders, reviews, revenue, or temporal"
        except Exception as e:
            return f"Error during analysis: {str(e)}"

    async def _arun(self, analysis_type: str) -> str:
        """Async version."""
        return self._run(analysis_type)

    def _analyze_customers(self) -> str:
        """Analyze customer behavior."""
        orders = self.loader.orders
        customers = self.loader.customers

        # Merge to get customer_unique_id
        orders_with_customers = orders.merge(
            customers[['customer_id', 'customer_unique_id']],
            on='customer_id'
        )

        # Calculate metrics
        total_customers = customers['customer_unique_id'].nunique()
        orders_per_customer = orders_with_customers.groupby('customer_unique_id').size()

        repeat_customers = (orders_per_customer > 1).sum()
        single_customers = (orders_per_customer == 1).sum()
        repeat_rate = (repeat_customers / total_customers) * 100

        avg_orders = orders_per_customer.mean()
        max_orders = orders_per_customer.max()

        # Customer states
        states = customers['customer_state'].value_counts().head(5)

        result = f"""
CUSTOMER ANALYSIS:
==================
Total Customers: {total_customers:,}
Single Purchase Customers: {single_customers:,} ({single_customers/total_customers*100:.2f}%)
Repeat Purchase Customers: {repeat_customers:,} ({repeat_rate:.2f}%)

Order Statistics:
- Average orders per customer: {avg_orders:.2f}
- Maximum orders by single customer: {max_orders}

⚠️  CRITICAL ISSUE: {100-repeat_rate:.2f}% of customers never return!

Top 5 Customer States:
{states.to_string()}

RECOMMENDATIONS:
1. Focus on retention - current churn rate is extreme
2. Implement loyalty program for second purchase
3. Send re-engagement emails at 30/60/90 days
4. Analyze why satisfied customers don't return
"""
        return result

    def _analyze_products(self) -> str:
        """Analyze product distribution."""
        products = self.loader.products
        order_items = self.loader.order_items

        # Category distribution
        total_products = len(products)
        categories = products['product_category_name'].value_counts()
        top_categories = categories.head(10)

        # Product sales
        product_sales = order_items.groupby('product_id').size()
        avg_sales = product_sales.mean()

        # Price analysis
        price_stats = order_items['price'].describe()

        result = f"""
PRODUCT ANALYSIS:
=================
Total Products: {total_products:,}
Total Categories: {categories.nunique()}

Top 10 Categories by Product Count:
{top_categories.to_string()}

Sales Distribution:
- Average sales per product: {avg_sales:.2f}
- Products with 0 sales: {(total_products - len(product_sales))}

Price Statistics:
{price_stats.to_string()}

INSIGHTS:
- Category concentration: Top category has {categories.iloc[0]/total_products*100:.1f}% of products
- Long tail: Many products with few/no sales
- Price range: ${price_stats['min']:.2f} to ${price_stats['max']:.2f}

RECOMMENDATIONS:
1. Focus inventory on top-selling categories
2. Implement dynamic pricing for slow movers
3. Bundle complementary products
"""
        return result

    def _analyze_orders(self) -> str:
        """Analyze order patterns."""
        orders = self.loader.orders

        # Order status
        status = orders['order_status'].value_counts()
        total_orders = len(orders)

        # Delivery performance
        delivered = (orders['order_status'] == 'delivered').sum()
        delivery_rate = (delivered / total_orders) * 100

        result = f"""
ORDER ANALYSIS:
===============
Total Orders: {total_orders:,}

Order Status Distribution:
{status.to_string()}

Delivery Performance:
- Successful deliveries: {delivered:,} ({delivery_rate:.2f}%)
- Failed/Cancelled: {total_orders - delivered:,}

INSIGHTS:
- High delivery success rate: {delivery_rate:.2f}%
- Operational excellence achieved
- Problem is NOT fulfillment - it's retention!

RECOMMENDATIONS:
1. Maintain delivery standards
2. Use delivery success as marketing point
3. Focus efforts on post-purchase engagement
"""
        return result

    def _analyze_reviews(self) -> str:
        """Analyze review patterns."""
        reviews = self.loader.reviews

        if reviews is None:
            return "Review data not available"

        # Review scores
        score_dist = reviews['review_score'].value_counts().sort_index()
        avg_score = reviews['review_score'].mean()

        # Positive vs negative
        positive = (reviews['review_score'] >= 4).sum()
        negative = (reviews['review_score'] <= 2).sum()
        neutral = (reviews['review_score'] == 3).sum()

        total = len(reviews)

        result = f"""
REVIEW ANALYSIS:
================
Total Reviews: {total:,}
Average Score: {avg_score:.2f} / 5.0

Score Distribution:
{score_dist.to_string()}

Sentiment Breakdown:
- Positive (4-5 stars): {positive:,} ({positive/total*100:.2f}%)
- Neutral (3 stars): {neutral:,} ({neutral/total*100:.2f}%)
- Negative (1-2 stars): {negative:,} ({negative/total*100:.2f}%)

⚠️  THE PARADOX:
- {positive/total*100:.2f}% positive reviews
- But 96.88% customers never return
- Satisfaction does NOT equal loyalty!

RECOMMENDATIONS:
1. High satisfaction is foundation - maintain it
2. Convert satisfaction into loyalty (programs, incentives)
3. Survey satisfied one-time customers: "Why didn't you return?"
4. A/B test retention campaigns on satisfied customers
"""
        return result

    def _analyze_revenue(self) -> str:
        """Analyze revenue patterns."""
        order_items = self.loader.order_items
        products = self.loader.products

        # Total revenue
        total_revenue = order_items['price'].sum()
        total_freight = order_items['freight_value'].sum()
        gmv = total_revenue + total_freight

        # Revenue by category
        items_with_category = order_items.merge(
            products[['product_id', 'product_category_name']],
            on='product_id',
            how='left'
        )

        revenue_by_category = items_with_category.groupby('product_category_name')['price'].sum().sort_values(ascending=False).head(10)

        # AOV
        orders = self.loader.orders
        aov = total_revenue / len(orders)

        result = f"""
REVENUE ANALYSIS:
=================
Total Revenue: ${total_revenue:,.2f}
Total Freight: ${total_freight:,.2f}
Gross Merchandise Value (GMV): ${gmv:,.2f}

Average Order Value (AOV): ${aov:.2f}

Top 10 Categories by Revenue:
{revenue_by_category.apply(lambda x: f"${x:,.2f}").to_string()}

INSIGHTS:
- Top category generates ${revenue_by_category.iloc[0]:,.2f}
- Freight represents {total_freight/gmv*100:.1f}% of GMV
- AOV of ${aov:.2f} provides baseline for discount strategies

RECOMMENDATIONS:
1. Increase AOV through bundling (target: +15%)
2. Cross-sell from top revenue categories
3. Premium shipping as upsell opportunity
4. Calculate LTV:CAC ratio urgently
"""
        return result

    def _analyze_temporal(self) -> str:
        """Analyze temporal patterns."""
        orders = self.loader.orders.copy()
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])

        # Date range
        min_date = orders['order_purchase_timestamp'].min()
        max_date = orders['order_purchase_timestamp'].max()
        duration_days = (max_date - min_date).days

        # Orders by month
        orders['year_month'] = orders['order_purchase_timestamp'].dt.to_period('M')
        monthly = orders.groupby('year_month').size()

        # Day of week
        orders['day_of_week'] = orders['order_purchase_timestamp'].dt.day_name()
        dow = orders['day_of_week'].value_counts()

        # Month
        orders['month'] = orders['order_purchase_timestamp'].dt.month_name()
        monthly_pattern = orders['month'].value_counts().head(5)

        result = f"""
TEMPORAL ANALYSIS:
==================
Data Range: {min_date.date()} to {max_date.date()}
Duration: {duration_days} days ({duration_days/30:.1f} months)

Monthly Statistics:
- Average: {monthly.mean():.0f} orders/month
- Peak month: {monthly.max()} orders
- Lowest month: {monthly.min()} orders

Top 5 Months (by order count):
{monthly_pattern.to_string()}

Top Day of Week:
{dow.head(3).to_string()}

INSIGHTS:
- Clear seasonality detected
- Peak ordering in mid-year (May-Aug)
- Opportunity for seasonal campaigns

RECOMMENDATIONS:
1. Prepare inventory for May-Aug surge
2. Run retention campaigns in off-peak months
3. Weekend vs weekday campaign timing
4. Time re-engagement emails for peak days
"""
        return result


class VisualizationTool(BaseTool):
    """Tool for generating visualizations."""

    name: str = "generate_visualization"
    description: str = """
    Generates visualizations of data distributions.
    Input should be one of:
    - 'customer_distribution': Pie chart of repeat vs single purchase
    - 'category_revenue': Bar chart of top categories by revenue
    - 'review_scores': Distribution of review scores
    - 'price_distribution': Histogram of product prices
    - 'temporal_trend': Line chart of orders over time

    Returns the path to the generated image file.
    """

    loader: OlistDataLoader = Field(default=None, exclude=True)
    output_dir: str = Field(default="outputs/chatbot_viz")

    def __init__(self, loader: OlistDataLoader, output_dir: str = "outputs/chatbot_viz"):
        super().__init__(loader=loader, output_dir=output_dir)

    def _run(self, viz_type: str) -> str:
        """Generate visualization."""
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            if 'customer' in viz_type.lower():
                return self._viz_customer_distribution(output_path)
            elif 'category' in viz_type.lower() or 'revenue' in viz_type.lower():
                return self._viz_category_revenue(output_path)
            elif 'review' in viz_type.lower():
                return self._viz_review_scores(output_path)
            elif 'price' in viz_type.lower():
                return self._viz_price_distribution(output_path)
            elif 'temporal' in viz_type.lower() or 'trend' in viz_type.lower():
                return self._viz_temporal_trend(output_path)
            else:
                return f"Unknown visualization type: {viz_type}"
        except Exception as e:
            return f"Error generating visualization: {str(e)}"

    async def _arun(self, viz_type: str) -> str:
        """Async version."""
        return self._run(viz_type)

    def _viz_customer_distribution(self, output_path: Path) -> str:
        """Visualize customer distribution."""
        orders = self.loader.orders
        customers = self.loader.customers

        orders_with_customers = orders.merge(
            customers[['customer_id', 'customer_unique_id']],
            on='customer_id'
        )

        orders_per_customer = orders_with_customers.groupby('customer_unique_id').size()

        single = (orders_per_customer == 1).sum()
        repeat = (orders_per_customer > 1).sum()

        plt.figure(figsize=(10, 6))
        plt.pie([single, repeat], labels=['Single Purchase', 'Repeat Purchase'],
                autopct='%1.1f%%', startangle=90, colors=['#ff6b6b', '#4ecdc4'])
        plt.title('Customer Distribution: Single vs Repeat Purchases', fontsize=14, fontweight='bold')

        filename = output_path / 'customer_distribution.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        return f"Visualization saved to: {filename}\n\nKey Insight: {single/(single+repeat)*100:.1f}% of customers make only one purchase!"

    def _viz_category_revenue(self, output_path: Path) -> str:
        """Visualize revenue by category."""
        order_items = self.loader.order_items
        products = self.loader.products

        items_with_category = order_items.merge(
            products[['product_id', 'product_category_name']],
            on='product_id',
            how='left'
        )

        revenue_by_category = items_with_category.groupby('product_category_name')['price'].sum().sort_values(ascending=False).head(15)

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(revenue_by_category)), revenue_by_category.values, color='steelblue')
        plt.yticks(range(len(revenue_by_category)), revenue_by_category.index)
        plt.xlabel('Revenue ($)', fontsize=12)
        plt.title('Top 15 Categories by Revenue', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)

        for i, v in enumerate(revenue_by_category.values):
            plt.text(v, i, f' ${v:,.0f}', va='center')

        filename = output_path / 'category_revenue.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        return f"Visualization saved to: {filename}\n\nTop category: {revenue_by_category.index[0]} with ${revenue_by_category.iloc[0]:,.2f}"

    def _viz_review_scores(self, output_path: Path) -> str:
        """Visualize review score distribution."""
        reviews = self.loader.reviews

        if reviews is None:
            return "Review data not available"

        score_dist = reviews['review_score'].value_counts().sort_index()

        plt.figure(figsize=(10, 6))
        colors = ['#d32f2f', '#f57c00', '#fbc02d', '#afb42b', '#388e3c']
        plt.bar(score_dist.index, score_dist.values, color=colors, edgecolor='black')
        plt.xlabel('Review Score', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Review Score Distribution', fontsize=14, fontweight='bold')
        plt.xticks(score_dist.index)
        plt.grid(axis='y', alpha=0.3)

        for i, v in enumerate(score_dist.values):
            pct = v / score_dist.sum() * 100
            plt.text(score_dist.index[i], v, f'{pct:.1f}%', ha='center', va='bottom')

        filename = output_path / 'review_scores.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        positive = (reviews['review_score'] >= 4).sum() / len(reviews) * 100
        return f"Visualization saved to: {filename}\n\nInsight: {positive:.1f}% positive reviews (4-5 stars)"

    def _viz_price_distribution(self, output_path: Path) -> str:
        """Visualize price distribution."""
        order_items = self.loader.order_items

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Histogram
        axes[0].hist(order_items['price'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Price ($)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Price Distribution', fontsize=14, fontweight='bold')
        axes[0].axvline(order_items['price'].median(), color='red', linestyle='--',
                       label=f'Median: ${order_items["price"].median():.2f}')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        # Box plot
        axes[1].boxplot(order_items['price'], vert=True)
        axes[1].set_ylabel('Price ($)', fontsize=12)
        axes[1].set_title('Price Box Plot (Outliers Shown)', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)

        filename = output_path / 'price_distribution.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        median = order_items['price'].median()
        mean = order_items['price'].mean()
        return f"Visualization saved to: {filename}\n\nPrice stats - Median: ${median:.2f}, Mean: ${mean:.2f}"

    def _viz_temporal_trend(self, output_path: Path) -> str:
        """Visualize temporal trends."""
        orders = self.loader.orders.copy()
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
        orders['year_month'] = orders['order_purchase_timestamp'].dt.to_period('M')

        monthly = orders.groupby('year_month').size()

        plt.figure(figsize=(14, 6))
        plt.plot(range(len(monthly)), monthly.values, marker='o', linewidth=2, markersize=6)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Number of Orders', fontsize=12)
        plt.title('Orders Over Time (Monthly)', fontsize=14, fontweight='bold')
        plt.xticks(range(0, len(monthly), 3), [str(m) for m in monthly.index[::3]], rotation=45)
        plt.grid(alpha=0.3)

        # Add trend line
        z = np.polyfit(range(len(monthly)), monthly.values, 1)
        p = np.poly1d(z)
        plt.plot(range(len(monthly)), p(range(len(monthly))), "r--", alpha=0.8, label='Trend')
        plt.legend()

        filename = output_path / 'temporal_trend.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        trend = "increasing" if z[0] > 0 else "decreasing"
        return f"Visualization saved to: {filename}\n\nTrend: Orders are {trend} over time"


class OptimizationTool(BaseTool):
    """Tool for suggesting optimizations."""

    name: str = "suggest_optimizations"
    description: str = """
    Suggests data-driven optimizations based on analysis.
    Input should be one of:
    - 'retention': Suggestions for improving customer retention
    - 'revenue': Suggestions for increasing revenue
    - 'operations': Suggestions for operational improvements
    - 'marketing': Suggestions for marketing strategies
    - 'product': Suggestions for product/inventory optimization

    Returns actionable recommendations.
    """

    loader: OlistDataLoader = Field(default=None, exclude=True)

    def __init__(self, loader: OlistDataLoader):
        super().__init__(loader=loader)

    def _run(self, optimization_area: str) -> str:
        """Generate optimization suggestions."""
        area = optimization_area.lower()

        if 'retention' in area:
            return self._optimize_retention()
        elif 'revenue' in area:
            return self._optimize_revenue()
        elif 'operation' in area:
            return self._optimize_operations()
        elif 'marketing' in area:
            return self._optimize_marketing()
        elif 'product' in area:
            return self._optimize_product()
        else:
            return f"Unknown optimization area: {optimization_area}"

    async def _arun(self, optimization_area: str) -> str:
        """Async version."""
        return self._run(optimization_area)

    def _optimize_retention(self) -> str:
        """Retention optimization suggestions."""
        return """
RETENTION OPTIMIZATION RECOMMENDATIONS:
=======================================

IMMEDIATE ACTIONS (Week 1-2):
1. Email Campaign Setup
   - Day 7: Thank you + product recommendations
   - Day 30: "Miss you" with 10% discount
   - Day 60: Last chance offer with free shipping
   - Day 90: Survey - "Why didn't you return?"

2. Quick Wins
   - Second purchase discount (15-20% off, 7-day expiration)
   - Free shipping on second order
   - Birthday month special offer

SHORT-TERM (Month 1-3):
3. Loyalty Program
   - Points per purchase (100 points = $1)
   - Tier system (Bronze/Silver/Gold)
   - Exclusive early access to sales
   - Member-only products

4. Personalization
   - Recommendations based on first purchase
   - Category-specific offers
   - Time-based triggers (replenishment reminders)

5. Subscription Pilot
   - Start with Health & Beauty (consumables)
   - Subscribe & Save (5-10% discount)
   - Flexible delivery schedules

MEDIUM-TERM (Month 3-6):
6. Brand Building
   - Make "Olist" visible in packaging
   - Unboxing experience improvement
   - Social media presence
   - Content marketing (buying guides)

7. Community Building
   - Product reviews with rewards
   - User-generated content campaigns
   - Referral program (give $10, get $10)

MEASUREMENT:
- Track repeat purchase rate weekly
- Cohort retention curves
- Email campaign performance
- LTV:CAC by channel

TARGET: Increase repeat rate from 3.12% to 10% in 6 months
"""

    def _optimize_revenue(self) -> str:
        """Revenue optimization suggestions."""
        order_items = self.loader.order_items
        aov = order_items['price'].mean()

        return f"""
REVENUE OPTIMIZATION RECOMMENDATIONS:
=====================================

Current AOV: ${aov:.2f}
Target AOV: ${aov*1.15:.2f} (+15%)

TACTICS:

1. Increase Average Order Value
   - Free shipping threshold at ${aov*1.5:.2f}
   - "Frequently bought together" bundles
   - Volume discounts (Buy 2, Get 10% off)
   - Gift with purchase over ${aov*2:.2f}

2. Cross-Selling
   - Complementary product recommendations
   - "Complete the look/set" suggestions
   - Category cross-sells (e.g., bed → bedding)

3. Up-Selling
   - Premium versions of viewed products
   - Extended warranties/services
   - Expedited shipping options
   - Gift wrapping services

4. Dynamic Pricing
   - Time-based promotions (flash sales)
   - Inventory clearance on slow movers
   - Premium pricing for trending items
   - Seasonal pricing adjustments

5. Subscription Revenue
   - Monthly subscription boxes
   - Auto-replenishment (consumables)
   - VIP membership ($X/month for benefits)

6. Marketplace Fees
   - Optimize commission structure
   - Premium placement fees for sellers
   - Advertising opportunities for brands

PROJECTED IMPACT:
- AOV increase: +15% = +$13.6M * 0.15 = $2.04M/year
- Subscription revenue: Estimate 2% adoption = $500K/year
- Cross-sell lift: +10% on 20% of orders = $272K/year

TOTAL POTENTIAL: $2.8M additional annual revenue
"""

    def _optimize_operations(self) -> str:
        """Operations optimization suggestions."""
        orders = self.loader.orders
        delivered = (orders['order_status'] == 'delivered').sum()
        total = len(orders)
        delivery_rate = delivered / total * 100

        return f"""
OPERATIONS OPTIMIZATION RECOMMENDATIONS:
========================================

Current Delivery Rate: {delivery_rate:.2f}%

MAINTAIN STRENGTHS:
1. Delivery Excellence
   - Current performance is excellent
   - Document best practices
   - Share success factors with new sellers
   - Use as competitive advantage in marketing

OPTIMIZATION AREAS:

2. Inventory Management
   - ABC analysis (focus on top 20% products)
   - Demand forecasting for seasonal items
   - Safety stock optimization
   - Drop-shipping for long-tail items

3. Fulfillment Speed
   - Same-day dispatch for top products
   - Regional warehouse strategy
   - Seller performance tiers
   - Expedited handling SLA

4. Cost Reduction
   - Bulk shipping negotiations
   - Packaging optimization (reduce weight/size)
   - Route optimization for local deliveries
   - Reverse logistics efficiency

5. Quality Control
   - Seller rating system
   - Product quality spot checks
   - Customer service response times
   - Returns/refunds process streamlining

6. Technology
   - Automated order routing
   - Real-time inventory sync
   - Predictive delivery times
   - Warehouse management system (WMS)

7. Seller Onboarding
   - Best practices training
   - Performance benchmarks
   - Quality standards enforcement
   - Graduated commission for top performers

COST SAVINGS POTENTIAL: 5-10% of operational expenses
SERVICE IMPROVEMENT: Maintain >97% while reducing costs
"""

    def _optimize_marketing(self) -> str:
        """Marketing optimization suggestions."""
        return """
MARKETING OPTIMIZATION RECOMMENDATIONS:
=======================================

CURRENT CHALLENGE:
- High acquisition success
- Zero retention success
- Need to shift from acquisition to retention

STRATEGY SHIFT:

1. Channel Reallocation
   REDUCE:
   - Broad acquisition campaigns
   - Generic paid search
   - One-time deal promotions

   INCREASE:
   - Retention marketing (email, SMS)
   - Remarketing to past customers
   - Loyalty program promotion
   - Referral incentives

2. Segmentation Strategy
   - New customers (welcome series)
   - Single-purchase customers (win-back)
   - Repeat customers (VIP treatment)
   - High-value customers (exclusive offers)
   - At-risk customers (save campaigns)

3. Content Marketing
   - Buying guides by category
   - Product comparison content
   - User reviews and testimonials
   - How-to videos and tutorials
   - Seasonal gift guides

4. Email Marketing Optimization
   - Behavioral triggers (browse, cart abandonment)
   - Personalized recommendations
   - Category-specific campaigns
   - Time-optimized sends
   - A/B test everything

5. Social Media Strategy
   - User-generated content campaigns
   - Influencer partnerships
   - Community building
   - Customer success stories
   - Brand storytelling

6. Paid Advertising
   - Retargeting past customers (highest ROAS)
   - Lookalike audiences of repeat buyers
   - Category-specific campaigns
   - Seasonal promotions
   - Brand awareness (build Olist identity)

7. Partnerships
   - Co-marketing with top sellers
   - Brand collaborations
   - Affiliate program
   - Strategic partnerships

BUDGET ALLOCATION RECOMMENDATION:
- 60% Retention (vs current ~20%)
- 30% Acquisition (vs current ~70%)
- 10% Brand Building (vs current ~10%)

KPIs TO TRACK:
- Repeat purchase rate (primary)
- Email open/click rates
- Retention campaign ROI
- Customer lifetime value
- Cost per retained customer
"""

    def _optimize_product(self) -> str:
        """Product/inventory optimization suggestions."""
        products = self.loader.products
        order_items = self.loader.order_items

        # Sales velocity
        product_sales = order_items.groupby('product_id').size()
        total_products = len(products)
        sold_products = len(product_sales)
        unsold = total_products - sold_products

        return f"""
PRODUCT & INVENTORY OPTIMIZATION:
==================================

CURRENT STATE:
- Total products: {total_products:,}
- Products with sales: {sold_products:,}
- Products with zero sales: {unsold:,} ({unsold/total_products*100:.1f}%)

OPTIMIZATION STRATEGIES:

1. Product Portfolio Rationalization
   - Analyze: 80/20 rule (20% products = 80% revenue)
   - Action: Phase out bottom 20% performers
   - Result: Reduced complexity, better margins

2. Category Optimization
   - Double down on Health & Beauty (top revenue)
   - Expand Watches & Gifts (high AOV)
   - Reduce long-tail low-performers
   - Add trending categories

3. Inventory Strategy
   A. Fast Movers (Top 20%):
      - Higher stock levels
      - Prime warehouse locations
      - Multiple suppliers

   B. Medium Movers (30-70%):
      - Moderate stock
      - Standard fulfillment
      - Monitor trends

   C. Slow Movers (Bottom 30%):
      - Low stock / on-demand
      - Clearance pricing
      - Consider discontinuation

4. New Product Introduction
   - Data-driven selection (market trends)
   - Limited initial inventory
   - Test-and-scale approach
   - Customer voting on new products

5. Bundling Strategy
   - Complementary product bundles
   - Starter kits by category
   - Gift sets (seasonal)
   - Subscription boxes

6. Private Label Opportunity
   - High-margin own-brand products
   - Categories with repeat potential
   - Quality differentiation
   - Brand loyalty building

7. Pricing Optimization
   - Competitive price monitoring
   - Dynamic pricing algorithms
   - Psychological pricing ($19.99 vs $20)
   - Bundle discounts

8. Assortment Planning
   - Seasonal rotations
   - Regional preferences
   - Trend forecasting
   - Customer preferences analysis

EXPECTED OUTCOMES:
- Reduce SKU count by 20% (focus on winners)
- Increase inventory turnover by 30%
- Improve gross margin by 5-8%
- Better cash flow management
"""


def get_chatbot_tools(loader: OlistDataLoader) -> List[BaseTool]:
    """Get all chatbot tools."""
    return [
        DataAnalysisTool(loader=loader),
        VisualizationTool(loader=loader),
        OptimizationTool(loader=loader)
    ]
