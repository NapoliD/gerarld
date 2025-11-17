"""
Market Basket Analysis using Apriori algorithm.
Finds association rules for product recommendations.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


class MarketBasketAnalyzer:
    """Analyze market basket patterns for cross-selling."""

    def __init__(self, loader):
        """
        Initialize market basket analyzer.

        Args:
            loader: OlistDataLoader instance.
        """
        self.loader = loader
        self.transactions = None
        self.frequent_itemsets = None
        self.association_rules = None

    def prepare_transactions(self) -> pd.DataFrame:
        """
        Prepare transaction data in basket format.

        Returns:
            DataFrame with transactions.
        """
        # Get order-product pairs
        items = self.loader.order_items.merge(
            self.loader.products[['product_id', 'product_category_name']],
            on='product_id',
            how='left'
        )

        # Group by order to get product lists
        transactions = (
            items.groupby('order_id')['product_category_name']
            .apply(list)
            .reset_index()
        )

        transactions.columns = ['order_id', 'products']

        self.transactions = transactions

        print(f"Prepared {len(transactions)} transactions")
        print(f"Avg items per transaction: {transactions['products'].apply(len).mean():.2f}")

        return transactions

    def find_frequent_itemsets(self, min_support: float = 0.01) -> pd.DataFrame:
        """
        Find frequent itemsets (simplified version).

        Args:
            min_support: Minimum support threshold.

        Returns:
            DataFrame of frequent itemsets.
        """
        if self.transactions is None:
            self.prepare_transactions()

        print(f"\nFinding frequent itemsets (min_support={min_support})...")

        # Count category pairs
        from collections import defaultdict

        pair_counts = defaultdict(int)
        total_transactions = len(self.transactions)

        for products in self.transactions['products']:
            # Remove duplicates and sort
            unique_products = list(set(products))

            # Generate pairs
            for i in range(len(unique_products)):
                for j in range(i + 1, len(unique_products)):
                    pair = tuple(sorted([unique_products[i], unique_products[j]]))
                    pair_counts[pair] += 1

        # Filter by min_support
        min_count = min_support * total_transactions

        frequent_pairs = []
        for pair, count in pair_counts.items():
            if count >= min_count:
                support = count / total_transactions
                frequent_pairs.append({
                    'itemset': pair,
                    'item_1': pair[0],
                    'item_2': pair[1],
                    'support': support,
                    'count': count
                })

        df = pd.DataFrame(frequent_pairs)
        df = df.sort_values('support', ascending=False)

        self.frequent_itemsets = df

        print(f"Found {len(df)} frequent category pairs")

        return df

    def generate_association_rules(
        self,
        min_confidence: float = 0.3,
        min_lift: float = 1.0
    ) -> pd.DataFrame:
        """
        Generate association rules from frequent itemsets.

        Args:
            min_confidence: Minimum confidence threshold.
            min_lift: Minimum lift threshold.

        Returns:
            DataFrame of association rules.
        """
        if self.frequent_itemsets is None:
            self.find_frequent_itemsets()

        print(f"\nGenerating association rules...")
        print(f"  Min confidence: {min_confidence}")
        print(f"  Min lift: {min_lift}")

        # Calculate support for individual items
        if self.transactions is None:
            self.prepare_transactions()

        item_support = {}
        total_transactions = len(self.transactions)

        for products in self.transactions['products']:
            for product in set(products):
                item_support[product] = item_support.get(product, 0) + 1

        for item in item_support:
            item_support[item] /= total_transactions

        # Generate rules for each frequent pair
        rules = []

        for _, row in self.frequent_itemsets.iterrows():
            item_1, item_2 = row['item_1'], row['item_2']
            support_12 = row['support']

            # Rule: item_1 -> item_2
            support_1 = item_support.get(item_1, 0)
            if support_1 > 0:
                confidence_1_to_2 = support_12 / support_1
                lift_1_to_2 = confidence_1_to_2 / item_support.get(item_2, 0.001)

                if confidence_1_to_2 >= min_confidence and lift_1_to_2 >= min_lift:
                    rules.append({
                        'antecedent': item_1,
                        'consequent': item_2,
                        'support': support_12,
                        'confidence': confidence_1_to_2,
                        'lift': lift_1_to_2
                    })

            # Rule: item_2 -> item_1
            support_2 = item_support.get(item_2, 0)
            if support_2 > 0:
                confidence_2_to_1 = support_12 / support_2
                lift_2_to_1 = confidence_2_to_1 / item_support.get(item_1, 0.001)

                if confidence_2_to_1 >= min_confidence and lift_2_to_1 >= min_lift:
                    rules.append({
                        'antecedent': item_2,
                        'consequent': item_1,
                        'support': support_12,
                        'confidence': confidence_2_to_1,
                        'lift': lift_2_to_1
                    })

        df = pd.DataFrame(rules)
        df = df.sort_values('lift', ascending=False)

        self.association_rules = df

        print(f"Generated {len(df)} association rules")

        if len(df) > 0:
            print(f"\nTop 5 rules by lift:")
            print(df.head()[['antecedent', 'consequent', 'confidence', 'lift']])

        return df

    def recommend_from_basket(
        self,
        current_basket: List[str],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Recommend products based on current basket.

        Args:
            current_basket: List of categories/products in basket.
            top_k: Number of recommendations.

        Returns:
            List of recommendations.
        """
        if self.association_rules is None:
            self.generate_association_rules()

        # Find rules where antecedent is in basket
        recommendations = {}

        for item in current_basket:
            matching_rules = self.association_rules[
                self.association_rules['antecedent'] == item
            ]

            for _, rule in matching_rules.iterrows():
                consequent = rule['consequent']
                if consequent not in current_basket:
                    # Weighted score: confidence * lift
                    score = rule['confidence'] * rule['lift']
                    recommendations[consequent] = max(
                        recommendations.get(consequent, 0),
                        score
                    )

        # Sort and return top K
        sorted_recs = sorted(
            recommendations.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            {
                'category': category,
                'score': float(score),
                'method': 'market_basket'
            }
            for category, score in sorted_recs[:top_k]
        ]


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from data_loader import OlistDataLoader

    loader = OlistDataLoader(data_dir='../datos')
    loader.load_all()

    analyzer = MarketBasketAnalyzer(loader)
    analyzer.prepare_transactions()

    # Find frequent itemsets
    frequent = analyzer.find_frequent_itemsets(min_support=0.01)
    print(f"\nTop 10 frequent pairs:")
    print(frequent.head(10))

    # Generate rules
    rules = analyzer.generate_association_rules(min_confidence=0.3, min_lift=1.5)

    # Test recommendation
    basket = ['cama_mesa_banho', 'beleza_saude']
    print(f"\nCurrent basket: {basket}")
    recs = analyzer.recommend_from_basket(basket, top_k=5)
    print(f"Recommendations: {recs}")
