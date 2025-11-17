"""
Command-line interface for Olist recommendation system.

Usage:
    python -m src.main --customer_id <ID> --top_k 5
    python -m src.main --train
    python -m src.main --evaluate
    python -m src.main --analytics
"""
import argparse
import sys
from pathlib import Path

from src.data_loader import OlistDataLoader
from src.model import OlistRecommender
from src.evaluate import RecommenderEvaluator
from src.analytics import OlistAnalytics


def setup_argparse():
    """Setup command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='Olist Product Recommendation System',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Main actions
    parser.add_argument(
        '--customer_id',
        type=str,
        help='Customer ID to generate recommendations for'
    )

    parser.add_argument(
        '--top_k',
        type=int,
        default=5,
        help='Number of recommendations to generate (default: 5)'
    )

    parser.add_argument(
        '--method',
        type=str,
        choices=['popularity', 'copurchase', 'hybrid'],
        default='hybrid',
        help='Recommendation method (default: hybrid)'
    )

    parser.add_argument(
        '--train',
        action='store_true',
        help='Train and save the recommendation model'
    )

    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate model performance'
    )

    parser.add_argument(
        '--analytics',
        action='store_true',
        help='Display analytics summary'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='./datos',
        help='Path to data directory (default: ./datos)'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default='./models/recommender.pkl',
        help='Path to model file (default: ./models/recommender.pkl)'
    )

    return parser


def train_model(data_dir: str, model_path: str):
    """Train and save recommendation model."""
    print("="*70)
    print("TRAINING RECOMMENDATION MODEL")
    print("="*70)

    # Load data
    print("\nLoading data...")
    loader = OlistDataLoader(data_dir=data_dir)
    loader.load_all()

    # Train model
    print("\nTraining model...")
    model = OlistRecommender(loader)
    model.fit(min_support=3)

    # Save model
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    print("\nTraining complete!")


def evaluate_model(data_dir: str, model_path: str):
    """Evaluate model performance."""
    print("="*70)
    print("EVALUATING RECOMMENDATION MODEL")
    print("="*70)

    # Load data
    print("\nLoading data...")
    loader = OlistDataLoader(data_dir=data_dir)
    loader.load_all()

    # Split train/test
    print("\nSplitting train/test...")
    evaluator = RecommenderEvaluator(loader, None)
    train_orders, test_orders = evaluator.train_test_split_by_time(test_size=0.2)
    print(f"Train orders: {len(train_orders)}, Test orders: {len(test_orders)}")

    # Train on training set
    train_loader = OlistDataLoader(data_dir=data_dir)
    train_loader.load_all()
    train_loader.orders = train_orders

    print("\nTraining model on training set...")
    model = OlistRecommender(train_loader)
    model.fit(min_support=3)

    # Evaluate on test set
    model.loader = loader
    evaluator = RecommenderEvaluator(loader, model)
    baseline_metrics, improved_metrics = evaluator.evaluate_baseline_vs_improved(test_orders, k=5)

    print("\nEvaluation complete!")


def generate_recommendations(
    customer_id: str,
    top_k: int,
    method: str,
    data_dir: str,
    model_path: str
):
    """Generate recommendations for a customer."""
    print("="*70)
    print("GENERATING PRODUCT RECOMMENDATIONS")
    print("="*70)

    # Load data
    print("\nLoading data...")
    loader = OlistDataLoader(data_dir=data_dir)
    loader.load_all()

    # Load or train model
    model = OlistRecommender(loader)

    if Path(model_path).exists():
        print(f"\nLoading model from {model_path}...")
        model.load(model_path)
    else:
        print(f"\nModel not found at {model_path}. Training new model...")
        model.fit(min_support=3)
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)

    # Check if customer exists
    customer_exists = customer_id in loader.orders['customer_id'].values

    if not customer_exists:
        print(f"\nWarning: Customer ID '{customer_id}' not found in dataset.")
        print("Generating recommendations for a new customer (using popularity)...")

    # Generate recommendations
    print(f"\nGenerating top {top_k} recommendations (method: {method})...")
    recommendations = model.recommend(customer_id, top_k=top_k, method=method)

    # Display recommendations
    print("\n" + "="*70)
    print(f"TOP {top_k} PRODUCT RECOMMENDATIONS FOR CUSTOMER: {customer_id}")
    print("="*70)

    for i, rec in enumerate(recommendations, 1):
        product_id = rec['product_id']
        category = rec.get('product_category_name', 'N/A')
        score = rec.get('purchase_count', rec.get('score', 'N/A'))

        print(f"\n{i}. Product ID: {product_id}")
        print(f"   Category: {category}")
        print(f"   Score: {score}")

    print("\n" + "="*70)


def show_analytics(data_dir: str):
    """Display analytics summary."""
    # Load data
    print("\nLoading data...")
    loader = OlistDataLoader(data_dir=data_dir)
    loader.load_all()

    # Compute analytics
    analytics = OlistAnalytics(loader)
    analytics.print_summary()


def main():
    """Main CLI entry point."""
    parser = setup_argparse()
    args = parser.parse_args()

    # Determine action
    if args.train:
        train_model(args.data_dir, args.model_path)
    elif args.evaluate:
        evaluate_model(args.data_dir, args.model_path)
    elif args.analytics:
        show_analytics(args.data_dir)
    elif args.customer_id:
        generate_recommendations(
            args.customer_id,
            args.top_k,
            args.method,
            args.data_dir,
            args.model_path
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
