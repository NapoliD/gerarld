"""
REST API for Olist recommendation system using FastAPI.

Usage:
    uvicorn src.api:app --reload
"""
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import logging
from pathlib import Path

from src.data_loader import OlistDataLoader
from src.model import OlistRecommender
from src.features import FeatureEngineer
from src.advanced_models import AdvancedRecommender

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Olist Recommendation API",
    description="Product recommendation system for Olist e-commerce platform",
    version="2.0.0"
)

# Global variables for models (loaded on startup)
loader = None
model = None
advanced_model = None
feature_engineer = None


# Pydantic models
class RecommendationRequest(BaseModel):
    customer_id: str = Field(..., description="Customer ID or Customer Unique ID")
    top_k: int = Field(5, ge=1, le=50, description="Number of recommendations")
    method: str = Field("hybrid", description="Recommendation method")


class RecommendationResponse(BaseModel):
    customer_id: str
    recommendations: List[Dict]
    method: str


class CustomerFeaturesResponse(BaseModel):
    customer_unique_id: str
    features: Dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    data_loaded: bool


@app.on_event("startup")
async def startup_event():
    """Load models and data on startup."""
    global loader, model, advanced_model, feature_engineer

    logger.info("Loading data and models...")

    try:
        # Load data
        loader = OlistDataLoader(data_dir='./datos')
        loader.load_all()

        # Load or train basic model
        model_path = Path('./models/recommender.pkl')
        model = OlistRecommender(loader)

        if model_path.exists():
            logger.info(f"Loading model from {model_path}")
            model.load(str(model_path))
        else:
            logger.info("Training new model...")
            model.fit(min_support=3)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(model_path))

        # Initialize advanced models
        advanced_model = AdvancedRecommender(loader)
        feature_engineer = FeatureEngineer(loader)

        logger.info("Startup complete!")

    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None and model.is_fitted,
        "data_loaded": loader is not None
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy" if (model and loader) else "unhealthy",
        "model_loaded": model is not None and model.is_fitted,
        "data_loaded": loader is not None
    }


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get product recommendations for a customer.

    Args:
        request: RecommendationRequest with customer_id, top_k, method

    Returns:
        RecommendationResponse with recommendations
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Get recommendations
        recommendations = model.recommend(
            customer_id=request.customer_id,
            top_k=request.top_k,
            method=request.method
        )

        return {
            "customer_id": request.customer_id,
            "recommendations": recommendations,
            "method": request.method
        }

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend/{customer_id}")
async def get_recommendations_simple(
    customer_id: str,
    top_k: int = Query(5, ge=1, le=50),
    method: str = Query("hybrid", regex="^(popularity|copurchase|hybrid)$")
):
    """
    Simplified GET endpoint for recommendations.

    Args:
        customer_id: Customer ID
        top_k: Number of recommendations
        method: Recommendation method

    Returns:
        Recommendations list
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        recommendations = model.recommend(
            customer_id=customer_id,
            top_k=top_k,
            method=method
        )

        return {
            "customer_id": customer_id,
            "recommendations": recommendations,
            "method": method
        }

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend/advanced/{customer_id}")
async def get_advanced_recommendations(
    customer_id: str,
    top_k: int = Query(5, ge=1, le=50),
    method: str = Query("ensemble", regex="^(content|category|ensemble)$")
):
    """
    Get recommendations using advanced models.

    Args:
        customer_id: Customer unique ID
        top_k: Number of recommendations
        method: Advanced method to use

    Returns:
        Advanced recommendations
    """
    if advanced_model is None:
        raise HTTPException(status_code=503, detail="Advanced model not loaded")

    try:
        if method == "content":
            recommendations = advanced_model.recommend_content_based(customer_id, top_k)
        elif method == "category":
            recommendations = advanced_model.recommend_category_aware(customer_id, top_k)
        elif method == "ensemble":
            recommendations = advanced_model.recommend_ensemble(customer_id, top_k)
        else:
            raise HTTPException(status_code=400, detail="Invalid method")

        return {
            "customer_id": customer_id,
            "recommendations": recommendations,
            "method": method
        }

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features/{customer_id}", response_model=CustomerFeaturesResponse)
async def get_customer_features(customer_id: str):
    """
    Get customer features (RFM, etc.).

    Args:
        customer_id: Customer unique ID

    Returns:
        Customer features
    """
    if feature_engineer is None:
        raise HTTPException(status_code=503, detail="Feature engineer not loaded")

    try:
        features = feature_engineer.get_customer_features(customer_id)

        if not features:
            raise HTTPException(status_code=404, detail="Customer not found")

        return {
            "customer_unique_id": customer_id,
            "features": features
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/explain/{customer_id}/{product_id}")
async def explain_recommendation(customer_id: str, product_id: str):
    """
    Explain why a product is recommended.

    Args:
        customer_id: Customer unique ID
        product_id: Product ID

    Returns:
        Explanation dictionary
    """
    if advanced_model is None:
        raise HTTPException(status_code=503, detail="Advanced model not loaded")

    try:
        explanation = advanced_model.explain_recommendation(customer_id, product_id)

        return explanation

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_statistics():
    """Get dataset statistics."""
    if loader is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    try:
        stats = {
            "total_orders": len(loader.orders),
            "total_customers": loader.customers['customer_unique_id'].nunique(),
            "total_products": len(loader.products),
            "total_categories": loader.products['product_category_name'].nunique(),
            "date_range": {
                "start": str(loader.orders['order_purchase_timestamp'].min()),
                "end": str(loader.orders['order_purchase_timestamp'].max())
            }
        }

        return stats

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
