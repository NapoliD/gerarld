# Project Architecture

Technical architecture and design decisions for the E-commerce Analytics & AI Chatbot project.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACES                          │
├─────────────┬──────────────┬──────────────┬────────────────┤
│   CLI       │   Chatbot    │   REST API   │   Notebooks    │
└─────┬───────┴──────┬───────┴──────┬───────┴────────┬───────┘
      │              │              │                │
      └──────────────┴──────────────┴────────────────┘
                            │
      ┌─────────────────────┴─────────────────────┐
      │           CORE MODULES                     │
      ├──────────────┬────────────┬───────────────┤
      │ Data Loader  │  Analytics │  AI Chatbot   │
      │ Churn Model  │  RecEngine │  Generic Tools│
      └──────┬───────┴────────┬───┴───────┬───────┘
             │                │           │
      ┌──────┴────────────────┴───────────┴──────┐
      │              DATA LAYER                    │
      ├─────────────┬──────────────┬──────────────┤
      │  CSV Files  │  Models      │  Outputs     │
      │  (datos/)   │  (models/)   │  (outputs/)  │
      └─────────────┴──────────────┴──────────────┘
```

---

## Core Components

### 1. Data Layer

**Purpose**: Data storage and access

**Components**:
- `datos/` - Raw CSV files (9 datasets)
- `models/` - Serialized ML models (.pkl files)
- `outputs/` - Generated visualizations and reports

**Design Decision**: Local file storage for simplicity and privacy
- No database required for prototype
- Easy to set up and use
- Suitable for datasets < 1GB

### 2. Data Access Layer

**Module**: `src/data_loader.py`

**Purpose**: Centralized data loading and preprocessing

**Key Classes**:
```python
class OlistDataLoader:
    def load_customers() -> pd.DataFrame
    def load_orders() -> pd.DataFrame
    def load_order_items() -> pd.DataFrame
    def prepare_churn_dataset() -> pd.DataFrame
    # ... more loaders
```

**Features**:
- Lazy loading (loads only when needed)
- Data validation
- Type conversions
- Missing value handling
- Reproducible preprocessing

**Design Pattern**: Singleton pattern for data loader instance

### 3. Business Logic Layer

#### a) Analytics Module

**Module**: `src/analytics.py`

**Purpose**: Business metrics and KPIs calculation

**Key Functions**:
```python
def analyze_business_metrics(customers, orders, items)
def calculate_repeat_rate(customers, orders)
def analyze_category_performance(items, products)
def analyze_delivery_success(orders)
```

**Output**: Dictionary of metrics
```python
{
    'total_gmv': 13600000.00,
    'repeat_purchase_rate': 0.0312,
    'delivery_success_rate': 0.9702,
    'avg_review_score': 4.09
}
```

#### b) Churn Prediction

**Module**: `src/churn_predictor.py`

**Purpose**: Identify at-risk customers

**Architecture**:
```python
class ChurnPredictor:
    def __init__(self)
        # Random Forest Classifier
        # SMOTE for class imbalance
        # StandardScaler for normalization

    def train(df)
    def predict(df)
    def predict_proba(df)
    def evaluate(df)
    def save_model(path)
    def load_model(path)
```

**Model Pipeline**:
1. Feature engineering (RFM metrics)
2. Handle class imbalance (SMOTE)
3. Train Random Forest
4. Cross-validation
5. Save model artifacts

**Features**:
- Recency, Frequency, Monetary (RFM)
- Review scores
- Delivery metrics
- Category preferences

#### c) Recommendation Engine

**Module**: `src/recommendation_engine.py`

**Purpose**: Product recommendations

**Architecture**:
```python
class RecommendationEngine:
    def recommend_by_popularity(n)
    def recommend_by_copurchase(customer_id, n)
    def recommend_hybrid(customer_id, n)
```

**Algorithms**:
1. **Popularity-based**: Most purchased products
2. **Co-purchase**: Products bought together
3. **Hybrid**: Combines both approaches

**Co-purchase Matrix**:
```python
{
    'product_A': {
        'product_B': 0.75,  # 75% confidence
        'product_C': 0.50
    }
}
```

#### d) AI Chatbot

**Modules**:
- `data_chatbot.py` - Main chatbot with custom tools
- `data_chatbot_pandas.py` - Advanced chatbot with pandas agent
- `src/generic_data_tools.py` - Analysis tools

**Architecture**:
```python
# Custom Tools Chatbot
DataAnalysisChatbot:
    ├── LangChain Agent (ReAct)
    ├── Gemini 2.5 Flash LLM
    ├── 5 Custom Tools:
    │   ├── ListCSVsTool
    │   ├── InspectCSVTool
    │   ├── AnalyzeCSVTool
    │   ├── DataQualityTool
    │   └── DataImprovementTool
    └── ConversationBufferMemory

# Pandas Agent Chatbot
PandasDataChatbot:
    ├── LangChain Pandas Agent
    ├── Gemini 2.5 Flash LLM
    ├── Dynamic Code Generation
    └── Direct DataFrame Access
```

**Tools Workflow**:
```
User Question
    ↓
LangChain Agent (ReAct)
    ↓
Tool Selection
    ↓
Tool Execution (Pandas Analysis)
    ↓
Result Formatting
    ↓
LLM Response Generation
    ↓
User Response
```

**Design Pattern**: Agent pattern with ReAct (Reasoning + Acting)

### 4. API Layer

**Module**: `src/api.py`

**Framework**: FastAPI

**Architecture**:
```python
FastAPI App
├── /health - Health check
├── /analytics/* - Business metrics
│   ├── /customer_distribution
│   ├── /top_categories
│   └── /review_distribution
├── /recommendations/{customer_id}/{n} - Get recommendations
└── /churn/predict - Churn prediction
```

**Design Principles**:
- RESTful design
- Automatic OpenAPI docs
- Pydantic models for validation
- Async endpoints for performance
- CORS enabled for web clients

### 5. Interface Layer

#### a) CLI

**Module**: `src/main.py`

**Purpose**: Command-line interface

**Design Pattern**: Argument parser with subcommands

```bash
python -m src.main [--analytics] [--train] [--evaluate]
                    [--customer_id ID] [--top_k N] [--method METHOD]
```

#### b) Chatbot Interface

**Modules**: `data_chatbot.py`, `data_chatbot_pandas.py`

**Purpose**: Natural language data analysis

**Design**: Interactive REPL (Read-Eval-Print Loop)

#### c) Notebooks

**Directory**: `notebooks/`

**Purpose**: Exploratory analysis and prototyping

**Files**:
- `01_data_exploration.py` - Initial EDA
- `02_verify_repeat_customers.py` - Churn analysis
- `03_eda_comprehensive.py` - Full analysis
- `04_data_balance_analysis.py` - Data balance

---

## Design Decisions

### 1. Why Pandas over Spark?

**Decision**: Use pandas for data processing

**Rationale**:
- Dataset size (~100K orders, ~500MB) fits in memory
- Faster development time
- Easier debugging and testing
- No cluster setup required
- Sufficient performance for use case

**When to switch to Spark**:
- Data > 10GB
- Distributed processing needed
- Real-time streaming required

### 2. Why Random Forest for Churn?

**Decision**: Use Random Forest Classifier

**Rationale**:
- Handles non-linear relationships
- Feature importance available
- Robust to outliers
- No extensive hyperparameter tuning needed
- Good performance on imbalanced data

**Alternatives considered**:
- Logistic Regression (too simple)
- XGBoost (overkill for dataset size)
- Neural Networks (requires more data)

### 3. Why Co-purchase over Collaborative Filtering?

**Decision**: Use co-purchase analysis for recommendations

**Rationale**:
- More interpretable (products bought together)
- Works with low repeat purchase rate (96.88% churn)
- Faster training and inference
- No cold-start problem for products
- Explainable recommendations

**Trade-off**: Less personalized than CF, but more practical

### 4. Why FastAPI over Flask?

**Decision**: Use FastAPI for REST API

**Rationale**:
- Automatic OpenAPI/Swagger docs
- Built-in data validation (Pydantic)
- Async support (better performance)
- Type hints (better IDE support)
- Modern and actively maintained

### 5. Why Gemini over OpenAI?

**Decision**: Use Google Gemini for AI chatbot

**Rationale**:
- Free tier available (1,500 requests/day)
- No credit card required
- Comparable performance to GPT
- Easy integration with LangChain
- Good for prototyping

**Note**: Architecture supports swapping LLMs easily

### 6. Why Custom Tools over Pandas Agent Only?

**Decision**: Provide both custom tools and pandas agent

**Rationale**:
- Custom tools: Faster, predictable, easier to control
- Pandas agent: Flexible, handles complex queries
- Best of both worlds approach
- Users can choose based on needs

---

## Data Flow

### 1. Analytics Flow

```
CSV Files → OlistDataLoader → Analytics Module → Metrics Dictionary
                    ↓
            DataFrame Cache
                    ↓
            Visualization Module → Charts (outputs/)
```

### 2. Churn Prediction Flow

```
CSV Files → prepare_churn_dataset() → Feature Engineering
                                            ↓
                                    Train/Test Split
                                            ↓
                                    SMOTE Oversampling
                                            ↓
                                    Random Forest Training
                                            ↓
                                    Model Evaluation
                                            ↓
                                    Save Model (.pkl)
```

### 3. Recommendation Flow

```
Order Items CSV → Build Co-purchase Matrix → Save Model
                        ↓
                Customer Purchase History
                        ↓
                Find Related Products
                        ↓
                Score & Rank
                        ↓
                Return Top N
```

### 4. Chatbot Flow

```
User Question → LangChain Agent → Tool Selection
                                        ↓
                                Load CSV (if needed)
                                        ↓
                                Pandas Analysis
                                        ↓
                                Format Results
                                        ↓
                                LLM Response
                                        ↓
                                User Response
```

---

## Scalability Considerations

### Current Limitations

1. **Data Size**: In-memory processing limits to ~10GB
2. **Concurrency**: Single-threaded pandas operations
3. **Real-time**: Batch processing only
4. **Distribution**: Single machine only

### Scaling Strategies

**For 10x More Data (1M+ orders)**:
- Use Dask for parallel pandas
- Implement data chunking
- Add database (PostgreSQL)
- Cache frequent queries

**For 100x More Data (10M+ orders)**:
- Migrate to Spark
- Use distributed storage (S3, HDFS)
- Implement data partitioning
- Add streaming pipeline

**For Production Deployment**:
- Add authentication (JWT)
- Implement rate limiting
- Use Redis for caching
- Add monitoring (Prometheus, Grafana)
- Deploy with Docker/Kubernetes
- Add CI/CD pipeline

---

## Testing Strategy

### Unit Tests

**Location**: `tests/`

**Coverage**:
- Data loading functions
- Feature engineering
- Model training/prediction
- API endpoints

**Framework**: pytest

**Run**: `pytest -v tests/`

### Integration Tests

**Focus**:
- End-to-end workflows
- API integration
- Model pipeline

### Performance Tests

**Metrics**:
- Data loading time
- Model training time
- API response time
- Memory usage

---

## Security Considerations

### Current Implementation

1. **API Keys**: Stored in `.env` (gitignored)
2. **Data Privacy**: All processing local (no cloud upload)
3. **Input Validation**: Pydantic models
4. **Error Handling**: Try-except blocks

### Production Requirements

1. **Authentication**: Add JWT tokens
2. **Authorization**: Role-based access control
3. **Rate Limiting**: Prevent abuse
4. **Logging**: Audit trail
5. **Encryption**: HTTPS/TLS
6. **SQL Injection**: Parameterized queries (if using DB)

---

## Future Enhancements

### Short-term (1-3 months)

- [ ] Add more ML models (XGBoost, LightGBM)
- [ ] Implement A/B testing framework
- [ ] Add real-time dashboards (Streamlit)
- [ ] Improve chatbot with memory

### Medium-term (3-6 months)

- [ ] Migrate to database (PostgreSQL)
- [ ] Add user authentication
- [ ] Implement caching (Redis)
- [ ] Add monitoring (Prometheus)
- [ ] Dockerize application

### Long-term (6-12 months)

- [ ] Migrate to Spark for big data
- [ ] Implement streaming pipeline
- [ ] Add deep learning models
- [ ] Multi-language support
- [ ] Cloud deployment (AWS/GCP/Azure)

---

## Tech Stack Summary

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.9+ |
| Data Processing | pandas | 2.0+ |
| ML Framework | scikit-learn | 1.3+ |
| Web Framework | FastAPI | 0.104+ |
| AI Framework | LangChain | 0.1+ |
| LLM | Google Gemini | 2.5 Flash |
| Visualization | matplotlib, seaborn | Latest |
| Testing | pytest | 7.4+ |
| Documentation | Markdown | - |
| Version Control | Git | - |

---

## References

- **LangChain**: https://python.langchain.com/
- **FastAPI**: https://fastapi.tiangolo.com/
- **Google Gemini**: https://ai.google.dev/
- **Scikit-learn**: https://scikit-learn.org/
- **Pandas**: https://pandas.pydata.org/

---

**Architecture Version**: 1.0
**Last Updated**: November 2025
