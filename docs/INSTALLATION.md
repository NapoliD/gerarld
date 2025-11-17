# Installation Guide

Complete installation instructions for the E-commerce Analytics & AI Chatbot project.

---

## Prerequisites

- **Python 3.9+** (Python 3.12 recommended)
- **pip** package manager
- **Git** (for cloning the repository)
- **Google Gemini API Key** (free tier available)

---

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Gerarld
```

### 2. Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies installed:**
- `pandas`, `numpy`, `scikit-learn` - Data processing & ML
- `fastapi`, `uvicorn` - REST API
- `langchain`, `langchain-google-genai` - AI chatbot
- `matplotlib`, `seaborn`, `plotly` - Visualizations
- `pytest` - Testing
- And more... (see `requirements.txt`)

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# Create .env file
touch .env  # Linux/Mac
type nul > .env  # Windows
```

Add your Gemini API key:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
DATA_DIR=./datos
```

### 5. Get Free Gemini API Key

1. Visit: https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the key and paste it in `.env`

**Free Tier Limits:**
- 60 requests per minute
- 1,500 requests per day
- Perfect for development!

### 6. Prepare Data Directory

```bash
# Create data directory
mkdir datos

# Place your CSV files here
# Expected files:
# - olist_customers_dataset.csv
# - olist_orders_dataset.csv
# - olist_products_dataset.csv
# - olist_order_items_dataset.csv
# - olist_order_reviews_dataset.csv
# - olist_order_payments_dataset.csv
# - olist_sellers_dataset.csv
# - olist_geolocation_dataset.csv
# - product_category_name_translation.csv
```

### 7. Verify Installation

**Test imports:**
```bash
python -c "import pandas, langchain, fastapi; print('✓ All dependencies installed')"
```

**Test chatbot:**
```bash
python data_chatbot.py
# Should initialize without errors
# Type 'quit' to exit
```

**Test analytics:**
```bash
python -m src.main --analytics --data_dir datos
```

**Run tests:**
```bash
pytest -v
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError`

**Solution:**
```bash
# Ensure virtual environment is activated
# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: `GOOGLE_API_KEY not found`

**Solution:**
1. Check `.env` file exists in project root
2. Verify `GOOGLE_API_KEY` is set correctly
3. No quotes around the key value
4. Restart terminal/IDE after creating `.env`

### Issue: `langchain-experimental` not found

**Solution:**
```bash
pip install langchain-experimental tabulate
```

### Issue: Gemini API quota exceeded

**Solution:**
- Wait for quota reset (daily limit)
- Check usage at: https://ai.dev/usage
- Consider using a different API key
- Use `models/gemini-2.5-flash` (higher quota) instead of Pro models

### Issue: Data files not found

**Solution:**
1. Place CSV files in `datos/` directory
2. Check file names match exactly (case-sensitive on Linux)
3. Verify paths in code are relative, not absolute

### Issue: Windows encoding errors

**Solution:**
```bash
# Set UTF-8 encoding
set PYTHONIOENCODING=utf-8  # Windows CMD
$env:PYTHONIOENCODING="utf-8"  # PowerShell
```

---

## Platform-Specific Notes

### Windows

- Use `\` for paths or raw strings: `r'C:\path'`
- Activate venv: `.venv\Scripts\activate`
- May need to run as Administrator for some operations

### Linux/Mac

- Use `/` for paths
- Activate venv: `source .venv/bin/activate`
- May need `sudo` for system-wide installations

---

## Optional: Install Development Tools

For contributors:

```bash
# Code formatting
pip install black isort

# Linting
pip install flake8 pylint

# Type checking
pip install mypy

# Documentation
pip install sphinx
```

---

## Verify Installation Success

Run this comprehensive check:

```bash
# 1. Check Python version
python --version

# 2. Check dependencies
pip list | grep -E "pandas|langchain|fastapi"

# 3. Test imports
python -c "from src.data_loader import OlistDataLoader; print('✓ Imports OK')"

# 4. Test chatbot initialization
python -c "from data_chatbot import DataAnalysisChatbot; print('✓ Chatbot OK')"

# 5. Run quick test
pytest tests/ -v --tb=short
```

**All checks passed?** ✅ You're ready to go!

---

## Next Steps

After successful installation:

1. **Read**: [Quick Start Guide](QUICKSTART.md)
2. **Explore**: [Usage Examples](USAGE.md)
3. **Try**: Run the AI chatbot - `python data_chatbot.py`
4. **Learn**: Read [FAQ](FAQ.md) for common questions

---

## Need Help?

- Check [FAQ](FAQ.md) for common issues
- Review [Troubleshooting](#troubleshooting) section above
- Open an issue on GitHub
- Review error messages carefully - they usually indicate the problem

---

**Installation complete! 🎉**

Ready to analyze your e-commerce data!
