# Olist Data Analysis Chatbot Guide

An AI-powered chatbot that analyzes your e-commerce data, generates visualizations, and provides optimization recommendations using Google Gemini and LangChain.

---

## What Can It Do?

The chatbot can:
- **Analyze data distributions** (customers, products, orders, reviews, revenue)
- **Generate visualizations** on-demand (charts, graphs, distributions)
- **Provide optimization suggestions** (retention, revenue, operations, marketing, products)
- **Answer questions** about your data using natural language
- **Remember conversation context** for follow-up questions

---

## Quick Start (5 Minutes)

### 1. Get Your Free Gemini API Key

1. Go to: https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key

### 2. Set Up Environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your API key
# GOOGLE_API_KEY=your_actual_key_here
```

**Windows PowerShell:**
```powershell
$env:GOOGLE_API_KEY="your_api_key_here"
```

**Linux/Mac:**
```bash
export GOOGLE_API_KEY="your_api_key_here"
```

### 3. Install Dependencies

```bash
pip install langchain langchain-google-genai langchain-core google-generativeai python-dotenv
```

Or install all project dependencies:
```bash
pip install -r requirements.txt
```

### 4. Run the Chatbot

```bash
python chatbot.py
```

That's it! The chatbot will load your data and be ready to answer questions.

---

## Example Conversations

### Example 1: Basic Analysis

```
You: Analyze customer behavior

Chatbot: Let me analyze the customer data...

CUSTOMER ANALYSIS:
==================
Total Customers: 96,096
Single Purchase Customers: 93,099 (96.88%)
Repeat Purchase Customers: 2,997 (3.12%)

⚠️ CRITICAL ISSUE: 96.88% of customers never return!

RECOMMENDATIONS:
1. Focus on retention - current churn rate is extreme
2. Implement loyalty program for second purchase
3. Send re-engagement emails at 30/60/90 days
```

### Example 2: Visualization Request

```
You: Show me a chart of customer distribution

Chatbot: I'll generate a visualization of customer distribution...

Visualization saved to: outputs/chatbot_viz/customer_distribution.png

Key Insight: 96.9% of customers make only one purchase!

You: Now show revenue by category

Chatbot: Creating revenue visualization...

Visualization saved to: outputs/chatbot_viz/category_revenue.png

Top category: cama_mesa_banho with $1,036,989.00
```

### Example 3: Optimization Advice

```
You: How can we improve retention?

Chatbot: Let me provide retention optimization recommendations...

RETENTION OPTIMIZATION RECOMMENDATIONS:
=======================================

IMMEDIATE ACTIONS (Week 1-2):
1. Email Campaign Setup
   - Day 7: Thank you + product recommendations
   - Day 30: "Miss you" with 10% discount
   - Day 60: Last chance offer with free shipping

2. Quick Wins
   - Second purchase discount (15-20% off, 7-day expiration)
   - Free shipping on second order
   ...
```

### Example 4: Follow-up Questions

```
You: What's our average order value?

Chatbot: Analyzing revenue patterns...
Average Order Value (AOV): $120.65

You: How can we increase it?

Chatbot: [Uses conversation context]
Here are strategies to increase AOV from $120.65 to $138.75 (+15%):

1. Free shipping threshold at $180
2. "Frequently bought together" bundles
3. Volume discounts (Buy 2, Get 10% off)
...
```

---

## Available Commands

### Analysis Commands

Ask about specific areas:
- "Analyze customers"
- "Analyze products"
- "Analyze orders"
- "Analyze reviews"
- "Analyze revenue"
- "Analyze temporal patterns"

### Visualization Commands

Request specific charts:
- "Show customer distribution"
- "Generate category revenue chart"
- "Create review scores visualization"
- "Show price distribution"
- "Display temporal trends"

### Optimization Commands

Get recommendations for:
- "Suggest retention optimizations"
- "How to increase revenue?"
- "Optimize operations"
- "Marketing strategies"
- "Product recommendations"

### General Questions

Ask anything about your data:
- "What's the repeat purchase rate?"
- "Which category generates most revenue?"
- "How many orders do we have?"
- "What's the average review score?"
- "When is our peak season?"

---

## How It Works

### Architecture

```
┌─────────────┐
│   You       │ ──> Ask questions
└─────────────┘
      │
      v
┌─────────────────────────────┐
│   Gemini AI (LangChain)     │ ──> Reasons about question
└─────────────────────────────┘
      │
      v
┌─────────────────────────────┐
│   Tool Selection            │ ──> Chooses right tool
└─────────────────────────────┘
      │
      v
┌──────────────────┬──────────────────┬─────────────────┐
│  analyze_data    │  generate_viz    │  suggest_opt    │
│  distribution    │  ualization      │  imizations     │
└──────────────────┴──────────────────┴─────────────────┘
      │
      v
┌─────────────────────────────┐
│   Your Olist Data           │ ──> Processes real data
└─────────────────────────────┘
      │
      v
┌─────────────────────────────┐
│   Results & Insights        │ ──> Returns answer
└─────────────────────────────┘
```

### Tools Available

1. **DataAnalysisTool**
   - Analyzes distributions and statistics
   - Computes KPIs and metrics
   - Identifies patterns and anomalies

2. **VisualizationTool**
   - Generates charts and graphs
   - Saves high-resolution images
   - Supports multiple visualization types

3. **OptimizationTool**
   - Provides data-driven recommendations
   - Suggests actionable strategies
   - Estimates potential impact

---

## Advanced Usage

### Programmatic Use

You can also use the chatbot programmatically:

```python
from chatbot import OlistChatbot

# Initialize
chatbot = OlistChatbot(
    data_dir='datos',
    api_key='your_api_key'
)

# Ask questions
response = chatbot.chat("What's the repeat purchase rate?")
print(response)

# Generate visualization
response = chatbot.chat("Show me revenue by category")
print(response)

# Get recommendations
response = chatbot.chat("Suggest retention optimizations")
print(response)
```

### Batch Questions

```python
questions = [
    "Analyze customer behavior",
    "Show temporal trends",
    "Suggest revenue optimizations"
]

for question in questions:
    print(f"\nQ: {question}")
    response = chatbot.chat(question)
    print(f"A: {response}\n")
```

### Custom Data Directory

```bash
python chatbot.py --data_dir /path/to/your/data
```

---

## Tips for Best Results

### 1. Be Specific

**Good:**
- "Analyze customer retention patterns"
- "Show me top 10 categories by revenue"
- "What percentage of customers make repeat purchases?"

**Less Good:**
- "Tell me about data"
- "Show me something"
- "What's the analysis?"

### 2. Ask Follow-up Questions

The chatbot remembers context:
```
You: What's our repeat purchase rate?
Bot: 3.12%

You: How does that compare to industry average?
Bot: [Uses context about repeat purchase rate]
```

### 3. Request Visualizations When Discussing Distributions

When talking about distributions, ask for charts:
- "Show me a chart of..."
- "Can you visualize..."
- "Generate a graph of..."

### 4. Ask for Specific Metrics

Instead of general questions, ask for specific KPIs:
- "What's the average order value?"
- "How many repeat customers do we have?"
- "What's the delivery success rate?"

---

## Troubleshooting

### Error: "Google API Key not found"

**Solution:**
```bash
# Set environment variable
export GOOGLE_API_KEY="your_key_here"

# Or pass as argument
python chatbot.py --api_key your_key_here

# Or create .env file
echo "GOOGLE_API_KEY=your_key_here" > .env
```

### Error: Module not found

**Solution:**
```bash
pip install langchain langchain-google-genai google-generativeai python-dotenv
```

### Chatbot gives generic answers

**Reason:** Not using tools properly

**Solution:** Rephrase question to be more specific:
- Instead of "Tell me about customers"
- Try: "Analyze customer distribution"

### Visualization not generating

**Check:**
1. matplotlib and seaborn are installed
2. Output directory is writable
3. Ask explicitly: "Generate visualization of X"

### API Rate Limits

**Gemini Free Tier Limits:**
- 60 requests per minute
- 1,500 requests per day

**Solution:** If you hit limits, wait a minute or upgrade to paid tier.

---

## Cost Information

### Gemini API Pricing

**Free Tier (no credit card required):**
- 60 requests/minute
- 1,500 requests/day
- Perfect for personal use and development

**Paid Tier (if you need more):**
- Gemini Pro: Very affordable ($0.00025 per 1K characters)
- Example: 1 million characters ~ $0.25

**For this chatbot:**
- Average question: 500-1000 characters
- Cost per conversation: < $0.01
- Very cost-effective!

---

## Examples by Use Case

### Business Owner / Stakeholder

```
Q: What's our biggest problem right now?
Q: Show me revenue trends
Q: How can we increase profits?
Q: What should be our top priority?
```

### Data Analyst

```
Q: Analyze customer behavior in detail
Q: Show correlations between metrics
Q: Generate all available visualizations
Q: What are the data quality issues?
```

### Marketing Manager

```
Q: Suggest marketing optimizations
Q: When is our peak season?
Q: Which customer segment should we target?
Q: What's the best channel for retention?
```

### Product Manager

```
Q: Analyze product performance
Q: Which categories should we expand?
Q: Show me slow-moving inventory
Q: Suggest product bundling strategies
```

---

## Extending the Chatbot

### Add New Tools

Create custom tools in `src/chatbot_tools.py`:

```python
class MyCustomTool(BaseTool):
    name: str = "my_custom_analysis"
    description: str = "What this tool does..."

    def _run(self, input: str) -> str:
        # Your analysis logic
        return "Results..."
```

### Add New Analysis Types

Extend `DataAnalysisTool` with new analysis methods:

```python
def _analyze_shipping(self) -> str:
    """Analyze shipping patterns."""
    # Your shipping analysis
    return "Shipping insights..."
```

### Customize Agent Behavior

Modify `AGENT_PROMPT` in `chatbot.py` to change how the agent thinks and responds.

---

## Comparison with Other Solutions

### vs Manual Analysis
- **Speed**: Instant vs hours
- **Consistency**: Always accurate vs human error
- **Availability**: 24/7 vs business hours
- **Cost**: Free (Gemini) vs analyst time

### vs Business Intelligence Tools
- **Setup**: Minutes vs weeks
- **Customization**: Code-based vs click-based
- **Natural Language**: Yes vs SQL/query languages
- **Cost**: Free/very cheap vs $$$$/month

### vs ChatGPT Data Analysis
- **Data Privacy**: Your data stays local
- **Customization**: Built for your specific data
- **Tools**: Custom analytics tools
- **Context**: Understands your business

---

## Security & Privacy

### Data Privacy

- Your data **never leaves your computer**
- Only the questions and tool results are sent to Gemini API
- Raw data is NOT sent to Google

### API Key Security

**DO NOT:**
- Commit .env file to git
- Share your API key publicly
- Hard-code API keys

**DO:**
- Use environment variables
- Keep .env in .gitignore
- Rotate keys periodically

---

## Future Enhancements

Potential additions:
- [ ] Slack/Teams integration
- [ ] Scheduled reports via email
- [ ] Web interface (Streamlit/Gradio)
- [ ] Voice input/output
- [ ] Multi-language support
- [ ] Export conversations as PDF
- [ ] Integration with data warehouse
- [ ] Automated insights generation

---

## FAQ

### Q: Do I need programming skills?

A: No! Just run the chatbot and ask questions in plain English.

### Q: Can I use a different LLM?

A: Yes! You can modify `chatbot.py` to use OpenAI, Anthropic Claude, or local models (Ollama).

### Q: How accurate are the recommendations?

A: Recommendations are based on actual data analysis. However, they should be validated with domain expertise.

### Q: Can I use this for production?

A: The analysis tools are production-ready. For production chatbot deployment, consider:
- Rate limiting
- Error handling
- Logging
- Monitoring
- User authentication

### Q: What if I run out of free API quota?

A: Either wait for quota reset (daily) or upgrade to paid tier (very affordable).

---

## Support & Resources

### Get Help

1. Check this guide
2. Review example conversations
3. Check tool descriptions in code
4. Ask the chatbot itself: "What can you do?"

### Learn More

- **LangChain Docs**: https://python.langchain.com/docs/
- **Gemini API Docs**: https://ai.google.dev/docs
- **Project README**: `README.md`
- **Data Analysis Guide**: `DATA_ANALYSIS_GUIDE.md`

---

## Example Session Transcript

```
$ python chatbot.py

================================================================================
INITIALIZING OLIST DATA ANALYSIS CHATBOT
================================================================================

[1/4] Loading Olist dataset...
Loaded olist_orders_dataset.csv: 99441 rows, 8 columns
Loaded olist_products_dataset.csv: 32951 rows, 9 columns
...
[OK] Data loaded successfully

[2/4] Setting up analysis tools...
[OK] 3 tools ready:
  - analyze_data_distribution: Analyzes the distribution of data...
  - generate_visualization: Generates visualizations of data...
  - suggest_optimizations: Suggests data-driven optimizations...

[3/4] Connecting to Gemini AI...
[OK] Connected to Gemini Pro

[4/4] Creating conversational agent...
[OK] Agent ready!

================================================================================
CHATBOT READY - Type your questions or 'quit' to exit
================================================================================

Examples of what you can ask:
  - Analyze customer behavior
  - Show me the revenue distribution
  - Generate a visualization of review scores
  - Suggest optimizations for retention

You: What's the biggest problem with our business?

Chatbot: Let me analyze the customer data...

Based on the analysis, the BIGGEST problem is:

**EXTREME CUSTOMER CHURN: 96.88% of customers never return**

Despite:
- 97% delivery success rate
- 4.09/5.0 average review score
- Strong operational performance

This is a critical retention crisis. Would you like me to suggest specific optimizations to address this?

You: Yes, please suggest retention strategies

Chatbot: I'll provide comprehensive retention optimization recommendations...

RETENTION OPTIMIZATION RECOMMENDATIONS:
=======================================

IMMEDIATE ACTIONS (Week 1-2):
1. Email Campaign Setup
   - Day 7: Thank you + product recommendations
   - Day 30: "Miss you" with 10% discount
   ...

[Full recommendations displayed]

You: Show me a visualization of customer distribution

Chatbot: Generating visualization...

Visualization saved to: outputs/chatbot_viz/customer_distribution.png

Key Insight: 96.9% of customers make only one purchase!

You: quit

Chatbot: Goodbye! Thanks for using the Olist Data Analysis Chatbot.
```

---

**Ready to start? Get your API key and run `python chatbot.py`!**

**Last Updated:** November 2025
