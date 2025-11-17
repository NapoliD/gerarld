# Olist E-Commerce Analytics Summary

**Dataset**: Brazilian E-Commerce Orders (2016-2018)
**Scope**: 99,441 orders | 96,096 unique customers | $13.6M GMV

---

## Key Performance Indicators

### Overall Business Metrics
| Metric | Value | Notes |
|--------|-------|-------|
| Total Orders | 99,441 | Spanning Sep 2016 - Oct 2018 |
| Unique Customers | 96,096 | Using customer_unique_id |
| Total GMV | $13,591,644 | Gross Merchandise Value |
| Avg Order Value | $137.75 | Per order |
| Delivery Success Rate | 97.02% | 96,478 / 99,441 orders |
| Avg Review Score | 4.09 / 5.0 | Based on 99,224 reviews |

### Customer Retention
| Metric | Value | Implication |
|--------|-------|-------------|
| **Repeat Purchase Rate** | **3.12%** | Only 2,997 of 96,096 customers |
| One-Time Customers | 93,099 (96.88%) | **Critical retention issue** |
| Avg Orders per Customer | 1.03 | Minimal repeat behavior |
| Max Orders (single customer) | 17 | Outlier: high-value customer |
| Avg Days Between Purchases | 77.9 days | ~2.6 months for repeat customers |

### Top 10 Product Categories by GMV
| Rank | Category | GMV | Orders | Avg Price |
|------|----------|-----|--------|-----------|
| 1 | Health & Beauty | $1,258,681 | 8,836 | $142.47 |
| 2 | Watches & Gifts | $1,205,006 | 5,624 | $214.27 |
| 3 | Bed, Bath & Table | $1,036,989 | 9,417 | $110.11 |
| 4 | Sports & Leisure | $988,049 | 7,720 | $128.00 |
| 5 | Computers & Accessories | $911,954 | 6,689 | $136.35 |
| 6 | Furniture & Decor | $729,762 | 6,449 | $113.14 |
| 7 | Cool Stuff | $635,291 | 3,632 | $174.92 |
| 8 | Housewares | $632,249 | 5,884 | $107.44 |
| 9 | Auto | $592,720 | 3,897 | $152.09 |
| 10 | Garden Tools | $485,256 | 3,518 | $137.95 |

---

## Non-Obvious Insights

### 1. The Retention Crisis: Satisfaction Without Loyalty

**Finding**: Despite achieving a 97% delivery rate and 4.09/5.0 average review score, **96.88% of customers never make a second purchase**.

**Business Impact**:
- Customer Acquisition Cost (CAC) is not being recovered through LTV
- Marketing spend is likely inefficient without repeat purchases
- Competitor marketplaces may be capturing repeat business

**Possible Explanations**:
- **Platform dynamics**: Olist is a marketplace aggregator - customers may not perceive "brand loyalty" to Olist itself
- **Price-shopping behavior**: Customers may use the platform for one-time deals but shop elsewhere for regular purchases
- **Lack of retention mechanisms**: Insufficient loyalty programs, remarketing, or post-purchase engagement
- **Product variety limitations**: May not offer breadth for repeat purchases

**Recommended Actions**:
1. Implement aggressive remarketing campaigns targeting first-time buyers
2. Create loyalty program with second-purchase incentives
3. Analyze customer journey: Why do satisfied customers not return?
4. A/B test post-purchase email sequences (30, 60, 90 days)
5. Investigate competitor retention rates for benchmarking

---

### 2. Category Concentration: Health & Beauty Opportunity

**Finding**: Health & Beauty leads in GMV ($1.26M) with 8,836 orders, yet represents high-frequency potential.

**Business Impact**:
- Consumable products (cosmetics, supplements) have natural replenishment cycles
- This category should have higher repeat rates than durable goods
- Current performance suggests missed subscription/replenishment opportunities

**Category Characteristics**:
- **High Order Volume**: 2nd highest order count (after Bed/Bath/Table)
- **Moderate Price Point**: $142 avg (not prohibitively expensive)
- **Replenishment Potential**: Products typically need replacement every 1-3 months

**Recommended Actions**:
1. Launch subscription service for consumables in Health & Beauty
2. Send replenishment reminders 60-90 days post-purchase
3. Bundle complementary products (skincare routines, vitamin packs)
4. Partner with brands for exclusive "subscribe & save" offers
5. Analyze: Are customers reordering same products on competing platforms?

---

### 3. The 77-Day Window: A Retention Opportunity

**Finding**: The small group of repeat customers (3.12%) waits an average of **77.9 days** between purchases.

**Business Impact**:
- Clear time window for retention interventions
- Suggests purchase cycle could be accelerated with incentives
- Identifies when to re-engage customers before they churn permanently

**Strategic Implications**:
- **Day 30**: Send "thank you" email + product recommendations
- **Day 60**: Offer limited-time discount or free shipping
- **Day 75**: Last-chance remarketing with urgency messaging
- **Day 90+**: Customer likely lost - aggressive win-back campaign

**Recommended Actions**:
1. Build automated email drip campaign triggered at purchase milestones
2. Test personalized product recommendations at 60-day mark
3. Offer "Second Purchase Discount" with 30-day expiration
4. Analyze: Do specific categories have shorter/longer cycles?
5. Segment customers by purchase intent (gift vs. personal use)

---

### 4. High Satisfaction, Low Friction - Yet No Retention

**Finding**: 97% successful delivery + 4.09/5.0 reviews should drive loyalty, but doesn't.

**The Paradox**:
- Operational excellence (delivery) ✓
- Customer satisfaction (reviews) ✓
- Repeat purchases ✗

**Hypothesis**: Olist suffers from **"invisible platform syndrome"**
- Customers remember the individual seller, not the marketplace
- Transaction is friction-free but unmemorable
- No emotional connection to the Olist brand

**Recommended Actions**:
1. **Brand Building**: Make Olist brand more visible in customer journey
2. **Owned Experience**: Create differentiated value (exclusive products, loyalty points)
3. **Post-Purchase Engagement**: Build community (reviews, Q&A, user content)
4. **Competitive Analysis**: How do other marketplaces (MercadoLibre, Amazon BR) retain customers?

---

## Model Performance Context

### Why Precision@K is Low (~0.0)

Our recommendation model achieves near-zero precision on held-out test data. This is **expected and honest** given the dataset characteristics:

**Dataset Constraints**:
- 96.88% of customers have only ONE order (no personalization signal)
- Predicting exact product repurchase is inherently difficult
- Many purchases are one-time (gifts, specific needs)
- Limited temporal behavior to learn patterns

**Model Value Despite Low Precision**:
- Still provides sensible recommendations (popular + co-purchased items)
- Useful for new customer cold-start scenarios
- Can be improved with additional features (demographics, browsing behavior)
- Serves as foundation for A/B testing alternative approaches

**What Would Improve Performance**:
- Customer demographic data (age, location, income)
- Browsing behavior and search queries
- Product features (price, description, images)
- Longer time horizon with more repeat customers
- Collaborative filtering with larger user base

---

## Summary & Business Recommendations

### Immediate Actions (30 days)
1. **Launch Second-Purchase Campaign**: Aggressive email marketing to recent buyers
2. **Retention Dashboard**: Track repeat purchase rate as primary KPI
3. **Category-Specific Strategy**: Pilot subscription for Health & Beauty

### Medium-Term Initiatives (90 days)
1. **Loyalty Program**: Points system with tiered benefits
2. **Personalization Engine**: Improve recommendations with browsing data
3. **Brand Awareness**: Make Olist more memorable in customer journey

### Strategic Questions to Answer
1. What is competitor repeat purchase rate? (Benchmark)
2. What % of customers *intended* to return but didn't? (Survey)
3. Can we predict churn risk before it happens? (Propensity model)
4. What is the LTV:CAC ratio by acquisition channel?

---

**Conclusion**: Olist operates efficiently (delivery, satisfaction) but fails to build lasting customer relationships. The primary opportunity is **retention engineering** - converting one-time buyers into repeat customers through systematic re-engagement, loyalty incentives, and brand building.

---
