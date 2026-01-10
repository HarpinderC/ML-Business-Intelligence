# Feature Definitions

Complete documentation of all 30 engineered features for churn prediction.

---

## RFM Features (6 features)

### 1. Recency
- **Definition:** Days since last purchase
- **Calculation:** (Latest date in training period) - (Customer's last purchase date)
- **Range:** 0-22 days
- **Interpretation:** Lower = more recent = less likely to churn

### 2. Frequency  
- **Definition:** Total number of unique purchase occasions
- **Calculation:** Count of distinct InvoiceNo per customer
- **Range:** 1-35 purchases
- **Interpretation:** Higher = more loyal = less likely to churn

### 3. Monetary
- **Definition:** Total revenue generated
- **Calculation:** Sum of (Quantity × UnitPrice) across all transactions
- **Range:** £15 - £27,834
- **Interpretation:** Higher = more valuable customer

### 4. Tenure
- **Definition:** Customer lifespan in days
- **Calculation:** (Last purchase date) - (First purchase date)
- **Range:** 0-22 days
- **Interpretation:** Longer = more established relationship

### 5. AvgOrderValue
- **Definition:** Average transaction value
- **Calculation:** Monetary / Frequency
- **Range:** £15 - £8,142
- **Interpretation:** Spending per visit

### 6. AvgBasketSize
- **Definition:** Average items per purchase
- **Calculation:** Total Quantity / Frequency
- **Range:** 1 - 612 items
- **Interpretation:** Purchase volume per visit

---

## Probabilistic Features (8 features)

### 7. prob_alive
- **Source:** BG/NBD model output
- **Definition:** Probability customer is still "alive" (not churned)
- **Range:** 0-1
- **Interpretation:** Higher = more likely to return

### 8-10. predicted_purchases_XXd
- **Source:** BG/NBD model predictions
- **Variants:** 30d, 90d, 180d forecasts
- **Definition:** Expected number of purchases in next X days
- **Interpretation:** Higher = more engaged customer

### 11. predicted_avg_value
- **Source:** Gamma-Gamma model output
- **Definition:** Expected average transaction value
- **Range:** £15 - £2,500
- **Interpretation:** Future spending level

### 12-14. CLV_XXd
- **Source:** BG/NBD × Gamma-Gamma
- **Variants:** 90d, 180d, 365d
- **Calculation:** predicted_purchases_XXd × predicted_avg_value
- **Interpretation:** Total expected future value

---

## Behavioral Velocity Features (6 features)

### 15. revenue_velocity
- **Definition:** Rate of change in spending
- **Calculation:** (Late period avg - Early period avg) / Early period avg
- **Range:** -1 to +∞
- **Interpretation:** Positive = increasing spend, Negative = declining

### 16. quantity_velocity
- **Definition:** Rate of change in items purchased
- **Calculation:** Trend in basket size over time
- **Interpretation:** Engagement trajectory

### 17. purchase_gap_velocity
- **Definition:** Acceleration/deceleration in purchase frequency
- **Calculation:** Change in days between purchases
- **Interpretation:** Negative = buying more frequently

### 18-19. early/late_period_revenue
- **Definition:** Revenue in first/second half of customer lifespan
- **Split:** Chronological midpoint
- **Use:** Component of velocity calculation

### 20. revenue_trend
- **Definition:** Binary indicator of spending direction
- **Values:** 1 = increasing, 0 = decreasing
- **Calculation:** revenue_velocity > 0

---

## Temporal Pattern Features (6 features)

### 21. day_of_week_diversity
- **Definition:** Number of unique weekdays with purchases
- **Range:** 1-7 days
- **Interpretation:** Higher = more varied shopping pattern

### 22. weekend_purchase_ratio
- **Definition:** Percentage of purchases on Saturday/Sunday
- **Range:** 0-1
- **Interpretation:** Shopping preference indicator

### 23. purchase_gap_mean
- **Definition:** Average days between consecutive purchases
- **Calculation:** Mean of all inter-purchase intervals
- **Interpretation:** Purchase frequency metric

### 24. purchase_gap_std
- **Definition:** Standard deviation of purchase gaps
- **Interpretation:** Consistency of purchase timing

### 25. purchase_gap_cv
- **Definition:** Coefficient of variation
- **Calculation:** purchase_gap_std / purchase_gap_mean
- **Interpretation:** Regularity (lower = more regular)

### 26. purchase_regularity
- **Definition:** Binary indicator of consistent timing
- **Values:** 1 = regular (CV < 0.5), 0 = irregular
- **Use:** Churn predictor (regular customers less likely to churn)

---

## Product Engagement Features (4 features)

### 27. unique_products
- **Definition:** Count of distinct products purchased
- **Calculation:** Count of unique StockCodes
- **Interpretation:** Product exploration level

### 28. avg_items_per_order
- **Definition:** Average basket depth
- **Calculation:** Total Quantity / Frequency
- **Interpretation:** Purchase volume per visit

### 29. product_diversity_ratio
- **Definition:** Exploration vs. volume ratio
- **Calculation:** unique_products / Total Quantity
- **Range:** 0-1
- **Interpretation:** Higher = more variety seeking

### 30. product_exploration_rate
- **Definition:** New product discovery rate
- **Calculation:** New products in second half vs. first half
- **Interpretation:** Engagement with product range

---

## Feature Engineering Principles

### Data Leakage Prevention
- All features calculated from **training period only** (Dec 2010)
- No information from validation period (2011) used in feature creation
- Scaler fit on training set, applied to test set

### Missing Value Handling
- Single-purchase customers: velocity features = 0 (no change)
- Zero division: handled with (+1) in denominators
- No imputation needed (complete data after filtering)

### Feature Scaling
- StandardScaler applied to all features before modeling
- Mean = 0, Std = 1 for each feature
- Prevents feature magnitude bias in tree-based models

---

## Feature Importance Rankings

**Top 5 by SHAP Importance:**
1. Monetary (0.0392)
2. predicted_purchases_180d (0.0261)
3. predicted_purchases_90d (0.0259)
4. CLV_180d (0.0238)
5. predicted_purchases_30d (0.0232)

**Category Contribution:**
- Static RFM: 69.3%
- Velocity: 30.7%

---

**Last Updated:** February 2026  
**Total Features:** 30  
**Feature Matrix:** 815 customers × 30 features
