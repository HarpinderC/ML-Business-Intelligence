"""
Configuration file for Streamlit dashboard.

Contains constants, thresholds, and styling configurations.
"""

# ============================================================================
# SEGMENT COLORS
# ============================================================================

SEGMENT_COLORS = {
    'Save At All Costs': '#FF4444',  # Red
    'Loyalists': '#00CC66',          # Green
    'Steady Base': '#2196F3',        # Blue
    'Let Go': '#999999'              # Gray
}


# ============================================================================
# RISK THRESHOLDS
# ============================================================================

RISK_THRESHOLDS = {
    'high': 0.70,      # 70% churn probability
    'medium': 0.40,    # 40% churn probability
    'low': 0.00        # Below 40%
}


# ============================================================================
# DEFAULT CAMPAIGN PARAMETERS
# ============================================================================

DEFAULT_CAMPAIGN_PARAMS = {
    'discount_rate': 20,        # 20% discount
    'expected_orders': 3,       # Expected orders if retained
    'campaign_cost': 5.00,      # £5 per customer campaign cost
    'avg_order_value': 95.00    # Average order value
}


# ============================================================================
# CLV THRESHOLDS
# ============================================================================

CLV_THRESHOLDS = {
    'high_value': 6540,   # Top 25% CLV threshold
    'medium_value': 3000,
    'low_value': 1000
}


# ============================================================================
# FEATURE CATEGORIES
# ============================================================================

FEATURE_CATEGORIES = {
    'rfm': [
        'Recency', 'Frequency', 'Monetary', 
        'Tenure', 'AvgOrderValue', 'AvgBasketSize'
    ],
    'probabilistic': [
        'prob_alive', 'predicted_purchases_30d', 'predicted_purchases_90d',
        'predicted_purchases_180d', 'predicted_avg_value',
        'CLV_90d', 'CLV_180d', 'CLV_365d'
    ],
    'velocity': [
        'revenue_velocity', 'quantity_velocity', 'purchase_gap_velocity',
        'early_period_revenue', 'late_period_revenue', 'revenue_trend'
    ],
    'temporal': [
        'day_of_week_diversity', 'weekend_purchase_ratio',
        'purchase_gap_mean', 'purchase_gap_std', 
        'purchase_gap_cv', 'purchase_regularity'
    ],
    'engagement': [
        'unique_products', 'avg_items_per_order',
        'product_diversity_ratio', 'product_exploration_rate'
    ]
}


# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

PLOT_CONFIG = {
    'height': 400,
    'template': 'plotly_white',
    'font_family': 'Segoe UI, sans-serif',
    'font_size': 12,
    'title_font_size': 16
}


# ============================================================================
# APP SETTINGS
# ============================================================================

APP_CONFIG = {
    'page_title': 'Customer Retention Analytics',
    'page_icon': '📊',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}


# ============================================================================
# MODEL INFORMATION
# ============================================================================

MODEL_INFO = {
    'churn_model': {
        'name': 'Random Forest',
        'auc': 0.64,
        'description': 'Diagnostic tool for behavioral pattern identification'
    },
    'clv_model': {
        'name': 'BG/NBD + Gamma-Gamma',
        'correlation': 0.699,
        'description': 'Probabilistic customer lifetime value forecasting'
    }
}
