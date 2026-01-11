"""
Utility functions for Streamlit dashboard.

Functions for loading models, data processing, predictions, and visualizations.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import shap
import matplotlib.pyplot as plt


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

def get_project_root():
    """Get project root directory."""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent


PROJECT_ROOT = get_project_root()
DIR_DATA = PROJECT_ROOT / 'data' / 'processed'
DIR_MODELS = PROJECT_ROOT / 'models'
DIR_RESULTS_OUTPUTS = PROJECT_ROOT / 'results' / 'outputs'


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models():
    """
    Load all trained models and artifacts.
    
    Returns
    -------
    dict
        Dictionary containing models, scaler, and feature names
    """
    models = {}
    
    # Load churn model
    with open(DIR_MODELS / 'best_churn_model.pkl', 'rb') as f:
        models['churn_model'] = pickle.load(f)
    
    # Load scaler
    with open(DIR_MODELS / 'feature_scaler.pkl', 'rb') as f:
        models['scaler'] = pickle.load(f)
    
    # Load feature names
    with open(DIR_MODELS / 'feature_names.json', 'r') as f:
        models['feature_names'] = json.load(f)
    
    # Load BG/NBD model
    with open(DIR_MODELS / 'bgf_model.pkl', 'rb') as f:
        models['bgf_model'] = pickle.load(f)
    
    # Load Gamma-Gamma model
    with open(DIR_MODELS / 'ggf_model.pkl', 'rb') as f:
        models['ggf_model'] = pickle.load(f)
    
    return models


def load_data():
    """
    Load customer data with predictions and segments.
    Ensures no duplicate columns are created during merging.
    """
    # Load core features
    df = pd.read_csv(DIR_DATA / 'churn_features.csv')
    
    # Load CLV predictions (Required for 'prob_alive')
    clv_preds = pd.read_csv(DIR_DATA / 'customer_clv_predictions.csv')
    
    # Load segments from results/outputs/
    segments = pd.read_csv(DIR_RESULTS_OUTPUTS / 'customer_segments_actionable.csv')
    
    # --- PREVENT DUPLICATES ---
    # Before merging, remove columns from 'df' if they exist in the incoming data
    # This prevents the "unsupported format string" Error
    cols_to_drop = [c for c in ['prob_alive', 'Segment', 'Churn_Risk', 'Predicted_CLV', 'CLV_365d', 'segment', 'churn_probability'] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # 1. Merge core features with prob_alive
    df = df.merge(
        clv_preds[['CustomerID', 'prob_alive']],
        on='CustomerID',
        how='left'
    )
    
    # 2. Merge with segments
    df = df.merge(
        segments[['CustomerID', 'Segment', 'Churn_Risk', 'Predicted_CLV']],
        on='CustomerID',
        how='left'
    )
    
    # 3. Rename for consistency
    df = df.rename(columns={
        'Segment': 'segment',
        'Churn_Risk': 'churn_probability',
        'Predicted_CLV': 'CLV_365d'
    })
    
    # Final check: Deduplicate column names if any overlap occurs
    df = df.loc[:, ~df.columns.duplicated()].copy()
    
    # Ensure AvgOrderValue exists for ROI calculator
    if 'AvgOrderValue' not in df.columns:
        df['AvgOrderValue'] = df['Monetary'] / df['Frequency'].replace(0, 1)
        
    # Fill NaNs to prevent metric formatting errors
    df['CLV_365d'] = df['CLV_365d'].fillna(0)
    df['churn_probability'] = df['churn_probability'].fillna(0)
    df['prob_alive'] = df['prob_alive'].fillna(0)
    
    return df


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_churn(customer_features, model, scaler, feature_names):
    """
    Predict churn probability for a customer.
    """
    if isinstance(customer_features, pd.Series):
        X = customer_features[feature_names].values.reshape(1, -1)
    else:
        X = np.array([customer_features[f] for f in feature_names]).reshape(1, -1)
    
    X_scaled = scaler.transform(X)
    churn_prob = model.predict_proba(X_scaled)[0, 1]
    
    return churn_prob


def calculate_clv(customer_features, bgf_model, ggf_model, time_period=365):
    """
    Calculate customer lifetime value.
    """
    frequency = customer_features['Frequency']
    recency = customer_features['Recency']
    T = customer_features['Tenure']
    monetary = customer_features['Monetary']
    
    predicted_purchases = bgf_model.predict(
        time_period,
        frequency,
        recency,
        T
    )
    
    if frequency > 0:
        predicted_value = ggf_model.conditional_expected_average_profit(
            frequency,
            monetary / frequency
        )
    else:
        predicted_value = monetary
    
    clv = predicted_purchases * predicted_value
    
    return clv


def get_customer_segment(churn_prob, clv, value_threshold=6540, risk_threshold=0.5):
    """
    Assign customer to risk-value segment.
    """
    if clv >= value_threshold and churn_prob >= risk_threshold:
        return 'Save At All Costs'
    elif clv >= value_threshold and churn_prob < risk_threshold:
        return 'Loyalists'
    elif clv < value_threshold and churn_prob >= risk_threshold:
        return 'Let Go'
    else:
        return 'Steady Base'


# ============================================================================
# ROI CALCULATIONS
# ============================================================================

def calculate_roi(target_customers, avg_clv, discount_rate, avg_order_value,
                 expected_orders, campaign_cost):
    """
    Calculate ROI for retention campaign.
    """
    discount_cost = avg_order_value * expected_orders * discount_rate
    cost_per_customer = discount_cost + campaign_cost
    total_budget = cost_per_customer * target_customers
    
    value_per_saved = avg_clv - discount_cost
    
    # Break-even rate calculation
    if value_per_saved <= 0:
        breakeven_rate = 1.0
    else:
        breakeven_rate = cost_per_customer / value_per_saved
        
    breakeven_customers = target_customers * breakeven_rate
    
    return {
        'cost_per_customer': cost_per_customer,
        'value_per_saved': value_per_saved,
        'total_budget': total_budget,
        'breakeven_rate': breakeven_rate,
        'breakeven_customers': breakeven_customers
    }


# ============================================================================
# SHAP EXPLANATIONS
# ============================================================================

def create_shap_explanation(customer_data, model, scaler, feature_names, top_n=10):
    """
    Create SHAP waterfall plot for individual customer.
    """
    # Extract and scale features
    X = customer_data[feature_names].values.reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    
    # Support for different SHAP output formats (Classifier vs Regressor)
    if isinstance(shap_values, list):
        shap_values_churn = shap_values[1][0]
    else:
        shap_values_churn = shap_values[0] if len(shap_values.shape) > 1 else shap_values
    
    # Create DataFrame for plotting
    feature_impact = pd.DataFrame({
        'feature': feature_names,
        'value': X[0],
        'impact': shap_values_churn
    }).sort_values('impact', key=abs, ascending=False).head(top_n)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#ff4b4b' if impact > 0 else '#00cc66' for impact in feature_impact['impact']]
    
    ax.barh(range(len(feature_impact)), feature_impact['impact'], color=colors, alpha=0.8)
    ax.set_yticks(range(len(feature_impact)))
    ax.set_yticklabels(feature_impact['feature'])
    ax.set_xlabel('SHAP Value (Impact on Churn Prediction)', fontweight='bold')
    ax.set_title(f'Top {top_n} Features Driving Churn Prediction', fontweight='bold', fontsize=14)
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff4b4b', label='Increases churn risk'),
        Patch(facecolor='#00cc66', label='Decreases churn risk')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    return fig