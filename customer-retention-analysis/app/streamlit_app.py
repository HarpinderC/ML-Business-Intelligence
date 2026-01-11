"""
Customer Segmentation & Retention Analysis - Streamlit Dashboard

A production-ready interactive dashboard for customer churn prediction,
segmentation analysis, and retention ROI calculation.

Author: Harpinder Singh Chhabra
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import shap
import matplotlib.pyplot as plt
from io import BytesIO

# Import utilities
from utils import (
    load_models,
    load_data,
    predict_churn,
    calculate_clv,
    get_customer_segment,
    calculate_roi,
    create_shap_explanation
)
from config import (
    SEGMENT_COLORS,
    RISK_THRESHOLDS,
    DEFAULT_CAMPAIGN_PARAMS
)

# Page configuration
st.set_page_config(
    page_title="Customer Retention Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high {
        color: #ff4b4b;
        font-weight: 700;
    }
    .risk-medium {
        color: #ffa500;
        font-weight: 700;
    }
    .risk-low {
        color: #00cc66;
        font-weight: 700;
    }
    .segment-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_all_models():
    """Load all trained models and artifacts."""
    return load_models()


@st.cache_data
def load_customer_data():
    """Load customer features and segments."""
    return load_data()


def display_customer_card(customer_data, predictions):
    """Display customer information card."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Customer ID", customer_data['CustomerID'])
        st.metric("Segment", predictions['segment'])
    
    with col2:
        churn_prob = predictions['churn_probability']
        risk_level = "HIGH" if churn_prob > RISK_THRESHOLDS['high'] else \
                     "MEDIUM" if churn_prob > RISK_THRESHOLDS['medium'] else "LOW"
        risk_class = f"risk-{risk_level.lower()}"
        
        st.metric("Churn Risk", f"{churn_prob:.1%}")
        st.markdown(f"<p class='{risk_class}'>Risk Level: {risk_level}</p>", 
                   unsafe_allow_html=True)
    
    with col3:
        st.metric("Predicted CLV (365d)", f"£{predictions['clv_365d']:,.2f}")
        st.metric("Alive Probability", f"{predictions['prob_alive']:.1%}")
    
    with col4:
        st.metric("Historical Revenue", f"£{customer_data['Monetary']:,.2f}")
        st.metric("Purchase Frequency", f"{int(customer_data['Frequency'])}")


def create_gauge_chart(value, title, max_value=1.0):
    """Create a gauge chart for probability visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={'text': title, 'font': {'size': 20}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#ff4b4b" if value > 0.7 else "#ffa500" if value > 0.4 else "#00cc66"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#e8f5e9'},
                {'range': [40, 70], 'color': '#fff3e0'},
                {'range': [70, 100], 'color': '#ffebee'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_segment_distribution(df):
    """Create segment distribution visualization."""
    segment_counts = df['segment'].value_counts()
    
    fig = go.Figure(data=[
        go.Bar(
            x=segment_counts.index,
            y=segment_counts.values,
            marker_color=[SEGMENT_COLORS.get(seg, '#cccccc') for seg in segment_counts.index],
            text=segment_counts.values,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Customer Segment Distribution",
        xaxis_title="Segment",
        yaxis_title="Number of Customers",
        height=400,
        showlegend=False
    )
    
    return fig


def create_clv_vs_risk_scatter(df):
    """Create CLV vs Risk scatter plot."""
    fig = px.scatter(
        df,
        x='CLV_365d',
        y='churn_probability',
        color='segment',
        size='Monetary',
        hover_data=['CustomerID', 'Frequency', 'Recency'],
        color_discrete_map=SEGMENT_COLORS,
        title="Risk-Value Matrix"
    )
    
    # Add threshold lines
    fig.add_hline(y=RISK_THRESHOLDS['high'], line_dash="dash", 
                  line_color="red", annotation_text="High Risk Threshold")
    
    fig.update_layout(
        xaxis_title="Predicted 365-Day CLV (£)",
        yaxis_title="Churn Probability",
        height=500
    )
    
    return fig


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application logic."""
    
    # Load models and data
    models = load_all_models()
    df = load_customer_data()
    
    # Sidebar
    st.sidebar.markdown("# 📊 Navigation")
    page = st.sidebar.radio(
        "Select Page:",
        ["🏠 Home", "🔍 Customer Lookup", "📈 Segment Dashboard", 
         "💰 ROI Calculator", "📊 Feature Importance", "📤 Batch Scoring"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "Customer retention analytics powered by probabilistic CLV modeling "
        "and machine learning. Identify at-risk customers and optimize "
        "retention investments."
    )
    
    # ========================================================================
    # PAGE: HOME
    # ========================================================================
    if page == "🏠 Home":
        st.markdown("<h1 class='main-header'>Customer Retention Analytics</h1>", 
                   unsafe_allow_html=True)
        st.markdown("### Predict churn, segment customers, and calculate retention ROI")
        
        st.markdown("---")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_customers = len(df)
            st.metric("Total Customers", f"{total_customers:,}")
        
        with col2:
            high_risk = (df['churn_probability'] > RISK_THRESHOLDS['high']).sum()
            st.metric("High Risk Customers", f"{high_risk:,}", 
                     delta=f"{high_risk/total_customers*100:.1f}%")
        
        with col3:
            avg_clv = df['CLV_365d'].mean()
            st.metric("Average CLV (365d)", f"£{avg_clv:,.0f}")
        
        with col4:
            save_segment = (df['segment'] == 'Save At All Costs').sum()
            st.metric("Save At All Costs", f"{save_segment:,}",
                     delta=f"{save_segment/total_customers*100:.1f}%")
        
        st.markdown("---")
        
        # Segment overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_segment_distribution(df), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_clv_vs_risk_scatter(df), use_container_width=True)
        
        # Quick insights
        st.markdown("### 💡 Key Insights")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.markdown("""
            **High-Priority Actions:**
            - Target **178 "Save At All Costs"** customers (avg CLV £6,540)
            - Break-even retention rate: **0.6%** (extremely favorable)
            - Projected ROI at 20% retention: **£225K+**
            """)
        
        with insights_col2:
            st.markdown("""
            **Model Performance:**
            - Probabilistic CLV: **0.699 correlation** with actual behavior
            - Churn classifier: **0.64 AUC** (diagnostic tool)
            - Top predictor: **Monetary** (historical spend matters most)
            """)
    
    # ========================================================================
    # PAGE: CUSTOMER LOOKUP
    # ========================================================================
    elif page == "🔍 Customer Lookup":
        st.markdown("<h1 class='main-header'>Customer Lookup</h1>", 
                   unsafe_allow_html=True)
        st.markdown("### Search for individual customer predictions and explanations")
        
        # Customer ID input
        customer_id = st.number_input(
            "Enter Customer ID:",
            min_value=int(df['CustomerID'].min()),
            max_value=int(df['CustomerID'].max()),
            value=int(df['CustomerID'].iloc[0]),
            step=1
        )
        
        if st.button("🔍 Lookup Customer", type="primary"):
            # Get customer data
            customer_data = df[df['CustomerID'] == customer_id]
            
            if len(customer_data) == 0:
                st.error(f"Customer ID {customer_id} not found in database.")
                return
            
            customer_data = customer_data.iloc[0]
            
            # Get predictions
            predictions = {
                'churn_probability': customer_data['churn_probability'],
                'clv_365d': customer_data['CLV_365d'],
                'prob_alive': customer_data['prob_alive'],
                'segment': customer_data['segment']
            }
            
            st.markdown("---")
            
            # Display customer card
            display_customer_card(customer_data, predictions)
            
            st.markdown("---")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(
                    create_gauge_chart(predictions['churn_probability'], "Churn Risk"),
                    use_container_width=True
                )
            
            with col2:
                st.plotly_chart(
                    create_gauge_chart(predictions['prob_alive'], "Alive Probability"),
                    use_container_width=True
                )
            
            # SHAP Explanation
            st.markdown("### 🔬 SHAP Explanation")
            st.markdown("**Why is this customer at risk?** (Top feature impacts)")
            
            try:
                # Create SHAP explanation
                shap_fig = create_shap_explanation(
                    customer_data,
                    models['churn_model'],
                    models['scaler'],
                    models['feature_names']
                )
                
                st.pyplot(shap_fig)
                
                st.markdown("""
                **How to read this chart:**
                - **Red bars** push prediction toward churn (increase risk)
                - **Blue bars** push prediction away from churn (decrease risk)
                - **Bar length** shows feature impact strength
                """)
            except Exception as e:
                st.warning(f"SHAP explanation not available: {str(e)}")
            
            # Recommendations
            st.markdown("### 💡 Recommended Actions")
            
            if predictions['segment'] == 'Save At All Costs':
                st.error("""
                **🚨 HIGH PRIORITY - Immediate Intervention Required**
                - Offer: 20% personalized discount
                - Communication: Personal email from account manager
                - Timeline: Contact within 48 hours
                - Expected value if retained: £{:,.2f}
                """.format(predictions['clv_365d']))
            
            elif predictions['segment'] == 'Loyalists':
                st.success("""
                **✅ VIP Retention Program**
                - Maintain regular engagement
                - Exclusive early access to new products
                - Loyalty rewards program
                - Expected lifetime value: £{:,.2f}
                """.format(predictions['clv_365d']))
            
            elif predictions['segment'] == 'Let Go':
                st.info("""
                **📧 Automated Win-Back Campaign**
                - Low-cost email automation
                - Generic discount offers
                - Minimal resource allocation
                - Expected value: £{:,.2f}
                """.format(predictions['clv_365d']))
            
            else:  # Steady Base
                st.info("""
                **📊 Standard Service**
                - Regular marketing communications
                - Standard customer service
                - Monitor for changes in behavior
                """)
    
    # ========================================================================
    # PAGE: SEGMENT DASHBOARD
    # ========================================================================
    elif page == "📈 Segment Dashboard":
        st.markdown("<h1 class='main-header'>Segment Dashboard</h1>", 
                   unsafe_allow_html=True)
        st.markdown("### Risk-Value Matrix and segment analytics")
        
        # Segment summary
        segment_summary = df.groupby('segment').agg({
            'CustomerID': 'count',
            'CLV_365d': 'mean',
            'churn_probability': 'mean',
            'Monetary': 'mean'
        }).round(2)
        
        segment_summary.columns = ['Count', 'Avg CLV (£)', 'Avg Churn Risk', 'Avg Historical Revenue (£)']
        segment_summary['% of Base'] = (segment_summary['Count'] / len(df) * 100).round(1)
        
        # Reorder columns
        segment_summary = segment_summary[['Count', '% of Base', 'Avg CLV (£)', 
                                          'Avg Churn Risk', 'Avg Historical Revenue (£)']]
        
        st.markdown("### 📊 Segment Summary")
        st.dataframe(segment_summary, use_container_width=True)
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_segment_distribution(df), use_container_width=True)
        
        with col2:
            # CLV by segment
            fig = px.box(
                df,
                x='segment',
                y='CLV_365d',
                color='segment',
                color_discrete_map=SEGMENT_COLORS,
                title="CLV Distribution by Segment"
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Risk-Value scatter
        st.plotly_chart(create_clv_vs_risk_scatter(df), use_container_width=True)
        
        # Segment details
        st.markdown("### 🎯 Segment Strategies")
        
        segments_info = {
            'Save At All Costs': {
                'icon': '🚨',
                'color': '#ff4b4b',
                'strategy': 'Immediate personalized intervention with 20% discount offer',
                'priority': 'CRITICAL'
            },
            'Loyalists': {
                'icon': '👑',
                'color': '#00cc66',
                'strategy': 'VIP program with exclusive benefits and proactive engagement',
                'priority': 'HIGH'
            },
            'Steady Base': {
                'icon': '📊',
                'color': '#2196f3',
                'strategy': 'Standard service with regular communications',
                'priority': 'MEDIUM'
            },
            'Let Go': {
                'icon': '📧',
                'color': '#999999',
                'strategy': 'Automated low-cost win-back campaigns only',
                'priority': 'LOW'
            }
        }
        
        for segment, info in segments_info.items():
            segment_data = df[df['segment'] == segment]
            if len(segment_data) > 0:
                with st.expander(f"{info['icon']} {segment} ({len(segment_data)} customers)"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Value", f"£{segment_data['CLV_365d'].sum():,.0f}")
                    with col2:
                        st.metric("Avg Risk", f"{segment_data['churn_probability'].mean():.1%}")
                    with col3:
                        st.metric("Priority", info['priority'])
                    
                    st.markdown(f"**Strategy:** {info['strategy']}")
    
    # ========================================================================
    # PAGE: ROI CALCULATOR
    # ========================================================================
    elif page == "💰 ROI Calculator":
        st.markdown("<h1 class='main-header'>ROI Calculator</h1>", 
                   unsafe_allow_html=True)
        st.markdown("### Calculate break-even and ROI for retention campaigns")
        
        # Target segment selection
        segment_options = ['Save At All Costs', 'Loyalists', 'All High-Value', 'Custom']
        selected_segment = st.selectbox("Target Segment:", segment_options)
        
        if selected_segment == 'Save At All Costs':
            target_df = df[df['segment'] == 'Save At All Costs']
        elif selected_segment == 'Loyalists':
            target_df = df[df['segment'] == 'Loyalists']
        elif selected_segment == 'All High-Value':
            target_df = df[df['segment'].isin(['Save At All Costs', 'Loyalists'])]
        else:  # Custom
            min_clv = st.slider("Minimum CLV (£):", 0, 30000, 5000, 500)
            target_df = df[df['CLV_365d'] >= min_clv]
        
        st.info(f"**Target Customers:** {len(target_df):,} | **Avg CLV:** £{target_df['CLV_365d'].mean():,.2f}")
        
        st.markdown("---")
        
        # Campaign parameters
        st.markdown("### 📋 Campaign Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            discount_rate = st.slider(
                "Discount Percentage:",
                min_value=0,
                max_value=50,
                value=DEFAULT_CAMPAIGN_PARAMS['discount_rate'],
                step=5,
                format="%d%%"
            ) / 100
            
            expected_orders = st.number_input(
                "Expected Orders (if retained):",
                min_value=1,
                max_value=10,
                value=DEFAULT_CAMPAIGN_PARAMS['expected_orders'],
                step=1
            )
        
        with col2:
            avg_order_value = st.number_input(
                "Average Order Value (£):",
                min_value=0.0,
                max_value=1000.0,
                value=float(target_df['AvgOrderValue'].mean()),
                step=10.0,
                format="%.2f"
            )
            
            campaign_cost = st.number_input(
                "Campaign Cost per Customer (£):",
                min_value=0.0,
                max_value=50.0,
                value=DEFAULT_CAMPAIGN_PARAMS['campaign_cost'],
                step=1.0,
                format="%.2f"
            )
        
        # Calculate ROI
        roi_results = calculate_roi(
            target_customers=len(target_df),
            avg_clv=target_df['CLV_365d'].mean(),
            discount_rate=discount_rate,
            avg_order_value=avg_order_value,
            expected_orders=expected_orders,
            campaign_cost=campaign_cost
        )
        
        st.markdown("---")
        
        # Results
        st.markdown("### 💵 Financial Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Cost per Customer",
                f"£{roi_results['cost_per_customer']:.2f}",
                help="Discount cost + Campaign execution cost"
            )
        
        with col2:
            st.metric(
                "Value per Saved Customer",
                f"£{roi_results['value_per_saved']:.2f}",
                help="CLV minus discount cost"
            )
        
        with col3:
            st.metric(
                "Total Campaign Budget",
                f"£{roi_results['total_budget']:,.2f}"
            )
        
        # Break-even
        st.markdown("### 🎯 Break-Even Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Break-Even Retention Rate",
                f"{roi_results['breakeven_rate']:.2%}",
                help="Minimum retention rate to cover campaign costs"
            )
        
        with col2:
            st.metric(
                "Customers Needed to Save",
                f"{int(roi_results['breakeven_customers'])}",
                help=f"Out of {len(target_df)} targeted"
            )
        
        # ROI scenarios
        st.markdown("### 📊 ROI Scenarios")
        
        scenarios = []
        for rate in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
            customers_saved = len(target_df) * rate
            revenue_saved = customers_saved * roi_results['value_per_saved']
            net_roi = revenue_saved - roi_results['total_budget']
            roi_pct = (net_roi / roi_results['total_budget']) * 100
            
            scenarios.append({
                'Retention Rate': f"{rate:.0%}",
                'Customers Saved': int(customers_saved),
                'Revenue Saved': f"£{revenue_saved:,.0f}",
                'Net ROI': f"£{net_roi:,.0f}",
                'ROI %': f"{roi_pct:.0f}%",
                'Status': '✅ Profitable' if net_roi > 0 else '❌ Loss'
            })
        
        scenarios_df = pd.DataFrame(scenarios)
        st.dataframe(scenarios_df, use_container_width=True)
        
        # Visualization
        retention_range = np.linspace(0, 0.70, 100)
        revenue_saved = retention_range * len(target_df) * roi_results['value_per_saved']
        net_roi_curve = revenue_saved - roi_results['total_budget']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=retention_range * 100,
            y=net_roi_curve,
            mode='lines',
            name='Net ROI',
            line=dict(color='#2196f3', width=3)
        ))
        
        fig.add_hline(
            y=0,
            line_dash="solid",
            line_color="black",
            line_width=2
        )
        
        fig.add_vline(
            x=roi_results['breakeven_rate'] * 100,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Break-even: {roi_results['breakeven_rate']:.1%}"
        )
        
        fig.update_layout(
            title="Campaign ROI by Retention Rate",
            xaxis_title="Retention Rate (%)",
            yaxis_title="Net ROI (£)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # PAGE: FEATURE IMPORTANCE
    # ========================================================================
    elif page == "📊 Feature Importance":
        st.markdown("<h1 class='main-header'>Feature Importance</h1>", 
                   unsafe_allow_html=True)
        st.markdown("### Global SHAP analysis and churn drivers")
        
        shap_importance = pd.read_csv(
            Path(__file__).parent.parent / 'results' / 'outputs' / 'shap_feature_importance.csv'
        )
        
        st.markdown("### 🎯 Top 15 Churn Drivers (SHAP Values)")
        
        
        top_15 = shap_importance.head(15)
        
        fig = go.Figure(go.Bar(
            y=top_15['feature'],
            x=top_15['importance'],
            orientation='h',
            marker_color='#1f77b4'
        ))
        
        fig.update_layout(
            xaxis_title="SHAP Importance",
            yaxis_title="",
            height=600,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Velocity vs Static comparison
        st.markdown("### ⚡ Velocity vs. Static Features")
        
        velocity_features = ['revenue_velocity', 'quantity_velocity', 'purchase_gap_velocity',
                            'early_period_revenue', 'late_period_revenue', 'revenue_trend']
        static_features = ['Recency', 'Frequency', 'Monetary', 'Tenure', 
                          'AvgOrderValue', 'AvgBasketSize']
        
        velocity_importance = shap_importance[
            shap_importance['feature'].isin(velocity_features)
        ]['importance'].sum()
        
        static_importance = shap_importance[
            shap_importance['feature'].isin(static_features)
        ]['importance'].sum()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Velocity Features",
                f"{velocity_importance:.4f}",
                delta=f"{velocity_importance/(velocity_importance+static_importance)*100:.1f}%"
            )
        
        with col2:
            st.metric(
                "Static RFM",
                f"{static_importance:.4f}",
                delta=f"{static_importance/(velocity_importance+static_importance)*100:.1f}%"
            )
        
        # Interpretation
        st.info("""
        **Key Insight:** Historical RFM metrics (69%) remain more predictive than 
        behavioral velocity (31%), but velocity features add valuable signals about 
        customer engagement trends.
        """)
        
        # Feature definitions
        with st.expander("📖 Feature Definitions"):
            st.markdown("""
            **RFM Features:**
            - **Recency:** Days since last purchase
            - **Frequency:** Total number of purchases
            - **Monetary:** Total revenue generated
            - **Tenure:** Days since first purchase
            
            **Probabilistic Features:**
            - **prob_alive:** BG/NBD probability customer is still active
            - **predicted_purchases_XXd:** Expected purchases in next XX days
            - **CLV_XXd:** Predicted customer lifetime value over XX days
            
            **Velocity Features:**
            - **revenue_velocity:** Rate of change in spending
            - **quantity_velocity:** Rate of change in items purchased
            - **purchase_gap_velocity:** Acceleration/deceleration of purchase frequency
            """)
    
    # ========================================================================
    # PAGE: BATCH SCORING
    # ========================================================================
    elif page == "📤 Batch Scoring":
        st.markdown("<h1 class='main-header'>Batch Scoring</h1>", 
                   unsafe_allow_html=True)
        st.markdown("### Upload CSV for bulk churn predictions")
        
        st.markdown("""
        Upload a CSV file with customer features to get churn predictions for multiple customers.
        
        **Required columns:** All 30 features from the training set.
        """)
        
        # Sample CSV download
        sample_df = df[models['feature_names']].head(5)
        csv_buffer = BytesIO()
        sample_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=csv_buffer,
            file_name="sample_customers.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV file:", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Read uploaded file
                upload_df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ File uploaded successfully! {len(upload_df)} customers found.")
                
                # Validate columns
                missing_cols = set(models['feature_names']) - set(upload_df.columns)
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                    return
                
                # Predict
                if st.button("🚀 Generate Predictions", type="primary"):
                    with st.spinner("Generating predictions..."):
                        # Scale features
                        X = upload_df[models['feature_names']]
                        X_scaled = models['scaler'].transform(X)
                        
                        # Predict
                        predictions = models['churn_model'].predict_proba(X_scaled)[:, 1]
                        
                        # Add to dataframe
                        results_df = upload_df.copy()
                        results_df['churn_probability'] = predictions
                        results_df['churn_risk'] = pd.cut(
                            predictions,
                            bins=[0, 0.4, 0.7, 1.0],
                            labels=['Low', 'Medium', 'High']
                        )
                        
                        # Display results
                        st.markdown("### 📊 Prediction Results")
                        st.dataframe(
                            results_df[['CustomerID', 'churn_probability', 'churn_risk']].head(20),
                            use_container_width=True
                        )
                        
                        # Summary
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            high_risk = (predictions > RISK_THRESHOLDS['high']).sum()
                            st.metric("High Risk", high_risk, 
                                     delta=f"{high_risk/len(predictions)*100:.1f}%")
                        
                        with col2:
                            medium_risk = ((predictions > RISK_THRESHOLDS['medium']) & 
                                         (predictions <= RISK_THRESHOLDS['high'])).sum()
                            st.metric("Medium Risk", medium_risk,
                                     delta=f"{medium_risk/len(predictions)*100:.1f}%")
                        
                        with col3:
                                    # .item() converts a 1-item series/array into a standard python float
                                    avg_clv_val = df['CLV_365d'].mean()
                                    if hasattr(avg_clv_val, 'item'):
                                        avg_clv_val = avg_clv_val.item()
                                    st.metric("Average CLV (365d)", f"£{avg_clv_val:,.0f}")
                        
                        # Download results
                        results_buffer = BytesIO()
                        results_df.to_csv(results_buffer, index=False)
                        results_buffer.seek(0)
                        
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=results_buffer,
                            file_name="churn_predictions.csv",
                            mime="text/csv"
                        )
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")


# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()