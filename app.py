from xml.parsers.expat import model
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="House Price Forecasting System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Custom CSS with animations and colors
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    [data-testid="stMainBlockContainer"] {
        padding-top: 2rem;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        animation: slideDown 0.6s ease-out;
    }
    
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        animation: fadeIn 0.6s ease-out;
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        animation: slideUp 0.6s ease-out;
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .hpi-box-positive {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.8rem;
        border-radius: 1.2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(17, 153, 142, 0.3);
        border: 2px solid rgba(56, 239, 125, 0.5);
        animation: popIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    
    .hpi-box-negative {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 1.8rem;
        border-radius: 1.2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(235, 51, 73, 0.3);
        border: 2px solid rgba(244, 92, 67, 0.5);
        animation: popIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    
    @keyframes popIn {
        0% {
            opacity: 0;
            transform: scale(0.9);
        }
        50% {
            transform: scale(1.05);
        }
        100% {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    .info-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .success-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 5px solid #11998e;
        margin: 1rem 0;
    }
    
    .input-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        border-top: 3px solid #667eea;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #667eea;
        margin: 1.5rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        border-radius: 0.8rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) !important;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.95rem;
        opacity: 0.95;
        font-weight: 500;
    }
    
    .caption-text {
        color: #667eea;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.3rem;
    }
    
    .field-guide-title {
        color: #667eea;
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 1.5rem;
    }
    
    .expandable-header {
        color: #764ba2;
        font-weight: 600;
        font-size: 1rem;
    }
    
    .hpi-score {
        font-size: 3rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    
    .trend-indicator {
        font-size: 2rem;
        margin: 0.5rem 0;
    }
    
    .divider {
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        height: 2px;
        margin: 2rem 0;
        border: none;
    }
    
    [data-testid="stExpander"] {
        border: 1px solid #e0e0e0 !important;
        border-radius: 0.8rem !important;
    }
    
    .stExpander {
        background-color: #f9f9f9;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        model = joblib.load("lgb_model.pkl") 
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"⚠️ Error loading models: {e}")
        return None, None

def create_features(df):
    """Feature engineering"""
    df = df.copy()
    
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year

    df['sale_list_gap'] = df['median_sale_price'] - df['median_list_price']
    df['ppsf_gap'] = df['median_ppsf'] - df['median_list_ppsf']

    df['sales_inventory_ratio'] = df['homes_sold'] / (df['inventory'] + 1)
    df['new_listing_pressure'] = df['new_listings'] / (df['inventory'] + 1)
    
    df['market_heat'] = (df['sold_above_list'] + df['off_market_in_two_weeks']) / 2
    df['dom_inverse'] = 1 / (df['median_dom'] + 1)

    amenities = ['bank', 'bus', 'hospital', 'mall', 'park', 'restaurant', 'school', 'station', 'supermarket']
    df['amenities_score'] = df[amenities].sum(axis=1)

    df['population_density'] = df['Total Population'] / (df['Total Housing Units'] + 1)
    df['school_enrollment_rate'] = df['Total School Enrollment'] / (df['Total School Age Population'] + 1)

    df['unemployment_rate'] = df['Unemployed Population'] / (df['Total Labor Force'] + 1)
    df['poverty_rate'] = df['Total Families Below Poverty'] / (df['Total Housing Units'] + 1)
    df['rent_price_ratio'] = df['Median Rent'] / (df['Median Home Value'] + 1)
    
    return df

def calculate_hpi(current_price, previous_price, base_index=100):
    """Calculate House Price Index"""
    
    if previous_price is None or previous_price == 0:
        return None
    
    hpi = (current_price / previous_price) * base_index
    price_change_pct = ((current_price - previous_price) / previous_price) * 100
    price_change_abs = current_price - previous_price
    
    return {
        'hpi': hpi,
        'price_change_pct': price_change_pct,
        'price_change_abs': price_change_abs,
        'is_increase': price_change_pct > 0
    }

def display_hpi_result(hpi_data):
    """Display HPI results with styling"""
    if not hpi_data:
        return
    
    is_increase = hpi_data['is_increase']
    box_class = "hpi-box-positive" if is_increase else "hpi-box-negative"
    emoji = "📈" if is_increase else "📉"
    trend_text = "Market Appreciation" if is_increase else "Market Depreciation"
    
    st.markdown(f"""
    <div class="{box_class}">
        <h3>{emoji} House Price Index (HPI) Analysis</h3>
        <div class="hpi-score">{hpi_data['hpi']:.2f}</div>
        <p style="font-size: 1.2rem; margin: 0.5rem 0;">
            Price Change: <strong>{hpi_data['price_change_pct']:+.2f}%</strong>
        </p>
        <p style="font-size: 1rem; margin: 0.5rem 0;">
            Absolute Change: <strong>${hpi_data['price_change_abs']:+,.0f}</strong>
        </p>
        <p style="font-size: 1.1rem; margin-top: 1rem; font-weight: 600;">
            {trend_text}
        </p>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.markdown('<div class="main-header">🏠 House Price Forecasting System</div>', unsafe_allow_html=True)
    
    model, scaler = load_models()
    
    if model is None or scaler is None:
        st.stop()

    menu = ["Single Prediction", "Field Guide", "Batch Prediction", "HPI Dashboard"]
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice == "Single Prediction":
        st.markdown('<div class="section-title">📊 Property Market Analysis</div>', unsafe_allow_html=True)
        
        with st.expander("📖 How to Use This Tool", expanded=False):
            st.markdown("""
            1. **Enter current market data** - prices, inventory, and market metrics
            2. **Input area demographics** - population and income data
            3. **Set previous price** - to calculate HPI trend
            4. **Click Predict** - get the forecasted price
            
            **Quick Tips:**
            - Use real data from your local real estate market
            - Previous price should be from 3-6 months ago
            - Inventory and sales data from MLS reports
            """)
        
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        
        st.markdown('<div class="section-title">💰 Market Metrics</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            st.markdown('<p class="caption-text">📅 When is this market data from?</p>', unsafe_allow_html=True)
            date = st.date_input("Analysis Date", datetime.now())
            
            st.markdown('<p class="caption-text">💰 Recent homes sold around this price</p>', unsafe_allow_html=True)
            median_sale_price = st.number_input(
                "Current Median Sale Price ($)",
                value=400000,
                step=10000,
                help="The middle price point of recently sold homes"
            )
            
            st.markdown('<p class="caption-text">📌 Sellers are listing homes at this price</p>', unsafe_allow_html=True)
            median_list_price = st.number_input(
                "Median List Price ($)",
                value=395000,
                step=10000,
                help="The middle asking price of homes currently on market"
            )
            
            st.markdown('<p class="caption-text">✅ Transaction volume indicator</p>', unsafe_allow_html=True)
            homes_sold = st.number_input(
                "Homes Sold (past month)",
                value=50,
                step=1,
                help="Number of homes that sold in the past month"
            )
            
            st.markdown('<p class="caption-text">📊 Supply available in the market</p>', unsafe_allow_html=True)
            inventory = st.number_input(
                "Active Inventory",
                value=200,
                step=10,
                help="Number of homes currently for sale"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            st.markdown('<p class="caption-text">⏱️ How long homes take to sell</p>', unsafe_allow_html=True)
            median_dom = st.number_input(
                "Median Days on Market",
                value=30,
                step=1,
                help="Average time a home sits before selling"
            )
            
            st.markdown('<p class="caption-text">📐 Price normalized by size</p>', unsafe_allow_html=True)
            median_ppsf = st.number_input(
                "Median Price Per Sq Ft ($)",
                value=250,
                step=10,
                help="Cost per square foot for homes in area"
            )
            
            st.markdown('<p class="caption-text">📐 What sellers are asking per sq ft</p>', unsafe_allow_html=True)
            median_list_ppsf = st.number_input(
                "Median List Price Per Sq Ft ($)",
                value=245,
                step=10,
                help="Asking price per square foot"
            )
            
            st.markdown('<p class="caption-text">🔥 Sign of a hot market</p>', unsafe_allow_html=True)
            sold_above_list = st.number_input(
                "Homes Sold Above List Price",
                value=10,
                step=1,
                help="How many homes sold for MORE than asking"
            )
            
            st.markdown('<p class="caption-text">⚡ Market speed indicator</p>', unsafe_allow_html=True)
            off_market_in_two_weeks = st.number_input(
                "Homes Off Market < 2 Weeks",
                value=15,
                step=1,
                help="Homes off market within 14 days"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">👥 Area Demographics</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            pop = st.number_input("Total Population", value=50000, step=1000)
            st.markdown('<p class="caption-text">👥 How many people live here</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            units = st.number_input("Total Housing Units", value=20000, step=500)
            st.markdown('<p class="caption-text">🏘️ Total housing stock</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            income = st.number_input("Per Capita Income ($)", value=45000, step=1000)
            st.markdown('<p class="caption-text">💵 Average purchasing power</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📈 Historical Comparison</div>', unsafe_allow_html=True)
        
        prev_col1, prev_col2 = st.columns(2)
        with prev_col1:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            avg_sale_to_list = st.number_input(
                "Average Sale to List Ratio",
                value=0.98,
                step=0.01,
                min_value=0.5,
                max_value=1.5,
                help="Sales Price ÷ List Price"
            )
            st.markdown('<p class="caption-text">📊 Ratio of sold vs asking price</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with prev_col2:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            previous_price = st.number_input(
                "Previous Period Price ($) - For HPI",
                value=380000,
                step=10000,
                help="Median price from 3-6 months ago"
            )
            st.markdown('<p class="caption-text">📈 Earlier price point for HPI calculation</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        
        col_button = st.columns([1, 1, 1])
        with col_button[1]:
            if st.button("🔮 Generate Prediction", use_container_width=True):
                input_dict = {
                    'date': [pd.to_datetime(date)],
                    'median_sale_price': [median_sale_price],
                    'median_list_price': [median_list_price],
                    'median_ppsf': [median_ppsf],
                    'median_list_ppsf': [median_list_ppsf],
                    'homes_sold': [homes_sold],
                    'inventory': [inventory],
                    'new_listings': [inventory * 0.3], 
                    'pending_sales': [homes_sold * 1.1], 
                    'median_dom': [median_dom],
                    'sold_above_list': [sold_above_list],
                    'off_market_in_two_weeks': [off_market_in_two_weeks],
                    'avg_sale_to_list': [avg_sale_to_list],
        
                    'Total Population': [pop],
                    'Total Housing Units': [units],
                    'Median Age': [35],
                    'Per Capita Income': [income],
                    'Median Commute Time': [25],
        
                    'Total School Age Population': [pop * 0.2],
                    'Total School Enrollment': [pop * 0.18],
                    'Total Labor Force': [pop * 0.5],
                    'Unemployed Population': [pop * 0.02],
                    'Median Rent': [1500],
                    'Median Home Value': [median_sale_price],
                    'Total Families Below Poverty': [units * 0.05],
        
                    'bank': 0, 'bus': 0, 'hospital': 0, 'mall': 0, 'park': 0, 
                    'restaurant': 0, 'school': 0, 'station': 0, 'supermarket': 0
                }
                
                df_input = pd.DataFrame(input_dict)
                
                for lag in [1, 3, 6, 12]:
                    df_input[f'price_lag_{lag}'] = median_sale_price 
                for w in [3, 6, 12]:
                    df_input[f'price_roll_mean_{w}'] = median_sale_price

                df_processed = create_features(df_input)
                
                # Ensure all expected columns exist
                expected_cols = scaler.feature_names_in_
                for col in expected_cols:
                    if col not in df_processed.columns:
                        df_processed[col] = 0  # Add missing columns with zeros

# Keep only expected columns in correct order
                X = df_processed[expected_cols].copy()

# Scale inputs
                X_scaled = scaler.transform(X)

# Force 2D and proper dtype
                X_scaled = np.asarray(X_scaled, dtype=np.float64)
                if X_scaled.ndim == 1:
                    X_scaled = X_scaled.reshape(1, -1)

# Replace NaN or infinite values
                X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

# Predict safely
                try:
                    prediction = model.predict(X_scaled)[0]
                except Exception as e:
                    st.error(f"⚠️ Prediction failed: {e}")
                    prediction = median_sale_price  # fallback to current price 
                hpi_data = calculate_hpi(prediction, previous_price)
                
                st.markdown('<hr class="divider">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">✨ Prediction Results</div>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">Forecasted Price</div>
                        <div class="metric-value">${prediction:,.0f}</div>
                        <div style="font-size: 0.9rem; margin-top: 0.5rem;">
                            {'+' if prediction > median_sale_price else ''} ${prediction - median_sale_price:,.0f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    current_change = ((prediction - median_sale_price) / median_sale_price) * 100
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">Price Change %</div>
                        <div class="metric-value">{current_change:+.2f}%</div>
                        <div style="font-size: 0.9rem; margin-top: 0.5rem;">vs Current</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    if hpi_data:
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">HPI Score</div>
                            <div class="metric-value">{hpi_data['hpi']:.2f}</div>
                            <div style="font-size: 0.9rem; margin-top: 0.5rem;">
                                {hpi_data['price_change_pct']:+.2f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown('<hr class="divider">', unsafe_allow_html=True)
                
                if hpi_data:
                    display_hpi_result(hpi_data)
                    
                    st.markdown('<div class="section-title">📊 Detailed Analysis</div>', unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="info-card">
                            <strong>Previous Period Price:</strong> ${previous_price:,.2f}<br>
                            <strong>Forecasted Price:</strong> ${prediction:,.2f}<br>
                            <strong>Absolute Change:</strong> ${hpi_data['price_change_abs']:+,.2f}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        trend_emoji = "📈" if hpi_data['is_increase'] else "📉"
                        st.markdown(f"""
                        <div class="success-card">
                            <strong>HPI Interpretation:</strong><br>
                            HPI = {hpi_data['hpi']:.2f} (Base: 100)<br>
                            {trend_emoji} Price Movement: <strong>{abs(hpi_data['price_change_pct']):.2f}%</strong><br>
                            {'Market Appreciation' if hpi_data['is_increase'] else 'Market Depreciation'}
                        </div>
                        """, unsafe_allow_html=True)

    elif choice == "Field Guide":
        st.markdown('<div class="section-title">📚 Complete Field Guide</div>', unsafe_allow_html=True)
        
        st.info("Detailed explanation of every field in the prediction tool.")
        
        st.markdown('<div class="field-guide-title">Market Metrics</div>', unsafe_allow_html=True)
        
        with st.expander("💰 Current Median Sale Price"):
            st.markdown("""
            **What it is:** The middle price of homes that actually sold
            
            **Example:** Last month 10 homes sold:
            - $350K, $375K, $380K, $390K, $400K, $410K, $420K, $440K, $450K, $500K
            - Median = $405K
            
            **How to get it:** 
            - Check your local MLS
            - Zillow, Redfin, Realtor.com
            - Contact a local real estate agent
            
            **Why it matters:** Shows actual market price, not asking price
            """)
        
        with st.expander("📌 Median List Price"):
            st.markdown("""
            **What it is:** The middle asking price of homes currently for sale
            
            **How to get it:** Browse listings on Zillow, Realtor.com, or MLS
            
            **Why it matters:** Shows seller expectations vs buyer reality
            """)
        
        with st.expander("⏱️ Median Days on Market (DOM)"):
            st.markdown("""
            **What it is:** How long homes typically sit before selling
            
            **Interpretation:**
            - **< 15 days:** HOT market 🔥
            - **15-30 days:** BALANCED market ⚖️
            - **> 30 days:** SLOW market ❄️
            """)
        
        with st.expander("📐 Price Per Square Foot (PPSF)"):
            st.markdown("""
            **What it is:** Price normalized by home size
            
            **Example:**
            - Home: $400,000 for 2,000 sq ft
            - PPSF = $400,000 ÷ 2,000 = $200/sqft
            
            **Why it matters:** Compare prices fairly regardless of size
            """)
        
        with st.expander("🔥 Homes Sold Above List Price"):
            st.markdown("""
            **What it is:** Count of homes sold for MORE than asking
            
            **Interpretation:**
            - **High:** 🔥 Hot market, high competition
            - **Low:** ❄️ Cold market, buyer advantage
            """)
        
        st.markdown('<div class="field-guide-title">Demographics</div>', unsafe_allow_html=True)
        
        with st.expander("👥 Total Population"):
            st.markdown("""
            **What it is:** Total people in the area
            
            **Why it matters:** Larger populations = more housing demand
            """)
        
        with st.expander("🏘️ Total Housing Units"):
            st.markdown("""
            **What it is:** Total homes/apartments available
            
            **Why it matters:** Calculate supply/demand ratio
            """)
        
        with st.expander("💵 Per Capita Income"):
            st.markdown("""
            **What it is:** Average income per person
            
            **Why it matters:** Higher income = more buying power, higher prices
            """)
        
        st.markdown('<div class="field-guide-title">HPI (House Price Index)</div>', unsafe_allow_html=True)
        
        with st.expander("📈 What is HPI?"):
            st.markdown("""
            **Measures:** How much prices CHANGED over time
            
            **Interpretation:**
            - **HPI = 100:** Price equals base period
            - **HPI > 100:** 📈 Prices increased
            - **HPI < 100:** 📉 Prices decreased
            - **HPI = 110:** 10% increase
            
            **Example:**
            - 6 months ago: $300,000
            - Today: $330,000
            - HPI = 110 = 10% increase
            """)

    elif choice == "Batch Prediction":
        st.markdown('<div class="section-title">📊 Batch Prediction</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV with market data", type=['csv'])
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            data['date'] = pd.to_datetime(data['date'])
            processed = create_features(data)
            
            st.success("✅ Data processed successfully")
            st.dataframe(processed.head(10), use_container_width=True)
            
            csv = processed.to_csv(index=False)
            st.download_button(
                label="Download Processed Data",
                data=csv,
                file_name="processed_predictions.csv",
                mime="text/csv"
            )

    elif choice == "HPI Dashboard":
        st.markdown('<div class="section-title">📈 HPI Market Dashboard</div>', unsafe_allow_html=True)
        
        quarters = ['2023-Q1', '2023-Q2', '2023-Q3', '2023-Q4', 
                   '2024-Q1', '2024-Q2', '2024-Q3', '2024-Q4']
        hpi_values = [95, 98, 102, 105, 108, 110, 112, 115]
        
        df_hpi = pd.DataFrame({
            'Quarter': quarters,
            'HPI': hpi_values,
            'Change %': [0, 3.16, 4.08, 2.94, 2.86, 1.85, 1.82, 2.68]
        })
        
        fig = px.line(
            df_hpi, 
            x='Quarter', 
            y='HPI',
            title='House Price Index Trend Over Time',
            markers=True,
            template='plotly_white',
            line_shape='linear'
        )
        
        fig.add_hline(y=100, line_dash="dash", line_color="#667eea", 
                     annotation_text="Base Index (100)", annotation_position="right")
        
        fig.update_layout(
            hovermode='x unified',
            height=500,
            yaxis_title="HPI Score",
            xaxis_title="Time Period",
            plot_bgcolor='rgba(240, 242, 246, 0.5)',
            paper_bgcolor='rgba(255, 255, 255, 0)',
            font=dict(family="Poppins, sans-serif", size=12, color="#333")
        )
        
        fig.update_traces(line=dict(color='#667eea', width=3))
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Current HPI</div>
                <div class="metric-value">{hpi_values[-1]:.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            change = hpi_values[-1] - hpi_values[0]
            pct_change = (change/hpi_values[0])*100
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Total Change</div>
                <div class="metric-value">{change:+.0f}</div>
                <div style="font-size: 0.9rem;">{pct_change:+.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_hpi = np.mean(hpi_values)
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Average HPI</div>
                <div class="metric-value">{avg_hpi:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            trend = "📈 Uptrend" if hpi_values[-1] > hpi_values[-2] else "📉 Downtrend"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Current Trend</div>
                <div style="font-size: 1.3rem; margin-top: 0.5rem;">{trend}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        
        with st.expander("📚 How to Read HPI"):
            st.markdown("""
            | HPI Value | Meaning | 
            |-----------|---------|
            | **100** | Price = Base period price |
            | **> 100** | 📈 Price increased |
            | **< 100** | 📉 Price decreased |
            | **110** | 10% increase |
            | **90** | 10% decrease |
            """)

if __name__ == "__main__":
    main()