import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# ======================
# Page Configuration
# ======================
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide"
)

# ======================
# Custom CSS Styling
# ======================
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .metric-value {
        font-size: 3.5rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .risk-low {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.4);
    }
    .risk-medium {
        background: linear-gradient(135deg, #F2994A 0%, #F2C94C 100%);
        box-shadow: 0 4px 15px rgba(242, 153, 74, 0.4);
    }
    .risk-high {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        box-shadow: 0 4px 15px rgba(235, 51, 73, 0.4);
    }
    .insight-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    </style>
""", unsafe_allow_html=True)

# ======================
# Data & Model Functions
# ======================
@st.cache_data
def create_synthetic_data():
    """Create a small synthetic dataset for demonstration."""
    data = pd.DataFrame({
        'Tenure': [2, 5, 8, 14, 20, 24, 36, 48, 60, 12],
        'MonthlyCharges': [85, 92, 45, 70, 55, 30, 25, 55, 40, 90],
        'Churn': [1, 1, 0, 1, 0, 0, 0, 0, 0, 1]
    })
    return data

@st.cache_resource
def train_model(_data):
    """Train a Logistic Regression model."""
    X = _data[['Tenure', 'MonthlyCharges']]
    y = _data['Churn']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

def create_gauge_chart(probability):
    """Create a beautiful gauge chart for churn probability."""
    if probability < 0.3:
        color = "#38ef7d"
    elif probability < 0.6:
        color = "#F2C94C"
    else:
        color = "#f45c43"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={'suffix': '%', 'font': {'size': 50, 'color': '#333'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#333"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#ddd",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(56, 239, 125, 0.2)'},
                {'range': [30, 60], 'color': 'rgba(242, 201, 76, 0.2)'},
                {'range': [60, 100], 'color': 'rgba(244, 92, 67, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "#333", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': "Arial"}
    )
    
    return fig

def create_risk_bar_chart(risk_level):
    """Create a horizontal bar chart for risk visualization."""
    levels = ['Low', 'Medium', 'High']
    colors = ['#38ef7d', '#F2C94C', '#f45c43']
    values = [100 if level == risk_level else 20 for level in levels]
    opacities = [1 if level == risk_level else 0.3 for level in levels]
    
    fig = go.Figure()
    
    for i, (level, color, value, opacity) in enumerate(zip(levels, colors, values, opacities)):
        fig.add_trace(go.Bar(
            y=[level],
            x=[value],
            orientation='h',
            marker=dict(
                color=color,
                opacity=opacity,
                line=dict(width=0)
            ),
            name=level,
            showlegend=False
        ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=14)),
        barmode='overlay'
    )
    
    return fig

# ======================
# Load Data & Train Model
# ======================
data = create_synthetic_data()
model, scaler = train_model(data)

# ======================
# Header
# ======================
st.markdown('<h1 class="main-header">📉 Customer Churn Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict customer churn risk using machine learning</p>', unsafe_allow_html=True)

# ======================
# Sidebar
# ======================
with st.sidebar:
    st.markdown("## 🎛️ Customer Profile")
    st.markdown("---")
    
    tenure = st.slider(
        "📅 Tenure (Months)",
        min_value=1,
        max_value=72,
        value=12,
        help="How long has the customer been with us?"
    )
    
    st.markdown("")
    
    monthly_charges = st.slider(
        "💰 Monthly Charges ($)",
        min_value=20,
        max_value=120,
        value=50,
        help="Customer's monthly bill amount"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Training Data")
    st.dataframe(data, use_container_width=True, height=200)

# ======================
# Prediction
# ======================
input_features = scaler.transform([[tenure, monthly_charges]])
churn_probability = model.predict_proba(input_features)[0][1]
stay_probability = 1 - churn_probability

# Determine risk level
if churn_probability < 0.3:
    risk_level = "Low"
    risk_class = "risk-low"
    risk_emoji = "✅"
    risk_message = "This customer appears stable and satisfied."
elif churn_probability < 0.6:
    risk_level = "Medium"
    risk_class = "risk-medium"
    risk_emoji = "⚠️"
    risk_message = "This customer may need some attention."
else:
    risk_level = "High"
    risk_class = "risk-high"
    risk_emoji = "🚨"
    risk_message = "Immediate intervention recommended!"

# ======================
# Main Content
# ======================
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### 🎯 Churn Probability")
    st.plotly_chart(create_gauge_chart(churn_probability), use_container_width=True)

with col2:
    st.markdown("### 📈 Risk Assessment")
    st.plotly_chart(create_risk_bar_chart(risk_level), use_container_width=True)
    
    st.markdown(f"""
        <div class="metric-card {risk_class}">
            <div class="metric-label">Current Risk Level</div>
            <div class="metric-value">{risk_emoji} {risk_level}</div>
        </div>
    """, unsafe_allow_html=True)

# ======================
# Insights Section
# ======================
st.markdown("---")
st.markdown("### 💡 Customer Insights")

col3, col4, col5 = st.columns(3)

with col3:
    st.markdown(f"""
        <div class="insight-box">
            <h4>📅 Tenure Analysis</h4>
            <p><strong>{tenure} months</strong></p>
            <p>{'New customer - higher churn risk' if tenure < 12 else 'Established customer - good retention' if tenure > 24 else 'Moderate tenure - monitor engagement'}</p>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div class="insight-box">
            <h4>💰 Spending Pattern</h4>
            <p><strong>${monthly_charges}/month</strong></p>
            <p>{'High spender - ensure value delivery' if monthly_charges > 70 else 'Budget tier - consider upselling' if monthly_charges < 40 else 'Mid-range - balanced pricing'}</p>
        </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
        <div class="insight-box">
            <h4>🎯 Recommendation</h4>
            <p><strong>{risk_level} Priority</strong></p>
            <p>{risk_message}</p>
        </div>
    """, unsafe_allow_html=True)

# ======================
# Footer
# ======================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #6c757d;'>Built with Streamlit & Scikit-Learn | Model: Logistic Regression</p>",
    unsafe_allow_html=True
)
